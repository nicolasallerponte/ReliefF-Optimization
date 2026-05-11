"""
Experimento 03 - Robustez ante ruido.

Pregunta: ¿Cómo degrada el rendimiento de cada algoritmo ante distintos tipos
de ruido en el dataset CorrAL-100?

Tipos de ruido evaluados:
  - gaussiano  : ruido aditivo N(0, std) sobre las features relevantes
  - etiquetas  : inversión aleatoria de etiquetas (flip_y)
  - features   : corrupción bit-flip de features relevantes
  - missing    : valores faltantes imputados con la media

Método: para cada nivel de ruido × tipo × semilla, se calcula:
  - score_top5_rel_corr: proporción de {f0,f1,f2,f3,f5} entre los Top-5
  - avg_pos_rel: posición media (1-based) de {f0,f1,f2,f3}
Multi-seed: 5 semillas; se reporta media ± std.

Salidas:
  results/tablas/03_robustez.csv
  results/figuras/03_ruido_{tipo}.png  (× 4 tipos)
  results/figuras/03_radar_robustez.png
"""

import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from skrebate import ReliefF

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt import ANN, Proto
from relieff_opt.utils.conjuntos import _generar_corral100
from relieff_opt.utils.paleta import (
    COLORES, MARCADORES, ESTILOS_LINEA, GROSOR_LINEA,
    nueva_figura, guardar_figura,
)
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/03_robustez'
TAB_DIR = 'tablas/03_robustez'

SEMILLAS   = CFG['experimento']['semillas']
N_REPLICAS = 50              # réplicas de CorrAL-100 por semilla
NIVELES    = np.linspace(0, 0.30, 11)
ALGORITMOS = ['ReliefF', 'ANN', 'Proto']

# Configuración de presentación por tipo de perturbación.
# _XLIM_GRAFICA: a partir de qué nivel se corta el eje x (evitar colas extremas).
# _YLIM_RANKING_MAX: rango del eje y del ranking medio; un rango más amplio evita
# que diferencias de 1 posición parezcan exageradas.
_XLIM_GRAFICA     = {'gaussiano': 0.25, 'etiquetas': 0.25, 'features': 0.25, 'missing': 0.30}
_YLIM_RANKING_MAX = {'missing': 8}

# Features a evaluar en CorrAL-100
RELEVANTES = [0, 1, 2, 3]
OBJETIVO   = [0, 1, 2, 3, 5]   # 4 relevantes + correlacionada


# ---------------------------------------------------------------------------
# Perturbaciones de ruido
# ---------------------------------------------------------------------------

def ruido_gaussiano(X, std, rng):
    if std == 0:
        return X.copy()
    X2 = X.copy()
    X2[:, :4] += rng.normal(0, std, size=(X.shape[0], 4))
    return X2

def ruido_etiquetas(y, prob, rng):
    if prob == 0:
        return y.copy()
    y2 = y.copy()
    idx = rng.choice(len(y), int(prob * len(y)), replace=False)
    y2[idx] = 1 - y2[idx]
    return y2

def ruido_features(X, prob, rng):
    if prob == 0:
        return X.copy()
    X2 = X.copy()
    for i in range(4):
        mascara = rng.random(X.shape[0]) < prob
        X2[mascara, i] = 1 - X2[mascara, i]
    return X2

def missing_values(X, prob, rng):
    if prob == 0:
        return X.copy()
    X2 = X.copy()
    for i in range(4):
        mascara = rng.random(X.shape[0]) < prob
        X2[mascara, i] = np.nan
        col = X2[:, i]
        X2[np.isnan(col), i] = np.nanmean(col)
    return X2

PERTURBACIONES = {
    'gaussiano': ruido_gaussiano,
    'etiquetas': lambda X, y, nivel, rng: (X.copy(), ruido_etiquetas(y, nivel, rng)),
    'features':  lambda X, y, nivel, rng: (ruido_features(X, nivel, rng), y.copy()),
    'missing':   lambda X, y, nivel, rng: (missing_values(X, nivel, rng), y.copy()),
}


# ---------------------------------------------------------------------------
# Evaluación de un método
# ---------------------------------------------------------------------------

def crear_selector(algoritmo, semilla):
    n_sel = CFG['experimento']['n_features_seleccionadas']
    if algoritmo == 'ReliefF':
        return ReliefF(n_features_to_select=n_sel, n_neighbors=CFG['relieff']['n_neighbors'])
    if algoritmo == 'ANN':
        return ANN(
            n_features_to_select=n_sel,
            n_neighbors=CFG['ann']['n_neighbors'],
            metric=CFG['ann']['metric'],
            M=CFG['ann']['M'],
            ef_construction=CFG['ann']['ef_construction'],
            ef_search=CFG['ann']['ef_search'],
            random_state=semilla,
        )
    if algoritmo == 'Proto':
        return Proto(
            n_features_to_select=n_sel,
            k_protos=CFG['proto']['k_protos'],
            sigma=CFG['proto']['sigma'],
            use_lvq=CFG['proto']['use_lvq'],
            metric=CFG['proto']['metric'],
            n_jobs=1,
        )
    raise ValueError(f"Algoritmo desconocido: {algoritmo}")


def evaluar(algoritmo, X, y, semilla):
    modelo = crear_selector(algoritmo, semilla)
    try:
        modelo.fit(X, y)
        ranking = modelo.rank() if hasattr(modelo, 'rank') else np.argsort(-modelo.feature_importances_)
        posiciones = {i: int(np.where(ranking == i)[0][0]) + 1 for i in range(6)}
        n_top5     = sum(1 for i in OBJETIVO if posiciones[i] <= 5)
        avg_pos    = float(np.mean([posiciones[i] for i in RELEVANTES]))
        return {
            'score_top5':    n_top5 / len(OBJETIVO),
            'avg_pos_rel':   avg_pos,
            'n_top5':        n_top5,
        }
    except Exception as e:
        logger.warning("  Error %s semilla=%d: %s", algoritmo, semilla, e)
        return {'score_top5': 0.0, 'avg_pos_rel': float(len(ranking)), 'n_top5': 0}


# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------

def ejecutar():
    filas = []
    for tipo in PERTURBACIONES:
        logger.info("Tipo de ruido: %s", tipo)
        for nivel in NIVELES:
            for semilla in SEMILLAS:
                rng = np.random.RandomState(semilla + int(nivel * 1000))
                X_base, y_base = _generar_corral100(semilla=semilla, n_replicas=N_REPLICAS)

                if tipo == 'gaussiano':
                    X, y = ruido_gaussiano(X_base, nivel, rng), y_base.copy()
                elif tipo == 'etiquetas':
                    X, y = X_base.copy(), ruido_etiquetas(y_base, nivel, rng)
                elif tipo == 'features':
                    X, y = ruido_features(X_base, nivel, rng), y_base.copy()
                else:
                    X, y = missing_values(X_base, nivel, rng), y_base.copy()

                for alg in ALGORITMOS:
                    metricas = evaluar(alg, X, y, semilla)
                    filas.append({
                        'tipo':       tipo,
                        'nivel':      round(nivel, 4),
                        'semilla':    semilla,
                        'algoritmo':  alg,
                        **metricas,
                    })

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'robustez', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Gráficas por tipo de ruido
# ---------------------------------------------------------------------------

def graficar_robustez(df):
    for tipo in df['tipo'].unique():
        xlim_max = _XLIM_GRAFICA.get(tipo, 0.30)
        sub = df[(df['tipo'] == tipo) & (df['nivel'] <= xlim_max)]
        agrup = sub.groupby(['nivel', 'algoritmo']).agg(
            score_media=('score_top5', 'mean'),
            score_std=('score_top5', 'std'),
            pos_media=('avg_pos_rel', 'mean'),
            pos_std=('avg_pos_rel', 'std'),
        ).reset_index()

        fig, (ax1, ax2) = nueva_figura(1, 2, tamano=(12, 4.5))

        for alg in ALGORITMOS:
            datos = agrup[agrup['algoritmo'] == alg]
            x = datos['nivel']
            ax1.plot(x, datos['score_media'],
                     color=COLORES[alg], marker=MARCADORES[alg],
                     linestyle=ESTILOS_LINEA[alg], linewidth=GROSOR_LINEA,
                     markersize=5, label=alg)
            ax1.fill_between(x,
                             datos['score_media'] - datos['score_std'],
                             datos['score_media'] + datos['score_std'],
                             color=COLORES[alg], alpha=0.15)

            ax2.plot(x, datos['pos_media'],
                     color=COLORES[alg], marker=MARCADORES[alg],
                     linestyle=ESTILOS_LINEA[alg], linewidth=GROSOR_LINEA,
                     markersize=5, label=alg)
            ax2.fill_between(x,
                             datos['pos_media'] - datos['pos_std'],
                             datos['pos_media'] + datos['pos_std'],
                             color=COLORES[alg], alpha=0.15)

        tipo_es = {'gaussiano': 'Gaussiano', 'etiquetas': 'Etiquetas',
                   'features': 'Features', 'missing': 'Valores faltantes'}
        ax1.set_xlabel('Nivel de ruido')
        ax1.set_ylabel('Score Top-5 (media ± std, 5 semillas)')
        ax1.set_title(f'Robustez ante ruido {tipo_es.get(tipo, tipo)}\n(banda = ±std sobre 5 semillas)')
        ax1.set_ylim(-0.05, 1.05)
        ax1.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.legend()

        ax2.set_xlabel('Nivel de ruido')
        ax2.set_ylabel('Posición media features relevantes (1-based)')
        ax2.set_title(f'Ranking de features relevantes\nante ruido {tipo_es.get(tipo, tipo)}')
        ylim_rank_max = _YLIM_RANKING_MAX.get(tipo, None)
        if ylim_rank_max is not None:
            ax2.set_ylim(0.5, ylim_rank_max)
        ax2.invert_yaxis()
        ax2.legend()

        fig.tight_layout()
        guardar_figura(fig, f'ruido_{tipo}', FIG_DIR)
        logger.info("  Figura guardada: 03_ruido_%s.png", tipo)


def graficar_radar(df):
    """Radar chart con las 4 dimensiones de robustez (nivel máximo de ruido)."""
    nivel_max = df['nivel'].max()
    sub = df[df['nivel'] == nivel_max]
    agrup = sub.groupby('algoritmo')['score_top5'].mean()

    etiquetas = ['Gaussiano', 'Etiquetas', 'Features', 'Missing']
    tipos = ['gaussiano', 'etiquetas', 'features', 'missing']
    n = len(tipos)
    angulos = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')

    for alg in ALGORITMOS:
        valores = [
            df[(df['tipo'] == t) & (df['algoritmo'] == alg)]['score_top5'].mean()
            for t in tipos
        ]
        valores += valores[:1]
        ax.plot(angulos, valores,
                color=COLORES[alg], linewidth=GROSOR_LINEA,
                linestyle=ESTILOS_LINEA[alg], label=alg)
        ax.fill(angulos, valores, color=COLORES[alg], alpha=0.10)

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(etiquetas, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=8)
    ax.set_title('Radar de robustez - CorrAL-100\n(score Top-5 medio, 5 semillas)',
                 pad=15, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    guardar_figura(fig, 'radar_robustez', FIG_DIR)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    df = ejecutar()
    graficar_robustez(df)
    graficar_radar(df)
    logger.info("Experimento 03 completado en %.1f min", (time.time() - t0) / 60)
