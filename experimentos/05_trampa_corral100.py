"""
Experimento 05 - Trampa en CorrAL-100.

Pregunta: ¿Logran los algoritmos discriminar entre las features relevantes (f0-f3),
la feature trampa/correlacionada (f5) y el ruido (f6-f99)?

Se analiza la posición media de cada feature de interés en el ranking,
así como si las 4 features relevantes están todas en el Top-K.

Multi-seed: 5 semillas; se reporta media ± std.

Salidas:
  results/tablas/05_trampa.csv
  results/figuras/05_posiciones_features.png
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

FIG_DIR = 'figuras/05_trampa_corral100'
TAB_DIR = 'tablas/05_trampa_corral100'

SEMILLAS   = CFG['experimento']['semillas']
N_REPLICAS = 50
ALGORITMOS = ['ReliefF', 'ANN', 'Proto']

# Grupos de features a analizar
GRUPOS = {
    'Relevantes (f0-f3)': [0, 1, 2, 3],
    'Trampa/Corr (f5)':   [5],
    'Irrelevante (f4)':   [4],
}


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def crear_selector(algoritmo, semilla):
    if algoritmo == 'ReliefF':
        return ReliefF(n_features_to_select=10, n_neighbors=10)
    if algoritmo == 'ANN':
        return ANN(n_features_to_select=10, n_neighbors=10, random_state=semilla)
    return Proto(n_features_to_select=10, sigma=0.15, k_protos=10, use_lvq=False, n_jobs=1)


def evaluar(algoritmo, X, y, semilla):
    modelo = crear_selector(algoritmo, semilla)
    modelo.fit(X, y)
    ranking = modelo.rank() if hasattr(modelo, 'rank') else np.argsort(-modelo.feature_importances_)
    posiciones = {i: int(np.where(ranking == i)[0][0]) + 1 for i in range(6)}
    precision_top4 = sum(1 for i in [0, 1, 2, 3] if posiciones[i] <= 4) / 4
    precision_top10 = sum(1 for i in [0, 1, 2, 3] if posiciones[i] <= 10) / 4
    return {
        'pos_f0': posiciones[0],
        'pos_f1': posiciones[1],
        'pos_f2': posiciones[2],
        'pos_f3': posiciones[3],
        'pos_f4': posiciones[4],
        'pos_f5': posiciones[5],
        'pos_rel_media': float(np.mean([posiciones[i] for i in [0, 1, 2, 3]])),
        'precision_top4':  round(precision_top4, 4),
        'precision_top10': round(precision_top10, 4),
    }


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def ejecutar():
    filas = []
    for semilla in SEMILLAS:
        X, y = _generar_corral100(semilla=semilla, n_replicas=N_REPLICAS)
        logger.info("Semilla %d: %s, clases=%s", semilla, X.shape, np.bincount(y))
        for alg in ALGORITMOS:
            try:
                metricas = evaluar(alg, X, y, semilla)
                filas.append({'semilla': semilla, 'algoritmo': alg, **metricas})
            except Exception as e:
                logger.warning("Error %s semilla=%d: %s", alg, semilla, e)

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'trampa', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Gráfica
# ---------------------------------------------------------------------------

def graficar(df):
    agrup = df.groupby('algoritmo').agg(
        pos_f0_m=('pos_f0', 'mean'), pos_f0_s=('pos_f0', 'std'),
        pos_f1_m=('pos_f1', 'mean'), pos_f1_s=('pos_f1', 'std'),
        pos_f2_m=('pos_f2', 'mean'), pos_f2_s=('pos_f2', 'std'),
        pos_f3_m=('pos_f3', 'mean'), pos_f3_s=('pos_f3', 'std'),
        pos_f4_m=('pos_f4', 'mean'), pos_f4_s=('pos_f4', 'std'),
        pos_f5_m=('pos_f5', 'mean'), pos_f5_s=('pos_f5', 'std'),
    )

    fig, ax = nueva_figura(tamano=(10, 5))
    x = np.arange(6)
    etiquetas_feat = ['f0\n(relev.)', 'f1\n(relev.)', 'f2\n(relev.)', 'f3\n(relev.)',
                      'f4\n(irrel.)', 'f5\n(corr.)']
    ancho = 0.25

    for i, alg in enumerate(ALGORITMOS):
        if alg not in agrup.index:
            continue
        fila = agrup.loc[alg]
        medias = [fila[f'pos_f{j}_m'] for j in range(6)]
        stds   = [fila[f'pos_f{j}_s'] for j in range(6)]
        offset = (i - 1) * ancho
        barras = ax.bar(x + offset, medias, width=ancho,
                        color=COLORES[alg], alpha=0.85, label=alg)
        ax.errorbar(x + offset, medias, yerr=stds,
                    fmt='none', color='#333333', capsize=3, linewidth=1)

    ax.axhline(y=10.5, color='gray', linestyle='--', linewidth=0.9, alpha=0.6,
               label='Top-10 límite')
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas_feat)
    ax.set_ylabel('Posición media en el ranking (1-based, menor = mejor)')
    ax.set_title('Posición media de features de interés en CorrAL-100\n(media ± std, 5 semillas)')
    ax.invert_yaxis()
    ax.legend()

    for xi, et in zip(x, etiquetas_feat):
        if 'relev' in et:
            ax.axvspan(xi - 0.45, xi + 0.45, color='#3498db', alpha=0.05, zorder=0)
        elif 'irrel' in et:
            ax.axvspan(xi - 0.45, xi + 0.45, color='#e74c3c', alpha=0.05, zorder=0)
        else:
            ax.axvspan(xi - 0.45, xi + 0.45, color='#f39c12', alpha=0.08, zorder=0)

    fig.tight_layout()
    guardar_figura(fig, 'posiciones_features', FIG_DIR)
    logger.info("Figura guardada: 05_posiciones_features.png")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    df = ejecutar()

    logger.info("\nResumen (media sobre 5 semillas):")
    agrup = df.groupby('algoritmo').agg(
        pos_rel_media=('pos_rel_media', 'mean'),
        precision_top4=('precision_top4', 'mean'),
        precision_top10=('precision_top10', 'mean'),
    ).round(3)
    logger.info("\n%s", agrup.to_string())

    graficar(df)
    logger.info("Experimento 05 completado en %.1f min", (time.time() - t0) / 60)
