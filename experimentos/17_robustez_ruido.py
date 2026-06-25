"""
Experimento 17 - Robustez de la selección frente al ruido de etiquetas.

Pregunta: al inyectar ruido en las etiquetas, ¿se degradan HNSW-ReliefF y
Proto-ReliefF MÁS rápido que ReliefF, o al mismo ritmo? Lo que importa no es la
degradación absoluta (todos los selectores empeoran con el ruido), sino si las
propuestas añaden fragilidad respecto al método exacto.

Método: sobre CorrAL-100 (ground truth conocido: 4 variables causales, 1
correlada y 95 de ruido) se invierte una fracción creciente de etiquetas, hasta
el 25%, y se ajustan los tres algoritmos
sobre los MISMOS datos ruidosos. Para cada nivel × semilla se mide:
  - recuperación = fracción de variables relevantes reales que caen en el top-k.
  - delta = recuperación de la propuesta menos la de ReliefF (degradación relativa).
Multi-seed: 5 semillas; se reportan medias.

Salidas:
  results/tablas/17_robustez_ruido/robustez.csv + .tex
  results/figuras/17_robustez_ruido/delta_recuperacion.png   (figura clave)
  results/figuras/17_robustez_ruido/recuperacion_absoluta.png (apoyo)
"""

import logging
import sys
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
from relieff_opt.utils.conjuntos import CATALOGO, obtener_dataset
from relieff_opt.utils.paleta import COLORES, nueva_figura, guardar_figura
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/17_robustez_ruido'
TAB_DIR = 'tablas/17_robustez_ruido'

N_SEL    = CFG['experimento']['n_features_seleccionadas']
SEMILLAS = [42, 123, 456, 789, 1234]
NIVELES  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
DATASETS = ['CorrAL100']

ALGORITMOS = ['ReliefF', 'ANN', 'Proto']
ETIQUETA   = {'ReliefF': 'ReliefF', 'ANN': 'HNSW-ReliefF', 'Proto': 'Proto-ReliefF'}


def pesos(algoritmo, X, y, semilla):
    """Ajusta el algoritmo y devuelve su vector de relevancia."""
    n_sel = min(N_SEL, X.shape[1])
    if algoritmo == 'ReliefF':
        sel = ReliefF(n_features_to_select=n_sel, n_neighbors=CFG['relieff']['n_neighbors'])
    elif algoritmo == 'ANN':
        sel = ANN(
            n_features_to_select=n_sel,
            n_neighbors=CFG['ann']['n_neighbors'],
            metric=CFG['ann']['metric'],
            M=CFG['ann']['M'],
            ef_construction=CFG['ann']['ef_construction'],
            ef_search=CFG['ann']['ef_search'],
            random_state=semilla,
        )
    else:
        sel = Proto(
            n_features_to_select=n_sel,
            k_protos=CFG['proto']['k_protos'],
            sigma=CFG['proto']['sigma'],
            use_lvq=CFG['proto']['use_lvq'],
            metric=CFG['proto']['metric'],
            n_jobs=1,
        )
    sel.fit(X, y)
    return np.asarray(sel.feature_importances_, dtype=float)


def ruido_etiquetas(y, frac, rng):
    """Invierte una fracción de las etiquetas (a otra clase elegida al azar)."""
    y2 = np.array(y).copy()
    n = len(y2)
    n_flip = int(round(frac * n))
    if n_flip == 0:
        return y2
    clases = np.unique(y2)
    idx = rng.choice(n, n_flip, replace=False)
    for i in idx:
        otras = clases[clases != y2[i]]
        y2[i] = rng.choice(otras)
    return y2


def recuperacion(w, relevantes, k):
    """Fracción de las variables relevantes reales que caen en el top-k."""
    top = set(np.argsort(-w)[:k])
    rel = set(int(r) for r in relevantes)
    return len(top & rel) / len(rel)


def ejecutar():
    filas = []
    for nombre in DATASETS:
        if nombre not in CATALOGO:
            logger.warning("Conjunto %s no está en el catálogo, se omite.", nombre)
            continue
        logger.info("Dataset: %s", nombre)
        for semilla in SEMILLAS:
            X, y, relevantes = obtener_dataset(nombre, semilla=semilla)
            if not relevantes:
                logger.warning("  %s sin ground truth de relevantes, se omite.", nombre)
                break
            n_sel = min(N_SEL, X.shape[1])
            for nivel in NIVELES:
                rng = np.random.RandomState(int(semilla * 1000 + nivel * 100))
                y_ruido = ruido_etiquetas(y, nivel, rng)
                rec = {}
                for alg in ALGORITMOS:
                    try:
                        w = pesos(alg, X, y_ruido, semilla)
                        rec[alg] = recuperacion(w, relevantes, n_sel)
                    except Exception as e:
                        logger.warning("  Error %s %s n=%.2f s=%d: %s", alg, nombre, nivel, semilla, e)
                        rec[alg] = np.nan
                filas.append({
                    'dataset': nombre, 'nivel': nivel, 'semilla': semilla,
                    'rec_relief': rec['ReliefF'], 'rec_ann': rec['ANN'], 'rec_proto': rec['Proto'],
                })

    raw = pd.DataFrame(filas)
    # Agregado: media sobre semillas por (dataset, nivel) + deltas relativos
    agg = raw.groupby(['dataset', 'nivel'], as_index=False).mean(numeric_only=True)
    agg = agg.drop(columns=['semilla'])
    agg['delta_ann']   = (agg['rec_ann']   - agg['rec_relief']).round(3)
    agg['delta_proto'] = (agg['rec_proto'] - agg['rec_relief']).round(3)
    for c in ['rec_relief', 'rec_ann', 'rec_proto']:
        agg[c] = agg[c].round(3)
    guardar_tabla(agg, 'robustez', TAB_DIR)
    return agg


def graficar(agg):
    niveles_pct = sorted(agg['nivel'].unique())
    x = [n * 100 for n in niveles_pct]

    # --- Figura clave: delta vs ReliefF ---
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.8 * len(DATASETS) + 0.5, 4.4), sharey=True)
    if len(DATASETS) == 1:
        axes = [axes]
    for ax, nombre in zip(axes, DATASETS):
        sub = agg[agg['dataset'] == nombre].sort_values('nivel')
        ax.axhline(0, color='#888888', linestyle='--', linewidth=1)
        ax.plot(x, sub['delta_ann'].values,   'o-', color=COLORES['ANN'],   label='HNSW-ReliefF $-$ ReliefF')
        ax.plot(x, sub['delta_proto'].values, 's-', color=COLORES['Proto'], label='Proto-ReliefF $-$ ReliefF')
        ax.set_title(nombre)
        ax.set_xlabel('Ruido de etiquetas (%)')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('$\\Delta$ recuperación (vs ReliefF)')
    axes[0].legend(loc='lower left', fontsize=9)
    fig.suptitle('Degradación relativa a ReliefF al aumentar el ruido de etiquetas', y=1.02)
    fig.tight_layout()
    guardar_figura(fig, 'delta_recuperacion', FIG_DIR)

    # --- Figura de apoyo: recuperación absoluta (3 curvas) ---
    fig2, axes2 = plt.subplots(1, len(DATASETS), figsize=(5.8 * len(DATASETS) + 0.5, 4.4), sharey=True)
    if len(DATASETS) == 1:
        axes2 = [axes2]
    for ax, nombre in zip(axes2, DATASETS):
        sub = agg[agg['dataset'] == nombre].sort_values('nivel')
        ax.plot(x, sub['rec_relief'].values, 'o-', color=COLORES['ReliefF'], label='ReliefF')
        ax.plot(x, sub['rec_ann'].values,    's-', color=COLORES['ANN'],     label='HNSW-ReliefF')
        ax.plot(x, sub['rec_proto'].values,  '^-', color=COLORES['Proto'],   label='Proto-ReliefF')
        ax.set_title(nombre)
        ax.set_xlabel('Ruido de etiquetas (%)')
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes2[0].set_ylabel('Recuperación de variables relevantes (top-' + str(N_SEL) + ')')
    axes2[0].legend(loc='lower left', fontsize=9)
    fig2.tight_layout()
    guardar_figura(fig2, 'recuperacion_absoluta', FIG_DIR)


if __name__ == '__main__':
    agg = ejecutar()
    print(agg.to_string(index=False))
    graficar(agg)
    logger.info("Listo.")
