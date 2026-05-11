"""
Experimento 07 - Convergencia del iter_ratio.

Pregunta: ¿Qué fracción de las muestras necesita ANN como queries para obtener
pesos similares a los del conjunto completo?

Método: CorrAL-100; se calcula la correlación de Spearman entre los pesos
obtenidos con iter_ratio=r y los pesos obtenidos con iter_ratio=1.0 (referencia).
Se evalúa r ∈ [0.05, 1.0] con 5 semillas.

NOTA: Ambos modelos usan force_exact=True para aislar el efecto del subsampling
de la estocasticidad propia del índice HNSW (que introduce varianza adicional
incluso entre dos ejecuciones con iter_ratio=1.0).

Salidas:
  results/tablas/07_convergencia_iter_ratio/iter_ratio.csv
  results/figuras/07_convergencia_iter_ratio/convergencia.png
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
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt import ANN
from relieff_opt.utils.conjuntos import _generar_corral100
from relieff_opt.utils.paleta import (
    COLORES, GROSOR_LINEA, nueva_figura, guardar_figura,
)
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/07_convergencia_iter_ratio'
TAB_DIR = 'tablas/07_convergencia_iter_ratio'

SEMILLAS   = CFG['experimento']['semillas']
N_REPLICAS = 50
RATIOS     = np.round(np.linspace(0.05, 1.0, 20), 3)


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def evaluar_ratio(X, y, ratio, semilla):
    """Correlación Spearman entre pesos con iter_ratio=ratio vs iter_ratio=1.0.

    force_exact=True en ambos modelos para eliminar la estocasticidad HNSW
    y medir solo el efecto del subsampling.
    """
    params_base = dict(
        n_features_to_select=CFG['experimento']['n_features_seleccionadas'],
        n_neighbors=CFG['ann']['n_neighbors'],
        metric=CFG['ann']['metric'],
        force_exact=True,   # búsqueda exacta: aísla iter_ratio del ruido HNSW
        random_state=semilla,
    )
    modelo_ref = ANN(iter_ratio=1.0, **params_base)
    modelo_ref.fit(X, y)
    pesos_ref = modelo_ref.feature_importances_

    modelo = ANN(iter_ratio=ratio, **params_base)
    modelo.fit(X, y)
    pesos = modelo.feature_importances_

    corr, _ = spearmanr(pesos_ref, pesos)
    return float(corr)


def ejecutar():
    filas = []
    for semilla in SEMILLAS:
        X, y = _generar_corral100(semilla=semilla, n_replicas=N_REPLICAS)
        logger.info("Semilla %d: %s", semilla, X.shape)
        for ratio in RATIOS:
            corr = evaluar_ratio(X, y, float(ratio), semilla)
            filas.append({'semilla': semilla, 'iter_ratio': ratio, 'spearman_r': round(corr, 4)})

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'iter_ratio', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Gráfica
# ---------------------------------------------------------------------------

def graficar(df):
    agrup = df.groupby('iter_ratio').agg(
        media=('spearman_r', 'mean'),
        std=('spearman_r', 'std'),
    ).reset_index()

    fig, ax = nueva_figura()
    ax.plot(agrup['iter_ratio'], agrup['media'],
            color=COLORES['ANN'], linewidth=GROSOR_LINEA,
            marker='o', markersize=5, label='ANN (correlación con ratio=1.0)')
    ax.fill_between(
        agrup['iter_ratio'],
        agrup['media'] - agrup['std'],
        agrup['media'] + agrup['std'],
        color=COLORES['ANN'], alpha=0.15,
    )

    # Líneas de referencia
    ax.axhline(0.95, color='gray', linestyle='--', linewidth=0.9, alpha=0.6, label='r = 0.95')
    ax.axhline(0.99, color='gray', linestyle=':',  linewidth=0.9, alpha=0.6, label='r = 0.99')

    ax.set_xlabel('iter_ratio (fracción de muestras usadas como query)')
    ax.set_ylabel('Correlación de Spearman con pesos completos (media ± std)')
    ax.set_title('Convergencia de los pesos ANN según iter_ratio\n'
                 '(búsqueda exacta; 5 semillas, CorrAL-100)')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    guardar_figura(fig, 'convergencia', FIG_DIR)
    logger.info("Figura guardada: 07_convergencia.png")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    df = ejecutar()
    graficar(df)

    agrup = df.groupby('iter_ratio')['spearman_r'].mean()
    umbral_095 = agrup[agrup >= 0.95].index.min()
    umbral_099 = agrup[agrup >= 0.99].index.min()
    logger.info("iter_ratio mínimo para r≥0.95: %.3f", umbral_095)
    logger.info("iter_ratio mínimo para r≥0.99: %.3f", umbral_099)
    logger.info("Experimento 07 completado en %.1f min", (time.time() - t0) / 60)
