"""
Experimento 11 - Crossover HNSW vs Búsqueda Exacta.

Pregunta: ¿La aproximación HNSW preserva la calidad de selección a todos los
tamaños de dataset? ¿A partir de qué n HNSW supera en velocidad a la búsqueda
exacta?

Método: Se comparan dos variantes de ANN sobre un dataset sintético variando
n_muestras de 200 a 30 000:
  - ANN_hnsw  : HNSW siempre activo (force_exact=False) — el algoritmo propuesto
  - ANN_exacto: sklearn brute-force  (force_exact=True)  — la búsqueda exacta

Para cada tamaño × semilla se mide:
  - F1 (5-fold stratified CV) → ¿la aproximación HNSW preserva calidad?
  - Tiempo de fit()           → ¿cuál es el punto de cruce computacional?

Multi-seed: 5 semillas; se reporta media ± std.

Salidas:
  results/tablas/11_crossover/crossover.csv + .tex
  results/figuras/11_crossover/f1_vs_n.png
  results/figuras/11_crossover/tiempo_vs_n.png
"""

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt import ANN
from relieff_opt.utils.paleta import (
    COLORES, MARCADORES, ESTILOS_LINEA, GROSOR_LINEA,
    nueva_figura, guardar_figura,
)
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/11_crossover'
TAB_DIR = 'tablas/11_crossover'

SEMILLAS   = CFG['experimento']['semillas']
TAMANOS    = [200, 500, 1000, 2000, 5000, 10000, 20000, 30000]
N_FEATURES = 30
N_INFORMATIVE = 20
N_CV = 5

VARIANTES = {
    'ANN_hnsw':   {'color': COLORES['ANN'],   'marker': MARCADORES['ANN'],   'linestyle': ESTILOS_LINEA['ANN'],   'label': 'ANN (HNSW)'},
    'ANN_exacto': {'color': '#e67e22',         'marker': 'D',                 'linestyle': '--',                   'label': 'ANN (exacto)'},
}


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def _f1_cv(modelo, X, y, semilla):
    """Calcula F1 binario con 5-fold CV usando KNN como clasificador final."""
    from sklearn.metrics import f1_score
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=semilla)
    scores = []
    for idx_train, idx_test in skf.split(X, y):
        X_tr, X_te = X[idx_train], X[idx_test]
        y_tr, y_te = y[idx_train], y[idx_test]
        modelo.fit(X_tr, y_tr)
        X_tr_sel = modelo.transform(X_tr)
        X_te_sel = modelo.transform(X_te)
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_tr_sel, y_tr)
        y_pred = clf.predict(X_te_sel)
        scores.append(f1_score(y_te, y_pred, average='binary'))
    return float(np.mean(scores))


def ejecutar():
    filas = []
    for n in TAMANOS:
        logger.info("n_muestras=%d", n)
        for semilla in SEMILLAS:
            X, y = make_classification(
                n_samples=n, n_features=N_FEATURES, n_informative=N_INFORMATIVE,
                n_redundant=5, random_state=semilla,
            )
            X = X.astype(np.float32)
            y = y.astype(np.int32)

            params_base = dict(
                n_features_to_select=CFG['experimento']['n_features_seleccionadas'],
                n_neighbors=CFG['ann']['n_neighbors'],
                metric=CFG['ann']['metric'],
                M=CFG['ann']['M'],
                ef_construction=CFG['ann']['ef_construction'],
                ef_search=CFG['ann']['ef_search'],
                random_state=semilla,
            )
            for nombre, force_exact in [('ANN_hnsw', False), ('ANN_exacto', True)]:
                modelo = ANN(force_exact=force_exact, **params_base)

                t0 = time.perf_counter()
                modelo.fit(X, y)
                tiempo = time.perf_counter() - t0

                f1 = _f1_cv(
                    ANN(force_exact=force_exact, **params_base),
                    X, y, semilla,
                )

                filas.append({
                    'n_muestras': n,
                    'variante':   nombre,
                    'semilla':    semilla,
                    'f1':         round(f1, 4),
                    'tiempo_s':   round(tiempo, 4),
                })

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'crossover', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def graficar(df):
    agrup = df.groupby(['n_muestras', 'variante']).agg(
        f1_media=('f1', 'mean'),
        f1_std=('f1', 'std'),
        t_media=('tiempo_s', 'mean'),
        t_std=('tiempo_s', 'std'),
    ).reset_index()

    # --- F1 vs n ---
    fig, ax = nueva_figura()
    for nombre, estilo in VARIANTES.items():
        d = agrup[agrup['variante'] == nombre]
        ax.plot(d['n_muestras'], d['f1_media'],
                color=estilo['color'], marker=estilo['marker'],
                linestyle=estilo['linestyle'], linewidth=GROSOR_LINEA,
                markersize=5, label=estilo['label'])
        ax.fill_between(d['n_muestras'],
                        d['f1_media'] - d['f1_std'],
                        d['f1_media'] + d['f1_std'],
                        color=estilo['color'], alpha=0.15)
    ax.set_xlabel('Número de muestras')
    ax.set_ylabel('F1 (media ± std, 5-fold CV, 5 semillas)')
    ax.set_title('Calidad de selección: HNSW vs búsqueda exacta')
    ax.legend()
    guardar_figura(fig, 'f1_vs_n', FIG_DIR)

    # --- Tiempo vs n (log-log) ---
    fig, ax = nueva_figura()
    for nombre, estilo in VARIANTES.items():
        d = agrup[agrup['variante'] == nombre]
        ax.loglog(d['n_muestras'], d['t_media'],
                  color=estilo['color'], marker=estilo['marker'],
                  linestyle=estilo['linestyle'], linewidth=GROSOR_LINEA,
                  markersize=5, label=estilo['label'])
        ax.fill_between(d['n_muestras'],
                        np.maximum(d['t_media'] - d['t_std'], 1e-6),
                        d['t_media'] + d['t_std'],
                        color=estilo['color'], alpha=0.15)
    ax.set_xlabel('Número de muestras (log)')
    ax.set_ylabel('Tiempo de fit() [s] (log, media ± std, 5 semillas)')
    ax.set_title('Coste computacional: HNSW vs búsqueda exacta')
    ax.legend()
    guardar_figura(fig, 'tiempo_vs_n', FIG_DIR)
    logger.info("Figuras guardadas: f1_vs_n.png, tiempo_vs_n.png")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    df = ejecutar()
    graficar(df)
    logger.info("Experimento 11 completado en %.1f min", (time.time() - t0) / 60)
