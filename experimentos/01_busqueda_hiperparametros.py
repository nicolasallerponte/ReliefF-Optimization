"""
Experimento 01 - Búsqueda de hiperparámetros.

Pregunta: ¿Cuáles son los hiperparámetros óptimos de ANN y Proto?

Método: grid search exhaustivo con 5-fold stratified CV sobre 12 datasets.
  - ANN:   n_neighbors × metric (rejilla reducida, representativa)
  - Proto: k_protos × sigma × use_lvq

Salidas:
  results/tablas/01_grid_ann.csv
  results/tablas/01_grid_proto.csv
  results/figuras/01_heatmap_ann.png
  results/figuras/01_heatmap_proto.png
"""

import logging
import time
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Configurar sys.path para importar el paquete
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt import ANN, Proto
from relieff_opt.utils.conjuntos import obtener_todos
from relieff_opt.utils.paleta import (
    COLORES, nueva_figura, guardar_figura, estilo_academico,
)
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/01_busqueda_hiperparametros'
TAB_DIR = 'tablas/01_busqueda_hiperparametros'

SEMILLAS = CFG['experimento']['semillas']
N_PLIEGUES = CFG['experimento']['cv_pliegues']


# ---------------------------------------------------------------------------
# Rejillas de búsqueda
# ---------------------------------------------------------------------------

# ANN emplea distancia euclidiana (L2) dado que hnswlib, la implementación de
# referencia de HNSW, opera nativamente en este espacio métrico. Manhattan (L1)
# no está soportada por hnswlib — intentar usarla provoca un fallback silencioso
# a sklearn brute-force, lo que invalida la comparación de backends.
# El grid incluye M y ef_construction, parámetros propios de HNSW que controlan
# la calidad del grafo de navegación y el equilibrio calidad/velocidad.
REJILLA_ANN = {
    'n_neighbors':     [5, 10, 15, 20],
    'M':               [8, 16, 32],
    'ef_construction': [100, 200],
}

REJILLA_PROTO = {
    'k_protos':  [3, 5, 10, 15],
    'sigma':     [0.10, 0.15, 0.20, 0.25],
    'use_lvq':   [False, True],
}


# ---------------------------------------------------------------------------
# Evaluación de una configuración
# ---------------------------------------------------------------------------

def evaluar_configuracion(selector, X, y, semilla):
    """5-fold CV; devuelve F1 medio sobre los pliegues."""
    skf = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=semilla)
    scores = []
    for idx_tr, idx_te in skf.split(X, y):
        Xtr, Xte = X[idx_tr], X[idx_te]
        ytr, yte = y[idx_tr], y[idx_te]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        try:
            selector.fit(Xtr, ytr)
            Xtr_s = selector.transform(Xtr)
            Xte_s = selector.transform(Xte)
            clf = RandomForestClassifier(
                n_estimators=CFG['experimento']['rf_n_estimators'],
                max_depth=CFG['experimento']['rf_max_depth'],
                random_state=semilla, n_jobs=1,
            )
            clf.fit(Xtr_s, ytr)
            scores.append(f1_score(yte, clf.predict(Xte_s), average='binary'))
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Grid search de ANN
# ---------------------------------------------------------------------------

def busqueda_ann(datasets):
    n_combos = (len(REJILLA_ANN['n_neighbors']) *
                len(REJILLA_ANN['M']) *
                len(REJILLA_ANN['ef_construction']))
    logger.info("Grid search ANN - %d configuraciones × %d datasets × %d semillas",
                n_combos, len(datasets), len(SEMILLAS))
    filas = []
    combos = list(product(
        REJILLA_ANN['n_neighbors'],
        REJILLA_ANN['M'],
        REJILLA_ANN['ef_construction'],
    ))
    for n_vecs, m, ef in combos:
        f1_por_dataset = {}
        for nombre, (X, y, _) in datasets.items():
            n_sel = min(CFG['ann']['n_features_to_select'], X.shape[1])
            f1s = []
            for semilla in SEMILLAS:
                selector = ANN(
                    n_neighbors=n_vecs,
                    M=m,
                    ef_construction=ef,
                    n_features_to_select=n_sel,
                    random_state=semilla,
                )
                f1s.append(evaluar_configuracion(selector, X, y, semilla))
            f1_por_dataset[nombre] = float(np.mean(f1s))
        fila = {
            'n_neighbors':     n_vecs,
            'M':               m,
            'ef_construction': ef,
            'f1_medio':        float(np.mean(list(f1_por_dataset.values()))),
        }
        fila.update({f'f1_{ds}': v for ds, v in f1_por_dataset.items()})
        filas.append(fila)
        logger.info("  ANN k=%d M=%d ef=%d → f1_medio=%.4f", n_vecs, m, ef, fila['f1_medio'])

    return pd.DataFrame(filas).sort_values('f1_medio', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Grid search de Proto
# ---------------------------------------------------------------------------

def busqueda_proto(datasets):
    logger.info("Grid search Proto - %d configuraciones × %d datasets × %d semillas",
                len(REJILLA_PROTO['k_protos']) * len(REJILLA_PROTO['sigma']) * len(REJILLA_PROTO['use_lvq']),
                len(datasets), len(SEMILLAS))
    filas = []
    combos = list(product(REJILLA_PROTO['k_protos'], REJILLA_PROTO['sigma'], REJILLA_PROTO['use_lvq']))
    for k, sigma, usar_lvq in combos:
        f1_por_dataset = {}
        for nombre, (X, y, _) in datasets.items():
            n_sel = min(CFG['proto']['n_features_to_select'], X.shape[1])
            f1s = []
            for semilla in SEMILLAS:
                selector = Proto(
                    k_protos=k,
                    sigma=sigma,
                    use_lvq=usar_lvq,
                    n_features_to_select=n_sel,
                    n_jobs=1,
                )
                f1s.append(evaluar_configuracion(selector, X, y, semilla))
            f1_por_dataset[nombre] = float(np.mean(f1s))
        fila = {
            'k_protos':  k,
            'sigma':     sigma,
            'use_lvq':   usar_lvq,
            'f1_medio':  float(np.mean(list(f1_por_dataset.values()))),
        }
        fila.update({f'f1_{ds}': v for ds, v in f1_por_dataset.items()})
        filas.append(fila)
        logger.info("  Proto k=%d sigma=%.2f lvq=%s → f1_medio=%.4f",
                    k, sigma, usar_lvq, fila['f1_medio'])

    return pd.DataFrame(filas).sort_values('f1_medio', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def graficar_heatmap_ann(df):
    """Un heatmap n_neighbors × M por cada valor de ef_construction."""
    ef_vals = sorted(df['ef_construction'].unique())
    fig, axes = nueva_figura(1, len(ef_vals), tamano=(7 * len(ef_vals), 4.5))
    if len(ef_vals) == 1:
        axes = [axes]
    vmin = df['f1_medio'].min()
    vmax = df['f1_medio'].max()
    for ax, ef in zip(axes, ef_vals):
        sub = df[df['ef_construction'] == ef]
        pivote = sub.pivot_table(index='n_neighbors', columns='M', values='f1_medio')
        sns.heatmap(
            pivote, annot=True, fmt='.4f', cmap='Blues',
            linewidths=0.5, ax=ax, vmin=vmin, vmax=vmax,
            cbar_kws={'label': 'F1 medio'},
        )
        ax.set_title(f'ef_construction = {ef}', pad=8)
        ax.set_xlabel('M (conectividad HNSW)')
        ax.set_ylabel('n_neighbors')
    fig.suptitle('Grid Search ANN — F1 medio (12 datasets, 5 semillas)', y=1.02)
    fig.tight_layout()
    guardar_figura(fig, 'heatmap_ann', FIG_DIR)


def graficar_heatmap_proto(df):
    for usar_lvq in [False, True]:
        sub = df[df['use_lvq'] == usar_lvq]
        pivote = sub.pivot_table(index='k_protos', columns='sigma', values='f1_medio')
        fig, ax = nueva_figura()
        sns.heatmap(
            pivote, annot=True, fmt='.4f', cmap='Reds',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'F1 medio'},
        )
        sufijo = 'con_lvq' if usar_lvq else 'sin_lvq'
        ax.set_title(f'Grid Search Proto ({sufijo}) - F1 medio (12 datasets, 5 semillas)', pad=10)
        ax.set_xlabel('sigma')
        ax.set_ylabel('k_protos')
        guardar_figura(fig, f'heatmap_proto_{sufijo}', FIG_DIR)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    logger.info("Cargando datasets (semilla base=42)...")
    datasets = obtener_todos(semilla=42)

    df_ann = busqueda_ann(datasets)
    guardar_tabla(df_ann, 'grid_ann', TAB_DIR)
    graficar_heatmap_ann(df_ann)

    df_proto = busqueda_proto(datasets)
    guardar_tabla(df_proto, 'grid_proto', TAB_DIR)
    graficar_heatmap_proto(df_proto)

    logger.info("Experimento 01 completado en %.1f min", (time.time() - t0) / 60)
    logger.info("Mejor ANN:   %s", df_ann.iloc[0][['n_neighbors', 'metric', 'f1_medio']].to_dict())
    logger.info("Mejor Proto: %s", df_proto.iloc[0][['k_protos', 'sigma', 'use_lvq', 'f1_medio']].to_dict())
