"""
Experimento 01 - Búsqueda de hiperparámetros.

Pregunta: ¿Cuáles son los hiperparámetros óptimos de ANN y Proto considerando
tanto la calidad de selección (F1) como el coste computacional (tiempo de fit)?

Método: grid search exhaustivo con 5-fold stratified CV sobre 12 datasets.
  - ANN:   n_neighbors × M × ef_construction × ef_search (48 combos)
  - Proto: k_protos × sigma × use_lvq (32 combos)

Para cada configuración se mide F1 medio (calidad) y tiempo medio de fit()
(eficiencia). La selección final usa la frontera de Pareto F1 vs tiempo.

Salidas:
  results/tablas/01_busqueda_hiperparametros/grid_ann.csv + .tex
  results/tablas/01_busqueda_hiperparametros/grid_proto.csv + .tex
  results/figuras/01_busqueda_hiperparametros/heatmap_ann_f1.png
  results/figuras/01_busqueda_hiperparametros/heatmap_ann_tiempo.png
  results/figuras/01_busqueda_hiperparametros/pareto_ann.png
  results/figuras/01_busqueda_hiperparametros/heatmap_proto_sin_lvq.png
  results/figuras/01_busqueda_hiperparametros/heatmap_proto_con_lvq.png
  results/figuras/01_busqueda_hiperparametros/pareto_proto.png
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

SEMILLAS   = CFG['experimento']['semillas']
N_PLIEGUES = CFG['experimento']['cv_pliegues']


# ---------------------------------------------------------------------------
# Rejillas de búsqueda
# ---------------------------------------------------------------------------

# ANN emplea distancia euclidiana (L2) dado que hnswlib, la implementación de
# referencia de HNSW, opera nativamente en este espacio métrico. Manhattan (L1)
# no está soportada por hnswlib — intentar usarla provoca un fallback silencioso
# a sklearn brute-force, lo que invalida la comparación de backends.
#
# n_neighbors es un parámetro compartido de ReliefF, fijado como constante
# global (N_NEIGHBORS_GLOBAL) igual para ANN, Proto y ReliefF en todos los
# experimentos. Optimizarlo solo para ANN rompería la comparabilidad entre
# algoritmos. El grid cubre únicamente los parámetros propios de HNSW:
#   M              : conectividad del grafo (calidad de construcción)
#   ef_construction: amplitud del beam en construcción (calidad vs tiempo build)
#   ef_search      : amplitud del beam en búsqueda   (calidad vs tiempo query)
N_NEIGHBORS_GLOBAL = 10   # alineado con skrebate ReliefF y resto de experimentos

REJILLA_ANN = {
    'M':               [8, 16, 32],
    'ef_construction': [100, 200],
    'ef_search':       [50, 200],
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
    """
    5-fold CV con RF como clasificador final.

    Devuelve dict con:
      f1        : F1 binario medio sobre los 5 pliegues
      tiempo_s  : tiempo medio de fit() del selector por pliegue (segundos)
    """
    skf = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=semilla)
    scores, tiempos = [], []
    for idx_tr, idx_te in skf.split(X, y):
        Xtr, Xte = X[idx_tr], X[idx_te]
        ytr, yte = y[idx_tr], y[idx_te]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        try:
            t0 = time.perf_counter()
            selector.fit(Xtr, ytr)
            tiempos.append(time.perf_counter() - t0)
            Xtr_s = selector.transform(Xtr)
            Xte_s = selector.transform(Xte)
            clf = RandomForestClassifier(
                n_estimators=CFG['experimento']['rf_n_estimators'],
                max_depth=CFG['experimento']['rf_max_depth'],
                random_state=semilla, n_jobs=1,
            )
            clf.fit(Xtr_s, ytr)
            scores.append(f1_score(yte, clf.predict(Xte_s), average='binary'))
        except Exception as e:
            logger.debug("Error en fold: %s", e)
            scores.append(0.0)
            tiempos.append(float('nan'))
    return {
        'f1':      float(np.mean(scores)),
        'tiempo_s': float(np.nanmean(tiempos)),
    }


# ---------------------------------------------------------------------------
# Grid search de ANN
# ---------------------------------------------------------------------------

def busqueda_ann(datasets):
    n_combos = (len(REJILLA_ANN['M']) *
                len(REJILLA_ANN['ef_construction']) *
                len(REJILLA_ANN['ef_search']))
    logger.info("Grid search ANN - %d configuraciones × %d datasets × %d semillas "
                "(n_neighbors=%d fijo)", n_combos, len(datasets), len(SEMILLAS), N_NEIGHBORS_GLOBAL)
    filas = []
    combos = list(product(
        REJILLA_ANN['M'],
        REJILLA_ANN['ef_construction'],
        REJILLA_ANN['ef_search'],
    ))
    for m, ef_c, ef_s in combos:
        f1s_ds, tiempos_ds = {}, {}
        for nombre, (X, y, _) in datasets.items():
            n_sel = min(CFG['ann']['n_features_to_select'], X.shape[1])
            f1s, ts = [], []
            for semilla in SEMILLAS:
                selector = ANN(
                    n_neighbors=N_NEIGHBORS_GLOBAL,
                    M=m,
                    ef_construction=ef_c,
                    ef_search=ef_s,
                    n_features_to_select=n_sel,
                    random_state=semilla,
                )
                res = evaluar_configuracion(selector, X, y, semilla)
                f1s.append(res['f1'])
                ts.append(res['tiempo_s'])
            f1s_ds[nombre]     = float(np.mean(f1s))
            tiempos_ds[nombre] = float(np.mean(ts))

        f1_medio     = float(np.mean(list(f1s_ds.values())))
        tiempo_medio = float(np.mean(list(tiempos_ds.values())))
        fila = {
            'M':               m,
            'ef_construction': ef_c,
            'ef_search':       ef_s,
            'f1_medio':        round(f1_medio, 4),
            'tiempo_medio_s':  round(tiempo_medio, 4),
            'f1_por_tiempo':   round(f1_medio / (tiempo_medio + 1e-9), 4),
        }
        fila.update({f'f1_{ds}': round(v, 4) for ds, v in f1s_ds.items()})
        filas.append(fila)
        logger.info("  ANN M=%d ef_c=%d ef_s=%d → f1=%.4f  t=%.3fs",
                    m, ef_c, ef_s, f1_medio, tiempo_medio)

    df = pd.DataFrame(filas).sort_values('f1_medio', ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Grid search de Proto
# ---------------------------------------------------------------------------

def busqueda_proto(datasets):
    n_combos = (len(REJILLA_PROTO['k_protos']) *
                len(REJILLA_PROTO['sigma']) *
                len(REJILLA_PROTO['use_lvq']))
    logger.info("Grid search Proto - %d configuraciones × %d datasets × %d semillas",
                n_combos, len(datasets), len(SEMILLAS))
    filas = []
    combos = list(product(REJILLA_PROTO['k_protos'], REJILLA_PROTO['sigma'], REJILLA_PROTO['use_lvq']))
    for k, sigma, usar_lvq in combos:
        f1s_ds, tiempos_ds = {}, {}
        for nombre, (X, y, _) in datasets.items():
            n_sel = min(CFG['proto']['n_features_to_select'], X.shape[1])
            f1s, ts = [], []
            for semilla in SEMILLAS:
                selector = Proto(
                    k_protos=k,
                    sigma=sigma,
                    use_lvq=usar_lvq,
                    n_features_to_select=n_sel,
                    n_jobs=1,
                )
                res = evaluar_configuracion(selector, X, y, semilla)
                f1s.append(res['f1'])
                ts.append(res['tiempo_s'])
            f1s_ds[nombre]     = float(np.mean(f1s))
            tiempos_ds[nombre] = float(np.mean(ts))

        f1_medio     = float(np.mean(list(f1s_ds.values())))
        tiempo_medio = float(np.mean(list(tiempos_ds.values())))
        fila = {
            'k_protos':       k,
            'sigma':          sigma,
            'use_lvq':        usar_lvq,
            'f1_medio':       round(f1_medio, 4),
            'tiempo_medio_s': round(tiempo_medio, 4),
            'f1_por_tiempo':  round(f1_medio / (tiempo_medio + 1e-9), 4),
        }
        fila.update({f'f1_{ds}': round(v, 4) for ds, v in f1s_ds.items()})
        filas.append(fila)
        logger.info("  Proto k=%d sigma=%.2f lvq=%s → f1=%.4f  t=%.3fs",
                    k, sigma, usar_lvq, f1_medio, tiempo_medio)

    df = pd.DataFrame(filas).sort_values('f1_medio', ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Utilidad: frontera de Pareto
# ---------------------------------------------------------------------------

def _pareto_mask(f1, tiempo):
    """Devuelve máscara booleana True para puntos Pareto-óptimos (max F1, min tiempo)."""
    n = len(f1)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if f1[j] >= f1[i] and tiempo[j] <= tiempo[i] and (f1[j] > f1[i] or tiempo[j] < tiempo[i]):
                pareto[i] = False
                break
    return pareto


# ---------------------------------------------------------------------------
# Gráficas ANN
# ---------------------------------------------------------------------------

def graficar_heatmaps_ann(df):
    """Heatmaps F1 y tiempo: n_neighbors × M, subplots por ef_construction."""
    ef_vals  = sorted(df['ef_construction'].unique())
    ef_s_vals = sorted(df['ef_search'].unique())
    n_cols   = len(ef_vals)

    for metrica, cmap, titulo_met in [
        ('f1_medio',       'Blues',   'F1 medio'),
        ('tiempo_medio_s', 'Oranges', 'Tiempo fit() medio [s]'),
    ]:
        fig, axes = nueva_figura(1, n_cols, tamano=(7 * n_cols, 4.5))
        if n_cols == 1:
            axes = [axes]
        vmin = df[metrica].min()
        vmax = df[metrica].max()
        for ax, ef_c in zip(axes, ef_vals):
            sub = df[df['ef_construction'] == ef_c].copy()
            # Agrega sobre ef_search tomando la media
            pivote = sub.groupby(['ef_search', 'M'])[metrica].mean().reset_index()
            pivote = pivote.pivot(index='ef_search', columns='M', values=metrica)
            sns.heatmap(
                pivote, annot=True,
                fmt='.4f' if 'f1' in metrica else '.3f',
                cmap=cmap, linewidths=0.5, ax=ax,
                vmin=vmin, vmax=vmax,
                cbar_kws={'label': titulo_met},
            )
            ax.set_title(f'ef_construction = {ef_c}', pad=8)
            ax.set_xlabel('M (conectividad HNSW)')
            ax.set_ylabel('ef_search')
        fig.suptitle(f'Grid Search ANN — {titulo_met} (n_neighbors={N_NEIGHBORS_GLOBAL} fijo, 12 datasets, 5 semillas)',
                     y=1.02)
        fig.tight_layout()
        nombre_fig = 'heatmap_ann_f1' if 'f1' in metrica else 'heatmap_ann_tiempo'
        guardar_figura(fig, nombre_fig, FIG_DIR)


def graficar_pareto_ann(df):
    """Scatter F1 vs tiempo con frontera de Pareto y etiqueta de la config óptima."""
    f1     = df['f1_medio'].values
    tiempo = df['tiempo_medio_s'].values
    pareto = _pareto_mask(f1, tiempo)

    fig, ax = nueva_figura(tamano=(8, 5))

    # Todos los puntos
    scatter = ax.scatter(
        tiempo[~pareto], f1[~pareto],
        c='#aab8c2', s=40, alpha=0.6, label='Configuraciones',
    )
    # Pareto-óptimos
    ax.scatter(
        tiempo[pareto], f1[pareto],
        c=COLORES['ANN'], s=80, zorder=5, label='Pareto-óptimos',
        edgecolors='white', linewidths=0.8,
    )

    # Línea de Pareto
    idx_p = np.where(pareto)[0]
    orden = idx_p[np.argsort(tiempo[idx_p])]
    ax.step(tiempo[orden], f1[orden], where='post',
            color=COLORES['ANN'], linewidth=1.2, linestyle='--', alpha=0.7)

    # Etiqueta de la config con mejor F1/tiempo
    mejor = df.loc[df['f1_por_tiempo'].idxmax()]
    ax.annotate(
        f"M={int(mejor['M'])}, ef_c={int(mejor['ef_construction'])}\n"
        f"ef_s={int(mejor['ef_search'])}",
        xy=(mejor['tiempo_medio_s'], mejor['f1_medio']),
        xytext=(10, -25), textcoords='offset points',
        fontsize=8, color=COLORES['ANN'],
        arrowprops=dict(arrowstyle='->', color=COLORES['ANN'], lw=1.0),
    )

    ax.set_xlabel('Tiempo medio de fit() [s]')
    ax.set_ylabel('F1 medio (12 datasets, 5 semillas)')
    ax.set_title('Pareto: calidad vs eficiencia — Grid Search ANN')
    ax.legend()
    fig.tight_layout()
    guardar_figura(fig, 'pareto_ann', FIG_DIR)


# ---------------------------------------------------------------------------
# Gráficas Proto
# ---------------------------------------------------------------------------

def graficar_heatmap_proto(df):
    for usar_lvq in [False, True]:
        sub = df[df['use_lvq'] == usar_lvq]
        sufijo = 'con_lvq' if usar_lvq else 'sin_lvq'

        for metrica, cmap, titulo_met in [
            ('f1_medio',       'Reds',    'F1 medio'),
            ('tiempo_medio_s', 'Oranges', 'Tiempo fit() medio [s]'),
        ]:
            pivote = sub.pivot_table(index='k_protos', columns='sigma', values=metrica)
            fig, ax = nueva_figura()
            sns.heatmap(
                pivote, annot=True,
                fmt='.4f' if 'f1' in metrica else '.3f',
                cmap=cmap, linewidths=0.5, ax=ax,
                cbar_kws={'label': titulo_met},
            )
            ax.set_title(f'Grid Search Proto ({sufijo}) — {titulo_met}\n(12 datasets, 5 semillas)', pad=10)
            ax.set_xlabel('sigma')
            ax.set_ylabel('k_protos')
            nombre_fig = f'heatmap_proto_{sufijo}_{"f1" if "f1" in metrica else "tiempo"}'
            guardar_figura(fig, nombre_fig, FIG_DIR)


def graficar_pareto_proto(df):
    f1     = df['f1_medio'].values
    tiempo = df['tiempo_medio_s'].values
    pareto = _pareto_mask(f1, tiempo)

    fig, ax = nueva_figura(tamano=(8, 5))

    ax.scatter(tiempo[~pareto], f1[~pareto],
               c='#aab8c2', s=40, alpha=0.6, label='Configuraciones')
    ax.scatter(tiempo[pareto], f1[pareto],
               c=COLORES['Proto'], s=80, zorder=5, label='Pareto-óptimos',
               edgecolors='white', linewidths=0.8)

    idx_p = np.where(pareto)[0]
    orden = idx_p[np.argsort(tiempo[idx_p])]
    ax.step(tiempo[orden], f1[orden], where='post',
            color=COLORES['Proto'], linewidth=1.2, linestyle='--', alpha=0.7)

    mejor = df.loc[df['f1_por_tiempo'].idxmax()]
    ax.annotate(
        f"k={int(mejor['k_protos'])}, σ={mejor['sigma']:.2f}\nlvq={mejor['use_lvq']}",
        xy=(mejor['tiempo_medio_s'], mejor['f1_medio']),
        xytext=(10, -25), textcoords='offset points',
        fontsize=8, color=COLORES['Proto'],
        arrowprops=dict(arrowstyle='->', color=COLORES['Proto'], lw=1.0),
    )

    ax.set_xlabel('Tiempo medio de fit() [s]')
    ax.set_ylabel('F1 medio (12 datasets, 5 semillas)')
    ax.set_title('Pareto: calidad vs eficiencia — Grid Search Proto')
    ax.legend()
    fig.tight_layout()
    guardar_figura(fig, 'pareto_proto', FIG_DIR)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    logger.info("Cargando datasets (semilla base=42)...")
    datasets = obtener_todos(semilla=42)

    df_ann = busqueda_ann(datasets)
    guardar_tabla(df_ann, 'grid_ann', TAB_DIR)
    graficar_heatmaps_ann(df_ann)
    graficar_pareto_ann(df_ann)

    df_proto = busqueda_proto(datasets)
    guardar_tabla(df_proto, 'grid_proto', TAB_DIR)
    graficar_heatmap_proto(df_proto)
    graficar_pareto_proto(df_proto)

    logger.info("Experimento 01 completado en %.1f min", (time.time() - t0) / 60)

    mejor_ann = df_ann.iloc[0]
    logger.info("Mejor ANN (F1):         M=%d ef_c=%d ef_s=%d → f1=%.4f  t=%.3fs",
                mejor_ann['M'], mejor_ann['ef_construction'], mejor_ann['ef_search'],
                mejor_ann['f1_medio'], mejor_ann['tiempo_medio_s'])

    pareto_ann = df_ann[_pareto_mask(df_ann['f1_medio'].values, df_ann['tiempo_medio_s'].values)]
    mejor_pareto_ann = pareto_ann.loc[pareto_ann['f1_por_tiempo'].idxmax()]
    logger.info("Mejor ANN (Pareto F1/t): M=%d ef_c=%d ef_s=%d → f1=%.4f  t=%.3fs",
                mejor_pareto_ann['M'], mejor_pareto_ann['ef_construction'],
                mejor_pareto_ann['ef_search'],
                mejor_pareto_ann['f1_medio'], mejor_pareto_ann['tiempo_medio_s'])

    mejor_proto = df_proto.iloc[0]
    logger.info("Mejor Proto (F1):  k=%d sigma=%.2f lvq=%s → f1=%.4f  t=%.3fs",
                mejor_proto['k_protos'], mejor_proto['sigma'],
                mejor_proto['use_lvq'], mejor_proto['f1_medio'], mejor_proto['tiempo_medio_s'])
