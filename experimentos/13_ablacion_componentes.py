"""
Experimento 13 - Ablación de componentes ANN-ReliefF.

Pregunta: ¿Cuánto aporta cada componente de ANN-ReliefF a la aceleración
computacional? ¿Se mantiene la calidad de selección en todas las variantes?

Descomposición de contribuciones:
  ReliefF            → baseline sklearn (sin Numba, sin HNSW, sin subsampling)
  ANN_exacto         → + Numba JIT (búsqueda exacta, todas las muestras)
  ANN_hnsw           → + HNSW (Numba + HNSW, todas las muestras)
  ANN_completo       → + subsampling (Numba + HNSW + iter_ratio auto) = ANN estándar

Experimento A — Escalabilidad:
  Dataset sintético n_features=30, n_informative=20.
  n_muestras ∈ [500, 1000, 2000, 5000, 10000, 20000, 30000], 5 semillas.
  Métrica: tiempo fit() medio ± std.

Experimento B — Calidad (F1):
  CorrAL-100 (1600×99), 5-fold CV, 5 semillas.
  Verifica que la aceleración no degrada la selección de features.

Salidas:
  results/tablas/13_ablacion_componentes/ablacion_tiempo.csv
  results/tablas/13_ablacion_componentes/ablacion_f1.csv
  results/figuras/13_ablacion_componentes/tiempo_loglog.png
  results/figuras/13_ablacion_componentes/f1_barras.png
  results/figuras/13_ablacion_componentes/speedup_acumulado.png
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt import ANN
from relieff_opt.utils.conjuntos import _generar_corral100
from relieff_opt.utils.paleta import (
    COLORES, MARCADORES, ESTILOS_LINEA, GROSOR_LINEA,
    nueva_figura, guardar_figura,
)
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/13_ablacion_componentes'
TAB_DIR = 'tablas/13_ablacion_componentes'

SEMILLAS   = CFG['experimento']['semillas']
N_PLIEGUES = CFG['experimento']['cv_pliegues']
N_SEL      = CFG['experimento']['n_features_seleccionadas']

TAMANOS = [500, 1000, 2000, 5000, 10000, 20000, 30000]
N_FEATURES    = 30
N_INFORMATIVE = 20

# Definición de variantes: nombre → kwargs para ANN (o None para ReliefF)
VARIANTES = {
    'ReliefF': None,
    'ANN_exacto':   dict(force_exact=True,  iter_ratio=1.0),
    'ANN_hnsw':     dict(force_exact=False, iter_ratio=1.0,
                         M=CFG['ann']['M'],
                         ef_construction=CFG['ann']['ef_construction'],
                         ef_search=CFG['ann']['ef_search']),
    'ANN_completo': dict(force_exact=False, iter_ratio=CFG['ann']['iter_ratio'],
                         M=CFG['ann']['M'],
                         ef_construction=CFG['ann']['ef_construction'],
                         ef_search=CFG['ann']['ef_search']),
}

ETIQUETAS = {
    'ReliefF':      'ReliefF\n(baseline)',
    'ANN_exacto':   'ANN exacto\n(+ Numba JIT)',
    'ANN_hnsw':     'ANN HNSW\n(+ HNSW)',
    'ANN_completo': 'ANN completo\n(+ subsampling)',
}


# ---------------------------------------------------------------------------
# Construcción de selectores
# ---------------------------------------------------------------------------

def crear_selector(nombre, n_sel, semilla):
    if nombre == 'ReliefF':
        return ReliefF(n_features_to_select=n_sel,
                       n_neighbors=CFG['relieff']['n_neighbors'])
    kwargs = VARIANTES[nombre]
    return ANN(
        n_features_to_select=n_sel,
        n_neighbors=CFG['ann']['n_neighbors'],
        metric=CFG['ann']['metric'],
        random_state=semilla,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Experimento A: Escalabilidad
# ---------------------------------------------------------------------------

def experimento_escalabilidad():
    logger.info("=== Experimento A: Escalabilidad ===")
    filas = []
    for n in TAMANOS:
        logger.info("  n=%d", n)
        for semilla in SEMILLAS:
            X, y = make_classification(
                n_samples=n, n_features=N_FEATURES, n_informative=N_INFORMATIVE,
                n_redundant=5, random_state=semilla,
            )
            X = X.astype(np.float32)
            y = y.astype(np.int32)
            for nombre in VARIANTES:
                sel = crear_selector(nombre, N_SEL, semilla)
                t0 = time.perf_counter()
                sel.fit(X, y)
                t = time.perf_counter() - t0
                filas.append({
                    'variante': nombre,
                    'n_muestras': n,
                    'semilla': semilla,
                    'tiempo_s': round(t, 5),
                })

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'ablacion_tiempo', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Experimento B: Calidad (F1) en CorrAL-100
# ---------------------------------------------------------------------------

def experimento_f1():
    logger.info("=== Experimento B: Calidad F1 en CorrAL-100 ===")
    filas = []
    for semilla in SEMILLAS:
        X, y = _generar_corral100(semilla=semilla, n_replicas=50)
        skf = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=semilla)
        for nombre in VARIANTES:
            f1s = []
            for idx_tr, idx_te in skf.split(X, y):
                Xtr, Xte = X[idx_tr], X[idx_te]
                ytr, yte = y[idx_tr], y[idx_te]
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xtr)
                Xte = sc.transform(Xte)
                try:
                    sel = crear_selector(nombre, N_SEL, semilla)
                    sel.fit(Xtr, ytr)
                    Xtr_s = sel.transform(Xtr)
                    Xte_s = sel.transform(Xte)
                    clf = RandomForestClassifier(
                        n_estimators=CFG['experimento']['rf_n_estimators'],
                        max_depth=CFG['experimento']['rf_max_depth'],
                        random_state=semilla, n_jobs=1,
                    )
                    clf.fit(Xtr_s, ytr)
                    f1s.append(f1_score(yte, clf.predict(Xte_s), average='binary'))
                except Exception as e:
                    logger.warning("    Error %s semilla=%d: %s", nombre, semilla, e)
                    f1s.append(0.0)
            filas.append({
                'variante': nombre,
                'semilla':  semilla,
                'f1_media': round(float(np.mean(f1s)), 4),
                'f1_std':   round(float(np.std(f1s)), 4),
            })
        logger.info("  Semilla %d completada", semilla)

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'ablacion_f1', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def graficar_tiempo(df):
    agrup = df.groupby(['n_muestras', 'variante']).agg(
        media=('tiempo_s', 'mean'),
        std=('tiempo_s', 'std'),
    ).reset_index()

    fig, ax = nueva_figura()
    for nombre in VARIANTES:
        datos = agrup[agrup['variante'] == nombre]
        x = datos['n_muestras'].values
        y = datos['media'].values
        ax.loglog(x, y,
                  color=COLORES.get(nombre, '#888'),
                  marker=MARCADORES.get(nombre, 'o'),
                  linestyle=ESTILOS_LINEA.get(nombre, '-'),
                  linewidth=GROSOR_LINEA, markersize=5,
                  label=ETIQUETAS[nombre].replace('\n', ' '))
    ax.set_xlabel('Número de muestras (log)')
    ax.set_ylabel('Tiempo de fit() [s] (log, media 5 semillas)')
    ax.set_title('Ablación: contribución de cada componente ANN\n(log-log, dataset sintético 30 features)')
    ax.legend(fontsize=9)
    guardar_figura(fig, 'tiempo_loglog', FIG_DIR)


def graficar_f1(df_f1):
    resumen = df_f1.groupby('variante').agg(
        media=('f1_media', 'mean'),
        std=('f1_media', 'std'),
    ).reindex(list(VARIANTES.keys())).reset_index()

    fig, ax = nueva_figura()
    colores_bar = [COLORES.get(v, '#888') for v in resumen['variante']]
    ax.bar(resumen['variante'], resumen['media'],
           color=colores_bar, alpha=0.85,
           yerr=resumen['std'], capsize=4)
    ax.set_xticklabels([ETIQUETAS[v] for v in resumen['variante']], fontsize=9)
    ax.set_ylabel('F1 medio (5-fold CV, 5 semillas)')
    ax.set_ylim(bottom=max(0, resumen['media'].min() - 0.05))
    ax.axhline(resumen[resumen['variante'] == 'ReliefF']['media'].values[0],
               color=COLORES['ReliefF'], linestyle='--', linewidth=1, alpha=0.7,
               label='F1 baseline ReliefF')
    ax.set_title('Ablación: calidad F1 por variante\n(CorrAL-100, 1600×99)')
    ax.legend(fontsize=9)
    guardar_figura(fig, 'f1_barras', FIG_DIR)


def graficar_speedup(df):
    """Speedup acumulado de cada variante vs ReliefF en n=30000."""
    agrup = df.groupby(['n_muestras', 'variante'])['tiempo_s'].mean().unstack()
    n_max = df['n_muestras'].max()
    fila = agrup.loc[n_max]
    ref = fila['ReliefF']

    speedups = {v: ref / fila[v] for v in VARIANTES if v != 'ReliefF'}
    componentes = list(speedups.keys())
    valores     = [speedups[c] for c in componentes]

    fig, ax = nueva_figura()
    colores_bar = [COLORES.get(c, '#888') for c in componentes]
    bars = ax.bar(componentes, valores, color=colores_bar, alpha=0.85)
    ax.set_xticklabels([ETIQUETAS[c] for c in componentes], fontsize=9)
    ax.set_ylabel(f'Speedup vs ReliefF  (n={n_max:,})')
    ax.set_title(f'Speedup acumulado por componente (n={n_max:,}, d={N_FEATURES})\n'
                 'Cada barra incluye todos los componentes anteriores')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'×{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    guardar_figura(fig, 'speedup_acumulado', FIG_DIR)

    logger.info("Speedup en n=%d:", n_max)
    for c, v in speedups.items():
        logger.info("  %s: ×%.1f vs ReliefF", c, v)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()

    df_tiempo = experimento_escalabilidad()
    df_f1     = experimento_f1()

    graficar_tiempo(df_tiempo)
    graficar_f1(df_f1)
    graficar_speedup(df_tiempo)

    logger.info("Resumen F1 (CorrAL-100):")
    resumen_f1 = df_f1.groupby('variante')['f1_media'].agg(['mean', 'std'])
    for v, row in resumen_f1.iterrows():
        logger.info("  %s: %.4f ± %.4f", v, row['mean'], row['std'])

    logger.info("Experimento 13 completado en %.1f min", (time.time() - t0) / 60)
