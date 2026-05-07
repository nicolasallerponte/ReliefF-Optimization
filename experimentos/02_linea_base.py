"""
Experimento 02 - Línea base.

Pregunta: ¿Cómo se comparan ReliefF, ANN y Proto con sus configuraciones óptimas
sobre los 12 datasets estándar?

Método: 5-fold CV × 5 semillas para cada algoritmo y dataset.
Se reporta F1 medio ± std sobre semillas.

Salidas:
  results/tablas/02_linea_base.csv
  results/figuras/02_comparacion_datasets.png
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt import ANN, Proto
from relieff_opt.utils.conjuntos import obtener_todos
from relieff_opt.utils.metricas import tabla_resumen, tests_significancia
from relieff_opt.utils.paleta import (
    COLORES, MARCADORES, ESTILOS_LINEA, GROSOR_LINEA,
    nueva_figura, guardar_figura, estilo_academico,
)
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/02_linea_base'
TAB_DIR = 'tablas/02_linea_base'

SEMILLAS    = CFG['experimento']['semillas']
N_PLIEGUES  = CFG['experimento']['cv_pliegues']
N_SEL       = CFG['experimento']['n_features_seleccionadas']


# ---------------------------------------------------------------------------
# Construcción de selectores
# ---------------------------------------------------------------------------

def crear_selectores(n_sel, semilla):
    return {
        'ReliefF': ReliefF(
            n_features_to_select=n_sel,
            n_neighbors=CFG['relieff']['n_neighbors'],
        ),
        'ANN': ANN(
            n_features_to_select=n_sel,
            n_neighbors=CFG['ann']['n_neighbors'],
            metric=CFG['ann']['metric'],
            iter_ratio=CFG['ann']['iter_ratio'],
            random_state=semilla,
        ),
        'Proto': Proto(
            n_features_to_select=n_sel,
            k_protos=CFG['proto']['k_prototipos'],
            sigma=CFG['proto']['sigma'],
            use_lvq=CFG['proto']['usar_lvq'],
            metric=CFG['proto']['metric'],
            n_jobs=1,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluar(selector, X, y, semilla):
    """5-fold CV → F1 medio y std."""
    skf = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=semilla)
    f1s = []
    for idx_tr, idx_te in skf.split(X, y):
        Xtr, Xte = X[idx_tr], X[idx_te]
        ytr, yte = y[idx_tr], y[idx_te]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
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
            f1s.append(f1_score(yte, clf.predict(Xte_s), average='binary'))
        except Exception:
            f1s.append(0.0)
    return float(np.mean(f1s)), float(np.std(f1s))


# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------

def ejecutar():
    logger.info("Cargando datasets...")
    filas = []
    scores_globales = {'ReliefF': [], 'ANN': [], 'Proto': []}

    for nombre, (X, y, _) in obtener_todos(semilla=SEMILLAS[0]).items():
        n_sel = min(N_SEL, X.shape[1])
        logger.info("  %s  (%d × %d, seleccionando %d)", nombre, *X.shape, n_sel)

        for algoritmo in ['ReliefF', 'ANN', 'Proto']:
            f1s_semillas = []
            for semilla in SEMILLAS:
                selectores = crear_selectores(n_sel, semilla)
                f1_media, _ = evaluar(selectores[algoritmo], X, y, semilla)
                f1s_semillas.append(f1_media)

            media  = float(np.mean(f1s_semillas))
            std    = float(np.std(f1s_semillas, ddof=1))
            scores_globales[algoritmo].append(media)
            filas.append({
                'dataset':    nombre,
                'algoritmo':  algoritmo,
                'n_features': X.shape[1],
                'n_sel':      n_sel,
                'f1_media':   round(media, 4),
                'f1_std':     round(std, 4),
            })

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'linea_base', TAB_DIR)

    # Tabla resumen global
    resumen = tabla_resumen(scores_globales)
    guardar_tabla(resumen, 'resumen', TAB_DIR)

    # Tests de significancia
    tests = tests_significancia(scores_globales)
    guardar_tabla(tests, 'significancia', TAB_DIR)

    logger.info("\nResumen global:")
    for _, fila in resumen.iterrows():
        logger.info(
            "  %s: %.4f ± %.4f  [IC95: %.4f–%.4f]",
            fila['algoritmo'], fila['media'], fila['std'],
            fila.get('ic_95_inf', float('nan')),
            fila.get('ic_95_sup', float('nan')),
        )

    return df


# ---------------------------------------------------------------------------
# Gráfica
# ---------------------------------------------------------------------------

def graficar(df):
    datasets = df['dataset'].unique()
    x = np.arange(len(datasets))
    ancho = 0.25

    fig, ax = nueva_figura(tamano=(12, 5))
    for i, alg in enumerate(['ReliefF', 'ANN', 'Proto']):
        sub = df[df['algoritmo'] == alg].set_index('dataset').reindex(datasets)
        offset = (i - 1) * ancho
        barras = ax.bar(
            x + offset,
            sub['f1_media'],
            width=ancho,
            label=alg,
            color=COLORES[alg],
            alpha=0.85,
        )
        ax.errorbar(
            x + offset,
            sub['f1_media'],
            yerr=sub['f1_std'],
            fmt='none',
            color='#333333',
            capsize=3,
            linewidth=1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('F1 (media ± std, 5 semillas)')
    ax.set_title('Comparación línea base - 12 datasets (5-fold CV, 5 semillas)')
    ax.set_ylim(0, 1.08)
    ax.legend(framealpha=0.9)
    guardar_figura(fig, 'comparacion_datasets', FIG_DIR)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    df = ejecutar()
    graficar(df)
    logger.info("Experimento 02 completado en %.1f min", (time.time() - t0) / 60)
