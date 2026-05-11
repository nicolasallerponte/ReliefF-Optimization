"""
Experimento 10 - Sensibilidad al número de features seleccionadas (k).

Pregunta: ¿Cómo varía el rendimiento de clasificación a medida que aumentamos
el número de features seleccionadas? ¿A partir de qué k se estabiliza?

Este experimento genera la "curva de rendimiento por k" para cada algoritmo,
útil para justificar la elección de k=10 en el resto de los experimentos.

Método: en CorrAL-100 y datasets reales, se evalúa F1 con k ∈ [1, 20].
5-fold CV × 5 semillas. Punto de inflexión estimado con la diferencia de
pendiente entre k consecutivos.

Salidas:
  results/tablas/10_sensibilidad_k/sensibilidad_k.csv
  results/figuras/10_sensibilidad_k/curva_k_{dataset}.png
  results/figuras/10_sensibilidad_k/codo_k.png
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
from relieff_opt.utils.conjuntos import obtener_dataset
from relieff_opt.utils.paleta import (
    COLORES, MARCADORES, ESTILOS_LINEA, GROSOR_LINEA,
    nueva_figura, guardar_figura,
)
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/10_sensibilidad_k'
TAB_DIR = 'tablas/10_sensibilidad_k'

SEMILLAS    = CFG['experimento']['semillas']
N_PLIEGUES  = CFG['experimento']['cv_pliegues']
ALGORITMOS  = ['ReliefF', 'ANN', 'Proto']
DATASETS    = ['CorrAL100', 'Breast_Cancer', 'Digits', 'AltaDim']
K_VALORES   = list(range(1, 21))   # k de 1 a 20


# ---------------------------------------------------------------------------
# Evaluación F1 para un k dado
# ---------------------------------------------------------------------------

def evaluar_k(selector_clase, X, y, k, semilla):
    """Ajusta el selector con n_features_to_select=k y evalúa 5-fold CV."""
    skf = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=semilla)
    f1s = []
    for idx_tr, idx_te in skf.split(X, y):
        Xtr, Xte = X[idx_tr], X[idx_te]
        ytr, yte = y[idx_tr], y[idx_te]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        try:
            k_real = min(k, X.shape[1])
            sel = selector_clase(n_features_to_select=k_real, semilla=semilla)
            sel.fit(Xtr, ytr)
            Xtr_s = sel.transform(Xtr)[:, :k_real] if Xtr_s.shape[1] > k_real else sel.transform(Xtr)
            Xte_s = sel.transform(Xte)[:, :k_real] if Xte_s.shape[1] > k_real else sel.transform(Xte)
            clf = RandomForestClassifier(
                n_estimators=CFG['experimento']['rf_n_estimators'],
                max_depth=CFG['experimento']['rf_max_depth'],
                random_state=semilla, n_jobs=1,
            )
            clf.fit(Xtr_s, ytr)
            f1s.append(f1_score(yte, clf.predict(Xte_s), average='binary'))
        except Exception:
            f1s.append(0.0)
    return float(np.mean(f1s))


def fabrica_selector(algoritmo):
    """Devuelve una función que crea el selector con n_features_to_select y semilla."""
    if algoritmo == 'ReliefF':
        def crear(n_features_to_select, semilla):
            return ReliefF(n_features_to_select=n_features_to_select,
                           n_neighbors=CFG['relieff']['n_neighbors'])
    elif algoritmo == 'ANN':
        def crear(n_features_to_select, semilla):
            return ANN(
                n_features_to_select=n_features_to_select,
                n_neighbors=CFG['ann']['n_neighbors'],
                metric=CFG['ann']['metric'],
                M=CFG['ann']['M'],
                ef_construction=CFG['ann']['ef_construction'],
                ef_search=CFG['ann']['ef_search'],
                random_state=semilla,
            )
    else:
        def crear(n_features_to_select, semilla):
            return Proto(
                n_features_to_select=n_features_to_select,
                k_protos=CFG['proto']['k_protos'],
                sigma=CFG['proto']['sigma'],
                use_lvq=CFG['proto']['use_lvq'],
                metric=CFG['proto']['metric'],
                n_jobs=1,
            )
    return crear


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def ejecutar():
    filas = []
    for nombre in DATASETS:
        logger.info("Dataset: %s", nombre)
        X, y, _ = obtener_dataset(nombre, semilla=SEMILLAS[0])
        k_max = min(max(K_VALORES), X.shape[1])
        ks = [k for k in K_VALORES if k <= k_max]

        for alg in ALGORITMOS:
            crear = fabrica_selector(alg)
            for k in ks:
                f1s = []
                for semilla in SEMILLAS:
                    X_s, y_s, _ = obtener_dataset(nombre, semilla=semilla)
                    skf = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=semilla)
                    pliegues_f1 = []
                    for idx_tr, idx_te in skf.split(X_s, y_s):
                        Xtr, Xte = X_s[idx_tr], X_s[idx_te]
                        ytr, yte = y_s[idx_tr], y_s[idx_te]
                        sc = StandardScaler()
                        Xtr = sc.fit_transform(Xtr)
                        Xte = sc.transform(Xte)
                        try:
                            sel = crear(n_features_to_select=k, semilla=semilla)
                            sel.fit(Xtr, ytr)
                            Xtr_s = sel.transform(Xtr)
                            Xte_s = sel.transform(Xte)
                            clf = RandomForestClassifier(
                                n_estimators=CFG['experimento']['rf_n_estimators'],
                                max_depth=CFG['experimento']['rf_max_depth'],
                                random_state=semilla, n_jobs=1,
                            )
                            clf.fit(Xtr_s, ytr)
                            pliegues_f1.append(f1_score(yte, clf.predict(Xte_s), average='binary'))
                        except Exception:
                            pliegues_f1.append(0.0)
                    f1s.append(float(np.mean(pliegues_f1)))

                filas.append({
                    'dataset':   nombre,
                    'algoritmo': alg,
                    'k':         k,
                    'f1_media':  round(float(np.mean(f1s)), 4),
                    'f1_std':    round(float(np.std(f1s, ddof=1)), 4),
                })
            logger.info("  %s completado", alg)

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'sensibilidad_k', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def punto_codo(ks, vals):
    """Estima el codo de la curva mediante distancia perpendicular máxima."""
    if len(ks) < 3:
        return ks[np.argmax(vals)]
    p1 = np.array([ks[0], vals[0]])
    p2 = np.array([ks[-1], vals[-1]])
    n = p2 - p1
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return ks[0]
    n = n / n_norm
    distancias = []
    for k, v in zip(ks, vals):
        p = np.array([k, v])
        dist = np.abs(np.cross(n, p1 - p))
        distancias.append(dist)
    return ks[np.argmax(distancias)]


def graficar_curvas(df):
    for nombre in df['dataset'].unique():
        sub = df[df['dataset'] == nombre]
        fig, ax = nueva_figura()

        for alg in ALGORITMOS:
            datos = sub[sub['algoritmo'] == alg].sort_values('k')
            ks    = datos['k'].tolist()
            medias = datos['f1_media'].tolist()
            stds   = datos['f1_std'].tolist()

            ax.plot(ks, medias,
                    color=COLORES[alg], marker=MARCADORES[alg],
                    linestyle=ESTILOS_LINEA[alg], linewidth=GROSOR_LINEA,
                    markersize=5, label=alg)
            ax.fill_between(ks,
                            [m - s for m, s in zip(medias, stds)],
                            [m + s for m, s in zip(medias, stds)],
                            color=COLORES[alg], alpha=0.12)

            codo = punto_codo(ks, medias)
            ax.axvline(codo, color=COLORES[alg], linestyle=':', linewidth=0.8, alpha=0.6)

        ax.set_xlabel('Número de features seleccionadas (k)')
        ax.set_ylabel('F1 (media ± std, 5 semillas)')
        ax.set_title(f'Curva de rendimiento por k - {nombre}\n(línea punteada = codo estimado)')
        ax.set_xlim(0, max(K_VALORES) + 1)
        ax.legend()
        guardar_figura(fig, f'curva_k_{nombre.lower()}', FIG_DIR)

    logger.info("Figuras de curvas guardadas en %s", FIG_DIR)


def graficar_codo_comparativo(df):
    """Gráfica resumen con el k de codo por dataset y algoritmo."""
    filas = []
    for nombre in df['dataset'].unique():
        for alg in ALGORITMOS:
            datos = df[(df['dataset'] == nombre) & (df['algoritmo'] == alg)].sort_values('k')
            if datos.empty:
                continue
            codo = punto_codo(datos['k'].tolist(), datos['f1_media'].tolist())
            filas.append({'dataset': nombre, 'algoritmo': alg, 'k_codo': codo})

    df_codo = pd.DataFrame(filas)
    guardar_tabla(df_codo, 'k_codo', TAB_DIR)

    datasets = df_codo['dataset'].unique()
    x = np.arange(len(datasets))
    ancho = 0.25

    fig, ax = nueva_figura(tamano=(10, 4.5))
    for i, alg in enumerate(ALGORITMOS):
        sub = df_codo[df_codo['algoritmo'] == alg].set_index('dataset').reindex(datasets)
        ax.bar(x + (i - 1) * ancho, sub['k_codo'], width=ancho,
               color=COLORES[alg], alpha=0.85, label=alg)

    ax.axhline(10, color='gray', linestyle='--', linewidth=0.9, label='k=10 (usado en TFG)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha='right')
    ax.set_ylabel('k de codo estimado')
    ax.set_title('k óptimo (punto de codo) por dataset y algoritmo')
    ax.legend()
    guardar_figura(fig, 'codo_k', FIG_DIR)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    df = ejecutar()
    graficar_curvas(df)
    graficar_codo_comparativo(df)
    logger.info("Experimento 10 completado en %.1f min", (time.time() - t0) / 60)
