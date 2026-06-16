"""
Experimento 15 - Escalabilidad en datasets reales masivos.

Pregunta: ¿se mantiene la viabilidad de HNSW-ReliefF y Proto-ReliefF en datasets
reales de gran escala, donde ReliefF exacto es directamente inejecutable?

Método: se cargan datasets reales grandes (CoverType ~581k, SUSY ~5M,
HEPMASS ~10.5M). Para cada uno se mide el tiempo de fit() del selector sobre el
conjunto de entrenamiento y se evalúa el F1 macro downstream con un único
holdout (no CV: inviable a esta escala). ReliefF/MultiSURF se omiten (O(n²)
inviable).

El clasificador downstream se entrena sobre un subconjunto acotado de las
muestras (N_CLF_MAX) para que el coste del RandomForest no domine; el selector,
en cambio, ve todo el conjunto de entrenamiento (ese es el punto de escala).

Salidas:
  results/tablas/15_escala_real/escala_real.csv
  results/figuras/15_escala_real/tiempo_f1.png
"""

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import matplotlib.pyplot as plt

from relieff_opt import ANN, Proto
from relieff_opt.utils.conjuntos import obtener_dataset
from relieff_opt.utils.experimento import guardar_tabla
from relieff_opt.utils.paleta import COLORES, nueva_figura, guardar_figura

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/15_escala_real'
TAB_DIR = 'tablas/15_escala_real'

# Datasets reales masivos. Comentar los que no se quieran/puedan descargar.
DATASETS = ['CoverType', 'SUSY', 'HEPMASS']

ALGORITMOS = ['ANN', 'Proto']
N_CLF_MAX = 200000   # tope de muestras para entrenar el RandomForest downstream
SEMILLA = 42


def crear_selector(alg, n_sel, semilla):
    if alg == 'ANN':
        return ANN(
            n_features_to_select=n_sel,
            n_neighbors=CFG['ann']['n_neighbors'],
            metric=CFG['ann']['metric'],
            M=CFG['ann']['M'],
            ef_construction=CFG['ann']['ef_construction'],
            ef_search=CFG['ann']['ef_search'],
            random_state=semilla,
        )
    return Proto(
        n_features_to_select=n_sel,
        k_protos=CFG['proto']['k_protos'],
        sigma=CFG['proto']['sigma'],
        use_lvq=CFG['proto']['use_lvq'],
        metric=CFG['proto']['metric'],
        n_jobs=-1,
    )


def evaluar(alg, X_tr, y_tr, X_te, y_te, n_sel):
    sel = crear_selector(alg, n_sel, SEMILLA)

    t0 = time.perf_counter()
    sel.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0

    X_tr_sel = sel.transform(X_tr)
    X_te_sel = sel.transform(X_te)

    # Cap del clasificador: el RF no debe dominar el coste ni la memoria.
    if len(X_tr_sel) > N_CLF_MAX:
        rng = np.random.RandomState(SEMILLA)
        idx = rng.choice(len(X_tr_sel), N_CLF_MAX, replace=False)
        X_tr_sel, y_tr = X_tr_sel[idx], y_tr[idx]

    clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 random_state=SEMILLA, n_jobs=-1)
    clf.fit(X_tr_sel, y_tr)
    f1 = f1_score(y_te, clf.predict(X_te_sel), average='macro')
    return t_fit, f1


def ejecutar():
    raw_path = Path(__file__).resolve().parents[1] / 'results' / TAB_DIR / 'escala_real_raw.csv'
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    columnas = ['dataset', 'n_muestras', 'n_features', 'algoritmo', 'tiempo_fit_s', 'f1']
    pd.DataFrame(columns=columnas).to_csv(raw_path, index=False)

    n_sel = CFG['experimento']['n_features_seleccionadas']
    filas = []
    for nombre in DATASETS:
        logger.info("Cargando %s ...", nombre)
        try:
            X, y, _ = obtener_dataset(nombre, semilla=SEMILLA)
        except Exception as e:
            logger.warning("No se pudo cargar %s: %s", nombre, e)
            continue
        logger.info("  %s: %s, balance=%.3f, ~%.2f GB", nombre, X.shape,
                    float(y.mean()), X.nbytes / 1e9)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEMILLA, stratify=y)
        del X, y

        # Estandarizado imprescindible: ANN y Proto usan distancia euclídea; sin
        # escalar, las features de gran magnitud dominan y Proto selecciona mal
        # (en CoverType pasa de F1=0.50 a 0.73). Scaler ajustado solo en train.
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)

        for alg in ALGORITMOS:
            try:
                t_fit, f1 = evaluar(alg, X_tr, y_tr, X_te, y_te, n_sel)
                fila = {
                    'dataset':      nombre,
                    'n_muestras':   X_tr.shape[0] + X_te.shape[0],
                    'n_features':   X_tr.shape[1],
                    'algoritmo':    alg,
                    'tiempo_fit_s': round(t_fit, 3),
                    'f1':           round(float(f1), 4),
                }
                logger.info("  %s/%s: fit=%.1fs  F1=%.4f", nombre, alg, t_fit, f1)
            except Exception as e:
                logger.warning("  %s/%s error: %s", nombre, alg, e)
                fila = {
                    'dataset': nombre, 'n_muestras': X_tr.shape[0] + X_te.shape[0],
                    'n_features': X_tr.shape[1], 'algoritmo': alg,
                    'tiempo_fit_s': float('nan'), 'f1': float('nan'),
                }
            filas.append(fila)
            pd.DataFrame([fila]).to_csv(raw_path, mode='a', header=False, index=False)

        del X_tr, X_te, y_tr, y_te

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'escala_real', TAB_DIR)
    return df


def graficar(df=None):
    """Dos paneles: tiempo de fit() y F1 macro por dataset y algoritmo."""
    if df is None:
        raw_path = Path(__file__).resolve().parents[1] / 'results' / TAB_DIR / 'escala_real_raw.csv'
        df = pd.read_csv(raw_path)

    datasets = list(dict.fromkeys(df['dataset']))   # preserva orden de aparición
    x = np.arange(len(datasets))
    ancho = 0.38

    fig, (ax_t, ax_f) = nueva_figura(1, 2, tamano=(12, 4.5))

    for j, alg in enumerate(ALGORITMOS):
        sub = df[df['algoritmo'] == alg].set_index('dataset')
        t = [sub.loc[d, 'tiempo_fit_s'] if d in sub.index else np.nan for d in datasets]
        f = [sub.loc[d, 'f1'] if d in sub.index else np.nan for d in datasets]
        desp = (j - 0.5) * ancho
        ax_t.bar(x + desp, t, ancho, label=alg, color=COLORES[alg])
        barras = ax_f.bar(x + desp, f, ancho, label=alg, color=COLORES[alg])
        for b, val in zip(barras, f):
            if not np.isnan(val):
                ax_f.text(b.get_x() + b.get_width() / 2, val + 0.01, f'{val:.2f}',
                          ha='center', va='bottom', fontsize=8)

    ax_t.set_ylabel('Tiempo de fit() [s]')
    ax_t.set_title('Coste de selección')
    ax_t.set_yscale('log')
    ax_f.set_ylabel('F1 macro (RandomForest downstream)')
    ax_f.set_title('Calidad de la selección')
    ax_f.set_ylim(0, 1.05)
    for ax in (ax_t, ax_f):
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()

    ruta = guardar_figura(fig, 'tiempo_f1', FIG_DIR)
    logger.info("Figura guardada: %s", ruta)


if __name__ == '__main__':
    t0 = time.time()
    df = ejecutar()
    graficar(df)
    logger.info("Experimento 15 completado en %.1f min", (time.time() - t0) / 60)
