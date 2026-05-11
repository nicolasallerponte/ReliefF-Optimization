"""
Experimento 04 - Escalabilidad temporal.

Pregunta: ¿Cómo escala el tiempo de ejecución de cada algoritmo con el tamaño
del dataset? ¿ANN rompe la complejidad O(n²) de ReliefF original?

Método: dataset sintético clasificación binaria, n_features=30, n_informative=20.
Se varía n_muestras de 500 a 30 000. Para cada tamaño × semilla se mide el
tiempo de fit().  Multi-seed: 5 semillas; se reporta media ± std.

Salidas:
  results/tablas/04_escalabilidad.csv
  results/figuras/04_tiempo_lineal.png
  results/figuras/04_tiempo_loglog.png
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
from sklearn.datasets import make_classification
from skrebate import ReliefF, MultiSURF

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt import ANN, Proto
from relieff_opt.utils.paleta import (
    COLORES, MARCADORES, ESTILOS_LINEA, GROSOR_LINEA,
    nueva_figura, guardar_figura,
)
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/04_escalabilidad'
TAB_DIR = 'tablas/04_escalabilidad'

SEMILLAS   = CFG['experimento']['semillas']
ALGORITMOS = ['ReliefF', 'MultiSURF', 'ANN', 'Proto']

TAMANOS = [500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000]

# ReliefF y MultiSURF son O(n²) — inviables a gran escala.
# Por encima de este umbral solo se miden ANN y Proto.
N_MAX_CUADRATICO = 10000
N_FEATURES = 30
N_INFORMATIVE = 20


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def medir_tiempo(algoritmo, X, y, semilla):
    n_sel = CFG['experimento']['n_features_seleccionadas']
    if algoritmo == 'ReliefF':
        sel = ReliefF(n_features_to_select=n_sel, n_neighbors=CFG['relieff']['n_neighbors'])
    elif algoritmo == 'MultiSURF':
        sel = MultiSURF(n_features_to_select=n_sel)
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
    t0 = time.perf_counter()
    sel.fit(X, y)
    return time.perf_counter() - t0


def ejecutar():
    filas = []
    for n in TAMANOS:
        # ReliefF y MultiSURF son O(n²): se omiten para n > N_MAX_CUADRATICO
        algs = ALGORITMOS if n <= N_MAX_CUADRATICO else ['ANN', 'Proto']
        if n > N_MAX_CUADRATICO:
            logger.info("n_muestras=%d  (solo ANN y Proto — ReliefF/MultiSURF inviables a esta escala)", n)
        else:
            logger.info("n_muestras=%d", n)
        for semilla in SEMILLAS:
            X, y = make_classification(
                n_samples=n, n_features=N_FEATURES, n_informative=N_INFORMATIVE,
                n_redundant=5, random_state=semilla,
            )
            X = X.astype(np.float32)
            y = y.astype(np.int32)
            for alg in algs:
                t = medir_tiempo(alg, X, y, semilla)
                filas.append({
                    'n_muestras': n,
                    'algoritmo':  alg,
                    'semilla':    semilla,
                    'tiempo_s':   round(t, 4),
                })

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'escalabilidad', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def _ajuste_potencia(x, y):
    """Ajuste log-log para estimar el exponente de escala."""
    log_x = np.log(x)
    log_y = np.log(y + 1e-9)
    coef = np.polyfit(log_x, log_y, 1)
    return coef[0], np.exp(coef[1])   # exponente, coeficiente


def graficar(df):
    agrup = df.groupby(['n_muestras', 'algoritmo']).agg(
        media=('tiempo_s', 'mean'),
        std=('tiempo_s', 'std'),
    ).reset_index()

    # Escala lineal
    fig, ax = nueva_figura()
    for alg in ALGORITMOS:
        datos = agrup[agrup['algoritmo'] == alg]
        ax.plot(datos['n_muestras'], datos['media'],
                color=COLORES[alg], marker=MARCADORES[alg],
                linestyle=ESTILOS_LINEA[alg], linewidth=GROSOR_LINEA,
                markersize=5, label=alg)
        ax.fill_between(datos['n_muestras'],
                        datos['media'] - datos['std'],
                        datos['media'] + datos['std'],
                        color=COLORES[alg], alpha=0.15)
    ax.set_xlabel('Número de muestras')
    ax.set_ylabel('Tiempo de fit() [s] (media ± std, 5 semillas)')
    ax.set_title('Escalabilidad temporal')
    ax.legend()
    guardar_figura(fig, 'tiempo_lineal', FIG_DIR)

    # Escala log-log
    fig, ax = nueva_figura()
    for alg in ALGORITMOS:
        datos = agrup[agrup['algoritmo'] == alg]
        x = datos['n_muestras'].values
        y = datos['media'].values
        ax.loglog(x, y,
                  color=COLORES[alg], marker=MARCADORES[alg],
                  linestyle=ESTILOS_LINEA[alg], linewidth=GROSOR_LINEA,
                  markersize=5, label=alg)
        exp, cte = _ajuste_potencia(x, y)
        x_fit = np.linspace(x.min(), x.max(), 100)
        ax.loglog(x_fit, cte * x_fit ** exp,
                  color=COLORES[alg], linestyle='--', linewidth=0.9, alpha=0.6,
                  label=f'{alg} ~ n^{exp:.2f}')

    ax.set_xlabel('Número de muestras (log)')
    ax.set_ylabel('Tiempo de fit() [s] (log)')
    ax.set_title('Escalabilidad temporal - escala log-log con ajuste potencial')
    ax.legend(fontsize=9)
    guardar_figura(fig, 'tiempo_loglog', FIG_DIR)
    logger.info("Figuras guardadas: 04_tiempo_lineal.png, 04_tiempo_loglog.png")


# ---------------------------------------------------------------------------
# Benchmark real: CoverType_10k (dataset real, n=10000, d=54)
# ---------------------------------------------------------------------------

def ejecutar_real():
    """Validación de escalabilidad con Cover Type (real, n=10000, 54 features).

    CoverType se descarga automáticamente por sklearn (~11 MB, caché local).
    Demuestra que el speedup de ANN se mantiene en datos reales.
    """
    from relieff_opt.utils.conjuntos import obtener_dataset
    logger.info("Cargando CoverType_10k (dataset real)...")
    try:
        X, y, _ = obtener_dataset('CoverType_10k', semilla=42)
        logger.info("  CoverType_10k: %s, balance=%.2f", X.shape, y.mean())
    except Exception as e:
        logger.warning("No se pudo cargar CoverType_10k: %s", e)
        return None

    filas = []
    for alg in ALGORITMOS:
        tiempos = []
        for semilla in SEMILLAS:
            try:
                t = medir_tiempo(alg, X, y, semilla)
                tiempos.append(t)
                logger.info("  %s semilla=%d: %.2fs", alg, semilla, t)
            except Exception as e:
                logger.warning("  %s semilla=%d error: %s", alg, semilla, e)
                tiempos.append(float('nan'))
        filas.append({
            'algoritmo':  alg,
            'n_muestras': X.shape[0],
            'n_features': X.shape[1],
            'tiempo_medio_s': round(float(np.nanmean(tiempos)), 4),
            'tiempo_std_s':   round(float(np.nanstd(tiempos)), 4),
        })
        logger.info("  %s → %.2f ± %.2f s", alg,
                    filas[-1]['tiempo_medio_s'], filas[-1]['tiempo_std_s'])

    df_real = pd.DataFrame(filas)
    guardar_tabla(df_real, 'escalabilidad_real', TAB_DIR)

    # Figura comparativa
    fig, ax = nueva_figura()
    colores_bar = [COLORES.get(alg, '#888888') for alg in df_real['algoritmo']]
    bars = ax.bar(df_real['algoritmo'], df_real['tiempo_medio_s'],
                  color=colores_bar, alpha=0.85,
                  yerr=df_real['tiempo_std_s'], capsize=4)
    ax.set_ylabel('Tiempo de fit() [s] (media ± std, 5 semillas)')
    ax.set_title('Escalabilidad — CoverType real (n=10000, d=54)\nCada barra = tiempo medio sobre 5 semillas')
    for bar, row in zip(bars, df_real.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{row.tiempo_medio_s:.1f}s', ha='center', va='bottom', fontsize=9)
    guardar_figura(fig, 'escalabilidad_real_barras', FIG_DIR)
    logger.info("Tabla y figura escalabilidad_real guardadas")
    return df_real


def ejecutar_real_mnist():
    """Validación de escalabilidad con MNIST (real, n=10000, 784 features).

    Alta dimensionalidad (784 features) amplifica el speedup de ANN respecto a
    ReliefF brute-force, cuyo coste es O(n² × d).
    """
    from relieff_opt.utils.conjuntos import obtener_dataset
    logger.info("Cargando MNIST_10k (dataset real, alta dimensión)...")
    try:
        X, y, _ = obtener_dataset('MNIST_10k', semilla=42)
        logger.info("  MNIST_10k: %s, balance=%.2f", X.shape, y.mean())
    except Exception as e:
        logger.warning("No se pudo cargar MNIST_10k: %s", e)
        return None

    filas = []
    for alg in ALGORITMOS:
        tiempos = []
        for semilla in SEMILLAS:
            try:
                t = medir_tiempo(alg, X, y, semilla)
                tiempos.append(t)
                logger.info("  %s semilla=%d: %.2fs", alg, semilla, t)
            except Exception as e:
                logger.warning("  %s semilla=%d error: %s", alg, semilla, e)
                tiempos.append(float('nan'))
        filas.append({
            'algoritmo':  alg,
            'n_muestras': X.shape[0],
            'n_features': X.shape[1],
            'tiempo_medio_s': round(float(np.nanmean(tiempos)), 4),
            'tiempo_std_s':   round(float(np.nanstd(tiempos)), 4),
        })
        logger.info("  %s → %.2f ± %.2f s", alg,
                    filas[-1]['tiempo_medio_s'], filas[-1]['tiempo_std_s'])

    df_mnist = pd.DataFrame(filas)
    guardar_tabla(df_mnist, 'escalabilidad_real_mnist', TAB_DIR)

    fig, ax = nueva_figura()
    colores_bar = [COLORES.get(alg, '#888888') for alg in df_mnist['algoritmo']]
    bars = ax.bar(df_mnist['algoritmo'], df_mnist['tiempo_medio_s'],
                  color=colores_bar, alpha=0.85,
                  yerr=df_mnist['tiempo_std_s'], capsize=4)
    ax.set_ylabel('Tiempo de fit() [s] (media ± std, 5 semillas)')
    ax.set_title('Escalabilidad — MNIST real (n=10000, d=784)\nCada barra = tiempo medio sobre 5 semillas')
    for bar, row in zip(bars, df_mnist.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{row.tiempo_medio_s:.1f}s', ha='center', va='bottom', fontsize=9)
    guardar_figura(fig, 'escalabilidad_real_mnist_barras', FIG_DIR)
    logger.info("Tabla y figura escalabilidad_real_mnist guardadas")
    return df_mnist


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    df = ejecutar()
    graficar(df)
    logger.info("Benchmark sintético completado.")
    logger.info("Iniciando benchmark real (CoverType_10k)...")
    ejecutar_real()
    logger.info("Iniciando benchmark real MNIST_10k (alta dimensión, d=784)...")
    ejecutar_real_mnist()
    logger.info("Experimento 04 completado en %.1f min", (time.time() - t0) / 60)
