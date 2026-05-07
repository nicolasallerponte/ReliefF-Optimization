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
from skrebate import ReliefF

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
ALGORITMOS = ['ReliefF', 'ANN', 'Proto']

TAMANOS = [500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000]
N_FEATURES = 30
N_INFORMATIVE = 20


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def medir_tiempo(algoritmo, X, y, semilla):
    if algoritmo == 'ReliefF':
        sel = ReliefF(n_features_to_select=10, n_neighbors=10)
    elif algoritmo == 'ANN':
        sel = ANN(n_features_to_select=10, random_state=semilla)
    else:
        sel = Proto(n_features_to_select=10, n_jobs=1)
    t0 = time.perf_counter()
    sel.fit(X, y)
    return time.perf_counter() - t0


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
            for alg in ALGORITMOS:
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
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    df = ejecutar()
    graficar(df)
    logger.info("Experimento 04 completado en %.1f min", (time.time() - t0) / 60)
