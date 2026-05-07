"""
Experimento 06 - Análisis estadístico.

Pregunta: ¿Son las diferencias de rendimiento entre algoritmos estadísticamente
significativas, y cuál es el tamaño del efecto?

Método:
  1. Carga los F1 medios de 02_linea_base.csv (una observación por dataset)
  2. Wilcoxon signed-rank test (pareado) para cada par de algoritmos
  3. Friedman test (global entre los 3 algoritmos)
  4. Tamaño de efecto d de Cohen para cada par
  5. IC 95% bootstrap para la media de F1 de cada algoritmo

Salidas:
  results/tablas/06_estadisticas_descriptivas.csv
  results/tablas/06_tests_significancia.csv
  results/figuras/06_boxplot_comparacion.png
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt.utils.metricas import tabla_resumen, tests_significancia
from relieff_opt.utils.paleta import COLORES, nueva_figura, guardar_figura
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/06_analisis_estadistico'
TAB_DIR = 'tablas/06_analisis_estadistico'

RAIZ = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def cargar_f1():
    """Lee results/tablas/02_linea_base.csv. Si no existe, lanza error claro."""
    ruta = RAIZ / 'results' / 'tablas' / '02_linea_base.csv'
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}.\n"
            "Ejecuta primero: python experimentos/02_linea_base.py"
        )
    df = pd.read_csv(ruta)
    datos = {}
    for alg in ['ReliefF', 'ANN', 'Proto']:
        sub = df[df['algoritmo'] == alg]
        datos[alg] = sub['f1_media'].values
    logger.info("F1 cargados: %d datasets por algoritmo", len(datos['ANN']))
    return datos, df


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------

def ejecutar():
    datos, df_base = cargar_f1()

    # Estadísticas descriptivas con IC 95%
    resumen = tabla_resumen(datos, ic=True)
    guardar_tabla(resumen, 'estadisticas_descriptivas', TAB_DIR)

    # Tests de significancia (Wilcoxon + Friedman + Cohen's d)
    tests = tests_significancia(datos)
    guardar_tabla(tests, 'tests_significancia', TAB_DIR)

    logger.info("\nEstadísticas descriptivas:")
    logger.info("\n%s", resumen.to_string(index=False))
    logger.info("\nTests de significancia:")
    logger.info("\n%s", tests.to_string(index=False))

    return datos, df_base


# ---------------------------------------------------------------------------
# Gráfica
# ---------------------------------------------------------------------------

def graficar(datos):
    fig, ax = nueva_figura()
    posiciones = list(range(len(datos)))
    nombres    = list(datos.keys())
    vals       = [datos[alg] for alg in nombres]

    bp = ax.boxplot(
        vals,
        positions=posiciones,
        widths=0.5,
        patch_artist=True,
        notch=True,
        medianprops=dict(color='white', linewidth=2),
    )

    for patch, alg in zip(bp['boxes'], nombres):
        patch.set_facecolor(COLORES[alg])
        patch.set_alpha(0.75)

    # Superposición de puntos individuales (jitter)
    for i, (alg, valores) in enumerate(datos.items()):
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(valores))
        ax.scatter(
            i + jitter, valores,
            color=COLORES[alg], edgecolors='white',
            s=40, linewidths=0.6, zorder=3, alpha=0.8,
        )

    ax.set_xticks(posiciones)
    ax.set_xticklabels(nombres)
    ax.set_ylabel('F1 (media sobre 5 semillas, 12 datasets)')
    ax.set_title('Distribución de F1 por algoritmo\n(boxplot con notch, puntos = datasets individuales)')
    guardar_figura(fig, 'boxplot_comparacion', FIG_DIR)
    logger.info("Figura guardada: 06_boxplot_comparacion.png")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    datos, df_base = ejecutar()
    graficar(datos)
