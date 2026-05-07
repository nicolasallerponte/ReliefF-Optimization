"""
Experimento 08 - Huella de carbono (Green AI).

Pregunta: ¿Cuánta energía y CO₂ consume cada algoritmo? ¿Existe un trade-off
entre rendimiento (F1) y eficiencia energética?

Método: se miden las emisiones de CO₂ equivalente con la librería codecarbon
durante el fit() de cada algoritmo en los 12 datasets. Se calcula:
  - Emisiones totales (kg CO₂eq) por algoritmo
  - F1 / g_CO₂eq : ratio eficiencia verde (más es mejor)
  - Curva de Pareto: F1 vs emisiones

Si codecarbon no está instalado, se usa tiempo de CPU como proxy de energía
(asumiendo consumo medio de 65 W) con advertencia visible.

Multi-seed: 5 semillas.

Salidas:
  results/tablas/08_huella_carbono/huella_carbono.csv
  results/tablas/08_huella_carbono/pareto.csv
  results/figuras/08_huella_carbono/pareto_f1_co2.png
  results/figuras/08_huella_carbono/ratio_verde.png

Nota: Este experimento es especialmente relevante para el contexto del TFG
en la Cátedra Inditex-UDC de IA en Algoritmos Verdes.
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
from skrebate import ReliefF

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt import ANN, Proto
from relieff_opt.utils.conjuntos import obtener_todos
from relieff_opt.utils.paleta import (
    COLORES, MARCADORES, nueva_figura, guardar_figura,
)
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/08_huella_carbono'
TAB_DIR = 'tablas/08_huella_carbono'

SEMILLAS   = CFG['experimento']['semillas']
ALGORITMOS = ['ReliefF', 'ANN', 'Proto']

# Consumo medio de CPU asumido cuando codecarbon no está disponible [Watts]
_CONSUMO_CPU_W = 65.0

# Intentar importar codecarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_DISPONIBLE = True
    logger.info("codecarbon detectado - se usarán emisiones reales de CO₂")
except ImportError:
    CODECARBON_DISPONIBLE = False
    logger.warning(
        "codecarbon no instalado. Se usa tiempo × %.0f W como proxy de energía.\n"
        "Instala con: uv add codecarbon", _CONSUMO_CPU_W
    )


# ---------------------------------------------------------------------------
# Medición de emisiones
# ---------------------------------------------------------------------------

def medir_emisiones(selector, X, y):
    """
    Devuelve (f1_placeholder, kg_co2eq, tiempo_s).
    El F1 no se calcula aquí; se calcula en 02_linea_base para reutilizarlo.
    """
    if CODECARBON_DISPONIBLE:
        tracker = EmissionsTracker(
            log_level='error',
            save_to_file=False,
            measure_power_secs=1,
        )
        tracker.start()
        t0 = time.perf_counter()
        selector.fit(X, y)
        t = time.perf_counter() - t0
        emisiones = tracker.stop()   # kg CO₂eq
    else:
        t0 = time.perf_counter()
        selector.fit(X, y)
        t = time.perf_counter() - t0
        # Proxy: tiempo × potencia → kWh → kg CO₂eq (factor 0.233 kg/kWh para España 2023)
        kwh = (t * _CONSUMO_CPU_W) / 3_600_000
        emisiones = kwh * 0.233

    return emisiones, t


def crear_selector(algoritmo, semilla):
    if algoritmo == 'ReliefF':
        return ReliefF(n_features_to_select=10, n_neighbors=10)
    if algoritmo == 'ANN':
        return ANN(n_features_to_select=10, random_state=semilla)
    return Proto(n_features_to_select=10, n_jobs=1)


# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------

def ejecutar():
    logger.info("Cargando datasets...")
    datasets = obtener_todos(semilla=SEMILLAS[0])
    filas = []

    for nombre, (X, y, _) in datasets.items():
        n_sel = min(CFG['experimento']['n_features_seleccionadas'], X.shape[1])
        logger.info("  %s (%d × %d)", nombre, *X.shape)

        for alg in ALGORITMOS:
            emisiones_lista = []
            tiempos_lista   = []
            for semilla in SEMILLAS:
                sel = crear_selector(alg, semilla)
                # Ajustar n_features_to_select
                if hasattr(sel, 'n_features_to_select'):
                    sel.n_features_to_select = n_sel
                try:
                    em, t = medir_emisiones(sel, X.copy(), y.copy())
                    emisiones_lista.append(em)
                    tiempos_lista.append(t)
                except Exception as e:
                    logger.warning("    Error %s semilla=%d: %s", alg, semilla, e)

            if emisiones_lista:
                filas.append({
                    'dataset':         nombre,
                    'algoritmo':       alg,
                    'n_muestras':      X.shape[0],
                    'n_features':      X.shape[1],
                    'kg_co2_media':    float(np.mean(emisiones_lista)),
                    'kg_co2_std':      float(np.std(emisiones_lista, ddof=1)),
                    'tiempo_s_media':  float(np.mean(tiempos_lista)),
                    'tiempo_s_std':    float(np.std(tiempos_lista, ddof=1)),
                    'metodo_medicion': 'codecarbon' if CODECARBON_DISPONIBLE else 'proxy_cpu',
                })

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'huella_carbono', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Tabla resumen y ratio verde
# ---------------------------------------------------------------------------

def calcular_pareto(df, df_f1):
    """
    Cruza las emisiones con los F1 de 02_linea_base.csv para construir
    la tabla de la frontera de Pareto (eficiencia energética vs rendimiento).
    """
    if df_f1 is None:
        return None

    df_merge = df.merge(
        df_f1[['dataset', 'algoritmo', 'f1_media']],
        on=['dataset', 'algoritmo'],
        how='left',
    )

    # Ratio verde: F1 por gramo de CO₂
    g_co2 = df_merge['kg_co2_media'] * 1000  # kg → g
    df_merge['ratio_verde'] = (df_merge['f1_media'] / (g_co2 + 1e-12)).round(2)

    pareto = df_merge.groupby('algoritmo').agg(
        f1_media=('f1_media', 'mean'),
        kg_co2_media=('kg_co2_media', 'mean'),
        ratio_verde=('ratio_verde', 'mean'),
    ).reset_index()

    guardar_tabla(pareto, 'pareto', TAB_DIR)
    return df_merge, pareto


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def graficar_pareto(pareto):
    fig, ax = nueva_figura()
    for _, fila in pareto.iterrows():
        alg = fila['algoritmo']
        ax.scatter(
            fila['kg_co2_media'] * 1e6,   # µg CO₂
            fila['f1_media'],
            color=COLORES[alg],
            marker=MARCADORES[alg],
            s=120, zorder=5, label=alg,
        )
        ax.annotate(
            alg,
            (fila['kg_co2_media'] * 1e6, fila['f1_media']),
            xytext=(6, 4), textcoords='offset points', fontsize=9,
        )

    ax.set_xlabel('Emisiones CO₂ medias por ejecución [µg CO₂eq]')
    ax.set_ylabel('F1 medio (12 datasets, 5 semillas)')
    ax.set_title('Frontera de Pareto: rendimiento vs huella de carbono\n(arriba-izquierda = óptimo)')
    ax.legend()
    guardar_figura(fig, 'pareto_f1_co2', FIG_DIR)


def graficar_ratio_verde(pareto):
    fig, ax = nueva_figura()
    algoritmos = pareto['algoritmo'].tolist()
    ratios     = pareto['ratio_verde'].tolist()
    colores    = [COLORES[a] for a in algoritmos]

    barras = ax.bar(algoritmos, ratios, color=colores, alpha=0.85, width=0.5)
    for barra, ratio in zip(barras, ratios):
        ax.text(barra.get_x() + barra.get_width() / 2,
                barra.get_height() * 1.01,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('F1 / g CO₂eq  (ratio eficiencia verde, mayor = mejor)')
    ax.set_title('Eficiencia verde por algoritmo\n(F1 obtenido por gramo de CO₂ emitido)')
    guardar_figura(fig, 'ratio_verde', FIG_DIR)
    logger.info("Figuras guardadas en %s", FIG_DIR)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    df = ejecutar()

    # Intentar cruzar con F1 de la línea base
    raiz = Path(__file__).resolve().parents[1]
    ruta_f1 = raiz / 'results' / 'tablas' / '02_linea_base' / 'linea_base.csv'
    df_f1 = pd.read_csv(ruta_f1) if ruta_f1.exists() else None
    if df_f1 is None:
        logger.warning("No se encontró 02_linea_base/linea_base.csv - ejecuta primero 02_linea_base.py")

    resultado = calcular_pareto(df, df_f1)
    if resultado is not None:
        df_merge, pareto = resultado
        graficar_pareto(pareto)
        graficar_ratio_verde(pareto)
        logger.info("\nRatio verde (F1/g CO₂):")
        logger.info("\n%s", pareto[['algoritmo', 'f1_media', 'kg_co2_media', 'ratio_verde']].to_string(index=False))

    logger.info("Experimento 08 completado en %.1f min", (time.time() - t0) / 60)
