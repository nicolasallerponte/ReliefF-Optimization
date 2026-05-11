"""
Experimento 14 - Latencia de compilación Numba (cold vs warm start).

Pregunta: ¿Cuánto overhead introduce la compilación JIT de Numba en la primera
llamada a fit()? ¿Es despreciable en uso real?

Método:
  Para cada tamaño de dataset (n ∈ [500, 2000, 10000]) se lanza un proceso
  hijo independiente que arranca sin caché Numba (cold start real). Dentro
  de cada proceso se ejecutan N_LLAMADAS llamadas consecutivas a ANN.fit().
  La primera es cold (incluye compilación JIT), las siguientes son warm.

  Se usa la configuración real de ANN (HNSW + iter_ratio auto) para medir
  el overhead que experimentaría un usuario real de la librería.

  Se mide también ReliefF sklearn como referencia libre de JIT.

Salidas:
  results/tablas/14_numba_warmup/warmup_raw.csv
  results/tablas/14_numba_warmup/warmup_resumen.csv
  results/figuras/14_numba_warmup/warmup_barras.png
  results/figuras/14_numba_warmup/warmup_relativo.png
"""

import logging
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/14_numba_warmup'
TAB_DIR = 'tablas/14_numba_warmup'

TAMANOS        = [500, 2000, 10000]
N_FEATURES     = 30
N_INFORMATIVE  = 20
N_LLAMADAS     = 5
SEMILLA        = CFG['experimento']['semillas'][0]


# ---------------------------------------------------------------------------
# Función que corre en el proceso hijo (JIT fresco cada vez)
# ---------------------------------------------------------------------------

def _medir_en_proceso_hijo(n, cfg_path_str, semilla, n_llamadas, cola):
    """Ejecutado en un proceso hijo — sin caché Numba heredada del padre."""
    import sys, time, warnings
    import numpy as np
    import yaml
    from pathlib import Path
    from sklearn.datasets import make_classification
    from skrebate import ReliefF

    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(Path(cfg_path_str).resolve().parents[2] / 'src'))
    from relieff_opt import ANN

    with open(cfg_path_str) as f:
        cfg = yaml.safe_load(f)

    X, y = make_classification(
        n_samples=n, n_features=30, n_informative=20,
        n_redundant=5, random_state=semilla,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    n_sel = cfg['experimento']['n_features_seleccionadas']

    tiempos_ann = []
    for _ in range(n_llamadas):
        sel = ANN(
            n_features_to_select=n_sel,
            n_neighbors=cfg['ann']['n_neighbors'],
            metric=cfg['ann']['metric'],
            M=cfg['ann']['M'],
            ef_construction=cfg['ann']['ef_construction'],
            ef_search=cfg['ann']['ef_search'],
            iter_ratio=cfg['ann']['iter_ratio'],
            random_state=semilla,
        )
        t0 = time.perf_counter()
        sel.fit(X, y)
        tiempos_ann.append(time.perf_counter() - t0)

    tiempos_rf = []
    for _ in range(n_llamadas):
        sel = ReliefF(
            n_features_to_select=n_sel,
            n_neighbors=cfg['relieff']['n_neighbors'],
        )
        t0 = time.perf_counter()
        sel.fit(X, y)
        tiempos_rf.append(time.perf_counter() - t0)

    cola.put((tiempos_ann, tiempos_rf))


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def ejecutar():
    filas = []
    for n in TAMANOS:
        logger.info("n=%d  (proceso hijo independiente — JIT cold real)", n)
        cola = mp.Queue()
        p = mp.Process(
            target=_medir_en_proceso_hijo,
            args=(n, str(CFG_PATH), SEMILLA, N_LLAMADAS, cola),
        )
        p.start()
        p.join()
        tiempos_ann, tiempos_rf = cola.get()

        for llamada, (t_ann, t_rf) in enumerate(zip(tiempos_ann, tiempos_rf), start=1):
            filas.append({
                'n_muestras': n,
                'llamada':    llamada,
                'tipo':       'cold' if llamada == 1 else 'warm',
                'tiempo_ann': round(t_ann, 4),
                'tiempo_rf':  round(t_rf, 4),
            })
            logger.info("  llamada %d | ANN=%.3fs  ReliefF=%.3fs", llamada, t_ann, t_rf)

    df = pd.DataFrame(filas)

    # guardar_tabla no está disponible en proceso padre antes de importar utils
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from relieff_opt.utils.experimento import guardar_tabla
    guardar_tabla(df, 'warmup_raw', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def graficar_barras(df):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from relieff_opt.utils.paleta import COLORES, GROSOR_LINEA, nueva_figura, guardar_figura

    fig, axes = nueva_figura(filas=1, columnas=len(TAMANOS),
                             tamano=(5 * len(TAMANOS), 4.5))

    for ax, n in zip(axes, TAMANOS):
        sub = df[df['n_muestras'] == n]
        llamadas = sub['llamada'].values
        t_ann = sub['tiempo_ann'].values
        t_rf  = sub['tiempo_rf'].values

        ax.plot(llamadas, t_ann,
                color=COLORES['ANN'], linewidth=GROSOR_LINEA,
                marker='o', markersize=5, label='ANN (HNSW)')
        ax.plot(llamadas, t_rf,
                color=COLORES['ReliefF'], linewidth=GROSOR_LINEA,
                marker='^', linestyle=':', markersize=5, label='ReliefF')
        ax.set_title(f'n = {n:,}')
        ax.set_xlabel('Llamada consecutiva')
        ax.set_ylabel('Tiempo fit() [s]')
        ax.set_xticks(llamadas)
        ax.legend(fontsize=8)

    fig.suptitle('Numba warm-up: tiempo de fit() en llamadas consecutivas\n'
                 '(llamada 1 = cold JIT; 2-5 = warm; proceso hijo por cada n)',
                 fontsize=11)
    guardar_figura(fig, 'warmup_barras', FIG_DIR)


def graficar_relativo(df):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from relieff_opt.utils.paleta import COLORES, nueva_figura, guardar_figura
    from relieff_opt.utils.experimento import guardar_tabla

    filas_res = []
    for n in TAMANOS:
        sub = df[df['n_muestras'] == n]
        cold_ann  = sub[sub['llamada'] == 1]['tiempo_ann'].values[0]
        warm_ann  = sub[sub['llamada'] > 1]['tiempo_ann'].mean()
        cold_rf   = sub[sub['llamada'] == 1]['tiempo_rf'].values[0]
        warm_rf   = sub[sub['llamada'] > 1]['tiempo_rf'].mean()
        overhead_abs = cold_ann - warm_ann
        filas_res.append({
            'n_muestras':     n,
            'cold_ann':       round(cold_ann, 4),
            'warm_ann':       round(warm_ann, 4),
            'cold_rf':        round(cold_rf, 4),
            'warm_rf':        round(warm_rf, 4),
            'overhead_abs_s': round(overhead_abs, 3),
            'overhead_factor': round(cold_ann / warm_ann, 1) if warm_ann > 0 else float('inf'),
        })

    df_res = pd.DataFrame(filas_res)
    guardar_tabla(df_res, 'warmup_resumen', TAB_DIR)

    import numpy as np
    fig, ax = nueva_figura()
    x = np.arange(len(TAMANOS))
    width = 0.35
    ax.bar(x - width / 2, df_res['cold_ann'], width,
           color=COLORES['ANN'], alpha=0.85, label='ANN cold (1ª llamada)')
    ax.bar(x + width / 2, df_res['warm_ann'], width,
           color=COLORES['ANN'], alpha=0.40, label='ANN warm (media 2-5)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'n={n:,}' for n in TAMANOS])
    ax.set_ylabel('Tiempo fit() [s]')
    ax.set_title('Overhead Numba JIT: cold vs warm start\n'
                 'La diferencia de altura = coste de compilación JIT')
    ax.legend(fontsize=9)

    for i, row in df_res.iterrows():
        ax.text(i - width / 2, row['cold_ann'] + 0.05,
                f'+{row["overhead_abs_s"]:.2f}s',
                ha='center', va='bottom', fontsize=8, color=COLORES['ANN'])

    guardar_figura(fig, 'warmup_relativo', FIG_DIR)

    logger.info("Resumen overhead Numba:")
    for _, row in df_res.iterrows():
        logger.info("  n=%6d | cold=%.3fs  warm=%.3fs  overhead=+%.3fs (×%.1f)",
                    row['n_muestras'], row['cold_ann'], row['warm_ann'],
                    row['overhead_abs_s'], row['overhead_factor'])


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    df = ejecutar()
    graficar_barras(df)
    graficar_relativo(df)
    logger.info("Experimento 14 completado en %.1f min", (time.time() - t0) / 60)
