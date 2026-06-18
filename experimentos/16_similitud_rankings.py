"""
Experimento 16 - Similitud de los rankings entre ReliefF y las propuestas.

Pregunta: ¿producen HNSW-ReliefF y Proto-ReliefF el mismo ranking de relevancia
que ReliefF, o solo un F1 de clasificación parecido? Compara directamente los
rankings, no el rendimiento aguas abajo.

Método: para cada conjunto y semilla se ajustan los tres algoritmos sobre los
mismos datos y se comparan sus vectores de relevancia mediante:
  - Correlación de Spearman entre los pesos de ReliefF y los de cada propuesta
    (mide si se conserva el orden completo de relevancia).
  - Solapamiento del top-k: fracción de las k variables seleccionadas que
    coinciden con las que selecciona ReliefF (mide si se eligen las mismas).
Se excluyen los conjuntos con menos de N_SEL + 2 variables, donde la comparación
es trivial.

Salidas:
  results/tablas/16_similitud_rankings/similitud.csv + .tex
  results/figuras/16_similitud_rankings/spearman_boxplot.png
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from skrebate import ReliefF

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt import ANN, Proto
from relieff_opt.utils.conjuntos import CATALOGO, obtener_dataset
from relieff_opt.utils.paleta import COLORES, nueva_figura, guardar_figura
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/16_similitud_rankings'
TAB_DIR = 'tablas/16_similitud_rankings'

N_SEL    = CFG['experimento']['n_features_seleccionadas']
SEMILLAS = [42, 123, 456, 789, 1234]

# Conjuntos de tamaño pequeño/medio de la comparación de calidad.
DATASETS = [
    'Breast_Cancer', 'Wine', 'Digits', 'Corral', 'XOR',
    'Moons', 'Circles', 'AltaDim', 'Desbalanceado', 'Ruidoso', 'Grande',
]


def pesos(algoritmo, X, y, semilla):
    """Ajusta el algoritmo y devuelve su vector de relevancia (feature_importances_)."""
    n_sel = min(N_SEL, X.shape[1])
    if algoritmo == 'ReliefF':
        sel = ReliefF(n_features_to_select=n_sel, n_neighbors=CFG['relieff']['n_neighbors'])
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
    sel.fit(X, y)
    return np.asarray(sel.feature_importances_, dtype=float)


def solapamiento_top(w_a, w_b, k):
    """Fracción de las k variables top que comparten ambos rankings."""
    top_a = set(np.argsort(-w_a)[:k])
    top_b = set(np.argsort(-w_b)[:k])
    return len(top_a & top_b) / k


def ejecutar():
    filas = []
    for nombre in DATASETS:
        if nombre not in CATALOGO:
            logger.warning("Conjunto %s no está en el catálogo, se omite.", nombre)
            continue

        X0, y0, _ = obtener_dataset(nombre, semilla=42)
        n_total = X0.shape[1]
        if n_total < N_SEL + 2:
            logger.info("Omitido %s (solo %d variables, comparación trivial).", nombre, n_total)
            continue

        n_sel = min(N_SEL, n_total)
        sp_ann, sp_proto, ov_ann, ov_proto = [], [], [], []

        for semilla in SEMILLAS:
            X, y, _ = obtener_dataset(nombre, semilla=semilla)
            try:
                w_ref   = pesos('ReliefF', X, y, semilla)
                w_ann   = pesos('ANN',     X, y, semilla)
                w_proto = pesos('Proto',   X, y, semilla)
            except Exception as e:
                logger.warning("  Error en %s (semilla %d): %s", nombre, semilla, e)
                continue

            c_ann, _   = spearmanr(w_ref, w_ann)
            c_proto, _ = spearmanr(w_ref, w_proto)
            sp_ann.append(c_ann)
            sp_proto.append(c_proto)
            ov_ann.append(solapamiento_top(w_ref, w_ann, n_sel))
            ov_proto.append(solapamiento_top(w_ref, w_proto, n_sel))

        if not sp_ann:
            continue

        filas.append({
            'dataset':        nombre,
            'spearman_ann':   round(float(np.mean(sp_ann)), 4),
            'spearman_proto': round(float(np.mean(sp_proto)), 4),
            'overlap_ann':    round(float(np.mean(ov_ann)), 3),
            'overlap_proto':  round(float(np.mean(ov_proto)), 3),
        })
        logger.info("  %-14s  rho_HNSW=%.3f  rho_Proto=%.3f  ov_HNSW=%.2f  ov_Proto=%.2f",
                    nombre, np.mean(sp_ann), np.mean(sp_proto),
                    np.mean(ov_ann), np.mean(ov_proto))

    df = pd.DataFrame(filas)
    # Fila de medias
    media = {
        'dataset': 'Media',
        'spearman_ann':   round(df['spearman_ann'].mean(), 4),
        'spearman_proto': round(df['spearman_proto'].mean(), 4),
        'overlap_ann':    round(df['overlap_ann'].mean(), 3),
        'overlap_proto':  round(df['overlap_proto'].mean(), 3),
    }
    df = pd.concat([df, pd.DataFrame([media])], ignore_index=True)
    guardar_tabla(df, 'similitud', TAB_DIR)
    return df


def graficar(df):
    datos = df[df['dataset'] != 'Media']
    fig, ax = nueva_figura(tamano=(7, 5))
    cajas = ax.boxplot(
        [datos['spearman_ann'].values, datos['spearman_proto'].values],
        labels=['ReliefF vs\nHNSW-ReliefF', 'ReliefF vs\nProto-ReliefF'],
        patch_artist=True, widths=0.5, showmeans=True,
    )
    for parche, color in zip(cajas['boxes'], [COLORES['ANN'], COLORES['Proto']]):
        parche.set_facecolor(color)
        parche.set_alpha(0.6)
    ax.set_ylabel('Correlación de Spearman del ranking')
    ax.set_title('Similitud del ranking de relevancia frente a ReliefF\n'
                 f'(media sobre {len(SEMILLAS)} semillas, por conjunto)')
    ax.set_ylim(0, 1.02)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    guardar_figura(fig, 'spearman_boxplot', FIG_DIR)


if __name__ == '__main__':
    df = ejecutar()
    print(df.to_string(index=False))
    graficar(df)
    logger.info("Listo.")
