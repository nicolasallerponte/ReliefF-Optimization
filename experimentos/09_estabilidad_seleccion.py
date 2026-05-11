"""
Experimento 09 - Estabilidad de la selección de características.

Pregunta: ¿Seleccionan los algoritmos las mismas features cuando el dataset
varía ligeramente? ¿Cuál es el más estable ante perturbaciones de los datos?

Método: Índice de Consistencia de Kuncheva (KCI) calculado sobre 10 subconjuntos
bootstrap (80% de n sin reemplazamiento). Esta métrica mide estabilidad ante
perturbación de los datos de entrenamiento — que es lo que importa en uso real.

  KCI = 1  → selección idéntica en todos los bootstraps
  KCI = 0  → estabilidad equivalente a selección aleatoria
  KCI < 0  → peor que el azar (inestabilidad)

Métrica adicional — precisión robusta al 80%: fracción de features relevantes
conocidas que aparecen en ≥80% de los bootstraps.

Nota: se excluyen datasets donde n_sel ≥ n_features (XOR, Moons, Circles)
porque KCI no está definido en ese caso.

Salidas:
  results/tablas/09_estabilidad_seleccion/kci.csv
  results/figuras/09_estabilidad_seleccion/kci_barplot.png
  results/figuras/09_estabilidad_seleccion/heatmap_{dataset}.png
"""

import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from skrebate import ReliefF

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt import ANN, Proto
from relieff_opt.utils.conjuntos import CATALOGO, obtener_dataset
from relieff_opt.utils.metricas import kuncheva_index
from relieff_opt.utils.paleta import (
    COLORES, MARCADORES, nueva_figura, guardar_figura,
)
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/09_estabilidad_seleccion'
TAB_DIR = 'tablas/09_estabilidad_seleccion'

N_SEL        = CFG['experimento']['n_features_seleccionadas']
ALGORITMOS   = ['ReliefF', 'ANN', 'Proto']
N_BOOTSTRAP  = 10
FRAC_BOOTSTRAP = 0.80

# Excluir datasets donde n_features <= N_SEL (KCI indefinido: k >= n)
DATASETS_VALIDOS = [
    'Corral', 'CorrAL100', 'AltaDim', 'Desbalanceado', 'Ruidoso', 'Grande',
]


# ---------------------------------------------------------------------------
# Selección de features con un algoritmo
# ---------------------------------------------------------------------------

def seleccionar(algoritmo, X, y, semilla):
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
    ranking = sel.rank() if hasattr(sel, 'rank') else np.argsort(-sel.feature_importances_)
    return list(ranking[:n_sel])


# ---------------------------------------------------------------------------
# Experimento con bootstrap
# ---------------------------------------------------------------------------

def ejecutar():
    filas = []
    heatmaps = {}
    rng = np.random.RandomState(42)

    for nombre in DATASETS_VALIDOS:
        meta = CATALOGO[nombre]
        logger.info("Dataset: %s", nombre)
        X_ref, y_ref, relevantes = obtener_dataset(nombre, semilla=42)
        n = len(X_ref)
        n_total = X_ref.shape[1]
        n_sel = min(N_SEL, n_total)

        # Generar índices bootstrap — iguales para todos los algoritmos (comparación justa)
        n_boot = int(n * FRAC_BOOTSTRAP)
        bootstrap_idxs  = [rng.choice(n, n_boot, replace=False) for _ in range(N_BOOTSTRAP)]
        bootstrap_seeds = [int(rng.randint(0, 10000))            for _ in range(N_BOOTSTRAP)]

        for alg in ALGORITMOS:
            subsets    = []
            frecuencia = np.zeros(n_total, dtype=int)

            for idx, seed_b in zip(bootstrap_idxs, bootstrap_seeds):
                Xb, yb = X_ref[idx], y_ref[idx]
                try:
                    subset = seleccionar(alg, Xb, yb, seed_b)
                    subsets.append(subset)
                    frecuencia[subset] += 1
                except Exception as e:
                    logger.warning("  Error %s %s: %s", alg, nombre, e)

            if len(subsets) < 2:
                continue

            kci = kuncheva_index(subsets, n_total, n_sel)
            heatmaps[(nombre, alg)] = frecuencia / len(subsets)

            if relevantes:
                umbral = 0.8
                precision_robusta = sum(
                    1 for r in relevantes
                    if frecuencia[r] / len(subsets) >= umbral
                ) / len(relevantes)
            else:
                precision_robusta = float('nan')

            filas.append({
                'dataset':               nombre,
                'algoritmo':             alg,
                'n_total_features':      n_total,
                'n_seleccionadas':       n_sel,
                'kci':                   round(kci, 4),
                'precision_robusta_80pct': round(precision_robusta, 4),
            })
            logger.info("  %s  KCI=%.3f  precision_robusta=%.2f", alg, kci, precision_robusta)

    df = pd.DataFrame(filas)
    guardar_tabla(df, 'kci', TAB_DIR)
    return df, heatmaps


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def graficar_kci(df):
    datasets = df['dataset'].unique()
    x = np.arange(len(datasets))
    ancho = 0.25

    fig, ax = nueva_figura(tamano=(max(10, len(datasets) * 1.2), 5))
    for i, alg in enumerate(ALGORITMOS):
        sub = df[df['algoritmo'] == alg].set_index('dataset').reindex(datasets)
        offset = (i - 1) * ancho
        ax.bar(x + offset, sub['kci'], width=ancho,
               color=COLORES[alg], alpha=0.85, label=alg)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Nivel aleatorio')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right')
    ax.set_ylabel('Índice de Kuncheva (KCI)')
    ax.set_title('Estabilidad ante perturbación de datos (bootstrap)\n'
                 f'(KCI, {N_BOOTSTRAP} bootstraps al {int(FRAC_BOOTSTRAP*100)}%; '
                 '1=perfecta estabilidad, 0=aleatoria)')
    ax.set_ylim(-0.2, 1.1)
    ax.legend()
    guardar_figura(fig, 'kci_barplot', FIG_DIR)
    logger.info("Figura guardada: kci_barplot.png")


def graficar_heatmaps(heatmaps, n_mostrar=15):
    """Heatmap de frecuencia de selección por feature y dataset."""
    datasets_unicos = sorted({k[0] for k in heatmaps})

    for nombre in datasets_unicos:
        relevantes = CATALOGO[nombre].get('relevantes')

        datos = []
        for alg in ALGORITMOS:
            if (nombre, alg) in heatmaps:
                freq = heatmaps[(nombre, alg)]
                top_idx = np.argsort(-freq)[:n_mostrar]
                datos.append({'algoritmo': alg, 'indices': top_idx, 'freq': freq})

        if not datos:
            continue

        todos_top = np.unique(np.concatenate([d['indices'] for d in datos]))[:n_mostrar]
        matriz = np.zeros((len(ALGORITMOS), len(todos_top)))

        for i, alg in enumerate(ALGORITMOS):
            if (nombre, alg) in heatmaps:
                for j, feat in enumerate(todos_top):
                    matriz[i, j] = heatmaps[(nombre, alg)][feat]

        etqs_feat = [
            f"f{idx}" + (" ★" if relevantes and idx in relevantes else "")
            for idx in todos_top
        ]

        fig, ax = plt.subplots(figsize=(max(10, len(todos_top) * 0.7), 3.5))
        fig.patch.set_facecolor('white')
        sns.heatmap(
            matriz,
            ax=ax,
            xticklabels=etqs_feat,
            yticklabels=ALGORITMOS,
            cmap='Blues',
            vmin=0, vmax=1,
            linewidths=0.5,
            annot=True, fmt='.2f',
            cbar_kws={'label': f'Frecuencia de selección ({N_BOOTSTRAP} bootstraps)'},
        )
        ax.set_title(f'Frecuencia de selección — {nombre}\n(★ = feature relevante conocida)')
        ax.tick_params(axis='x', labelsize=8)
        fig.tight_layout()
        guardar_figura(fig, f'heatmap_{nombre.lower()}', FIG_DIR)
        logger.info("Heatmap guardado: %s", nombre)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    df, heatmaps = ejecutar()
    graficar_kci(df)
    graficar_heatmaps(heatmaps)

    logger.info("\nKCI medio por algoritmo:")
    resumen = df.groupby('algoritmo')['kci'].agg(['mean', 'std']).round(4)
    logger.info("\n%s", resumen.to_string())

    logger.info("\nPrecisión robusta (features relevantes en ≥80%% de bootstraps):")
    prec = df.groupby('algoritmo')['precision_robusta_80pct'].agg(['mean', 'std']).round(4)
    logger.info("\n%s", prec.to_string())

    logger.info("Experimento 09 completado en %.1f min", (time.time() - t0) / 60)
