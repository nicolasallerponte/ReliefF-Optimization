"""
Experimento 09 - Estabilidad de la selección de características.

Pregunta: ¿Seleccionan siempre los mismos algoritmos los mismos atributos
en ejecuciones distintas? ¿Cuál es el más estable?

Método: Índice de Consistencia de Kuncheva (KCI) calculado sobre los conjuntos
de features seleccionadas en 5 semillas distintas.

  KCI = 1  → selección idéntica en todas las ejecuciones
  KCI = 0  → estabilidad equivalente a selección aleatoria
  KCI < 0  → peor que el azar (inestabilidad)

Además se genera un heatmap de frecuencia de selección por feature para
visualizar cuáles son más robustamente seleccionadas.

Se evalúa en todos los datasets con ground truth conocido (sintéticos).

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

SEMILLAS   = CFG['experimento']['semillas']
N_SEL      = CFG['experimento']['n_features_seleccionadas']
ALGORITMOS = ['ReliefF', 'ANN', 'Proto']

# Solo datasets con ground truth (sintéticos)
DATASETS_SINTETICOS = [n for n, m in CATALOGO.items() if m.get('relevantes') is not None]


# ---------------------------------------------------------------------------
# Selección de features con un algoritmo
# ---------------------------------------------------------------------------

def seleccionar(algoritmo, X, y, semilla):
    n_sel = min(N_SEL, X.shape[1])
    if algoritmo == 'ReliefF':
        sel = ReliefF(n_features_to_select=n_sel, n_neighbors=10)
    elif algoritmo == 'ANN':
        sel = ANN(n_features_to_select=n_sel, random_state=semilla)
    else:
        sel = Proto(n_features_to_select=n_sel, n_jobs=1)
    sel.fit(X, y)
    ranking = sel.rank() if hasattr(sel, 'rank') else np.argsort(-sel.feature_importances_)
    return list(ranking[:n_sel])


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def ejecutar():
    filas = []
    heatmaps = {}   # {(dataset, algoritmo): matriz de frecuencia}

    for nombre in DATASETS_SINTETICOS:
        meta = CATALOGO[nombre]
        logger.info("Dataset: %s", nombre)

        # Usamos semilla base para obtener dimensiones del dataset
        X_ref, y_ref, relevantes = obtener_dataset(nombre, semilla=SEMILLAS[0])
        n_total = X_ref.shape[1]
        n_sel   = min(N_SEL, n_total)

        for alg in ALGORITMOS:
            subsets = []
            frecuencia = np.zeros(n_total, dtype=int)

            for semilla in SEMILLAS:
                X, y, _ = obtener_dataset(nombre, semilla=semilla)
                try:
                    subset = seleccionar(alg, X, y, semilla)
                    subsets.append(subset)
                    frecuencia[subset] += 1
                except Exception as e:
                    logger.warning("  Error %s %s semilla=%d: %s", alg, nombre, semilla, e)

            if len(subsets) < 2:
                continue

            kci = kuncheva_index(subsets, n_total, n_sel)
            heatmaps[(nombre, alg)] = frecuencia / len(subsets)   # frecuencia relativa [0,1]

            # Precisión de relevantes: cuántas de las features relevantes aparecen en TODOS los subsets
            if relevantes:
                precision_todos = sum(
                    1 for r in relevantes if all(r in s for s in subsets)
                ) / len(relevantes)
            else:
                precision_todos = float('nan')

            filas.append({
                'dataset':          nombre,
                'algoritmo':        alg,
                'n_total_features': n_total,
                'n_seleccionadas':  n_sel,
                'kci':              round(kci, 4),
                'precision_siempre_relevantes': round(precision_todos, 4),
            })
            logger.info("  %s KCI=%.3f  siempre_relevantes=%.2f",
                        alg, kci, precision_todos)

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
    ax.set_title('Estabilidad de la selección de características\n(KCI, 5 semillas; 1=perfecta estabilidad, 0=aleatoria)')
    ax.set_ylim(-0.2, 1.1)
    ax.legend()
    guardar_figura(fig, 'kci_barplot', FIG_DIR)
    logger.info("Figura guardada: kci_barplot.png")


def graficar_heatmaps(heatmaps, n_mostrar=15):
    """Heatmap de frecuencia de selección por feature y dataset."""
    datasets_unicos = sorted({k[0] for k in heatmaps})

    for nombre in datasets_unicos:
        _, relevantes = CATALOGO[nombre].get('relevantes', None), CATALOGO[nombre].get('relevantes')

        datos = []
        for alg in ALGORITMOS:
            if (nombre, alg) in heatmaps:
                freq = heatmaps[(nombre, alg)]
                # Mostrar solo las n_mostrar features más frecuentemente seleccionadas
                top_idx = np.argsort(-freq)[:n_mostrar]
                datos.append({'algoritmo': alg, 'indices': top_idx, 'freq': freq})

        if not datos:
            continue

        # Unión de features más frecuentes
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
            cbar_kws={'label': 'Frecuencia de selección (5 semillas)'},
        )
        ax.set_title(f'Frecuencia de selección - {nombre}\n(★ = feature relevante conocida)')
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
    logger.info("Experimento 09 completado en %.1f min", (time.time() - t0) / 60)
