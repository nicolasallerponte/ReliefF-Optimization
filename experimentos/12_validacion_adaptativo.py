"""
Experimento 12 - Validación de fórmulas adaptativas.

Pregunta: ¿Seleccionan automáticamente los parámetros óptimos las fórmulas
adaptativas de k (ANN) y sigma (Proto) sin necesidad de búsqueda exhaustiva?

Las fórmulas se activan con los valores por defecto:
  ANN  (n_neighbors=15) : k = 15 / (1 + ratio × 5.0), clip [5, 15]
  Proto (sigma=0.15)    : sigma_eff = clip(0.10 × (1 + log(ratio / 0.05)), 0.10, 0.25)

Método: para cada dataset se compara el F1 del modo adaptativo contra los
valores fijos del grid (n_neighbors ∈ {5, 10, 20}; sigma ∈ {0.10, 0.20, 0.25}).
Se reporta el delta F1 respecto al mejor fijo por dataset, y si el adaptativo
cae dentro de ±0.02 del máximo se considera "equivalente al óptimo".

Salidas:
  results/tablas/12_validacion_adaptativo/ann_adaptativo.csv + .tex
  results/tablas/12_validacion_adaptativo/proto_adaptativo.csv + .tex
  results/figuras/12_validacion_adaptativo/curva_k_adaptativo.png
  results/figuras/12_validacion_adaptativo/curva_sigma_adaptativo.png
  results/figuras/12_validacion_adaptativo/delta_f1_ann.png
  results/figuras/12_validacion_adaptativo/delta_f1_proto.png
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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from relieff_opt import ANN, Proto
from relieff_opt.utils.conjuntos import obtener_todos
from relieff_opt.utils.paleta import (
    COLORES, nueva_figura, guardar_figura,
)
from relieff_opt.utils.experimento import guardar_tabla

CFG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'hiperparametros.yaml'
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

FIG_DIR = 'figuras/12_validacion_adaptativo'
TAB_DIR = 'tablas/12_validacion_adaptativo'

SEMILLAS   = CFG['experimento']['semillas']
N_CV       = CFG['experimento']['cv_pliegues']

# Valores fijos del grid a comparar (no incluir 15 ni 0.15 porque activan adaptativo)
K_FIJOS    = [5, 10, 20]
SIGMA_FIJOS = [0.10, 0.20, 0.25]

COLOR_ADAPT  = COLORES['ANN']      # azul para ANN adaptativo
COLOR_ANN    = '#95c8f0'           # azul claro para ANN fijo
COLOR_PADAPT = COLORES['Proto']    # rojo para Proto adaptativo
COLOR_PROTO  = '#f0a0a0'           # rojo claro para Proto fijo


# ---------------------------------------------------------------------------
# Fórmulas analíticas (para las curvas)
# ---------------------------------------------------------------------------

def formula_k(ratio):
    """Devuelve k adaptativo dado el ratio d/n."""
    k = 15.0 / (1 + ratio * 5.0)
    return int(np.clip(k, 5, 15))


def formula_sigma(ratio):
    """Devuelve sigma_eff adaptativo dado el ratio d/n."""
    base, ratio_base = 0.10, 0.05
    if ratio <= ratio_base:
        sigma_eff = base
    else:
        sigma_eff = base * (1 + np.log(ratio / ratio_base))
    return float(np.clip(sigma_eff, 0.10, 0.25))


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluar(selector, X, y, semilla):
    """5-fold CV con RF; devuelve F1 medio."""
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=semilla)
    scores = []
    for idx_tr, idx_te in skf.split(X, y):
        Xtr, Xte = X[idx_tr], X[idx_te]
        ytr, yte = y[idx_tr], y[idx_te]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        try:
            selector.fit(Xtr, ytr)
            Xtr_s = selector.transform(Xtr)
            Xte_s = selector.transform(Xte)
            clf = RandomForestClassifier(
                n_estimators=CFG['experimento']['rf_n_estimators'],
                max_depth=CFG['experimento']['rf_max_depth'],
                random_state=semilla, n_jobs=1,
            )
            clf.fit(Xtr_s, ytr)
            scores.append(f1_score(yte, clf.predict(Xte_s), average='binary'))
        except Exception as e:
            logger.warning("  Error en fold: %s", e)
            scores.append(0.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Experimento ANN
# ---------------------------------------------------------------------------

def experimento_ann(datasets):
    logger.info("ANN - adaptativo vs fijos %s", K_FIJOS)
    filas = []
    for nombre, (X, y, _) in datasets.items():
        n, d = X.shape
        ratio = d / n
        k_adapt = formula_k(ratio)
        n_sel = min(CFG['ann']['n_features_to_select'], d)
        logger.info("  %s  ratio=%.4f  k_adapt=%d", nombre, ratio, k_adapt)

        # Adaptativo
        f1s_adapt = [
            evaluar(ANN(n_neighbors=15, n_features_to_select=n_sel, random_state=s), X, y, s)
            for s in SEMILLAS
        ]
        fila = {
            'dataset':    nombre,
            'n':          n,
            'd':          d,
            'ratio':      round(ratio, 5),
            'k_adaptativo': k_adapt,
            'f1_adaptativo': round(float(np.mean(f1s_adapt)), 4),
            'std_adaptativo': round(float(np.std(f1s_adapt)), 4),
        }

        # Fijos
        for k in K_FIJOS:
            f1s = [
                evaluar(ANN(n_neighbors=k, n_features_to_select=n_sel, random_state=s), X, y, s)
                for s in SEMILLAS
            ]
            fila[f'f1_k{k}'] = round(float(np.mean(f1s)), 4)

        # Delta vs mejor fijo
        mejor_fijo = max(fila[f'f1_k{k}'] for k in K_FIJOS)
        fila['mejor_fijo']  = round(mejor_fijo, 4)
        fila['delta_f1']    = round(fila['f1_adaptativo'] - mejor_fijo, 4)
        fila['equivalente'] = abs(fila['delta_f1']) <= 0.02

        filas.append(fila)

    df = pd.DataFrame(filas).sort_values('ratio').reset_index(drop=True)
    guardar_tabla(df, 'ann_adaptativo', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Experimento Proto
# ---------------------------------------------------------------------------

def experimento_proto(datasets):
    logger.info("Proto - adaptativo (sigma=0.15) vs fijos %s", SIGMA_FIJOS)
    filas = []
    for nombre, (X, y, _) in datasets.items():
        n, d = X.shape
        ratio = d / n
        sigma_adapt = formula_sigma(ratio)
        n_sel = min(CFG['proto']['n_features_to_select'], d)
        logger.info("  %s  ratio=%.4f  sigma_adapt=%.4f", nombre, ratio, sigma_adapt)

        # Adaptativo (sigma=0.15 activa la fórmula)
        f1s_adapt = [
            evaluar(Proto(sigma=0.15, n_features_to_select=n_sel, use_lvq=False, n_jobs=1), X, y, s)
            for s in SEMILLAS
        ]
        fila = {
            'dataset':       nombre,
            'n':             n,
            'd':             d,
            'ratio':         round(ratio, 5),
            'sigma_adaptativo': round(sigma_adapt, 4),
            'f1_adaptativo':    round(float(np.mean(f1s_adapt)), 4),
            'std_adaptativo':   round(float(np.std(f1s_adapt)), 4),
        }

        # Fijos
        for sigma in SIGMA_FIJOS:
            f1s = [
                evaluar(Proto(sigma=sigma, n_features_to_select=n_sel, use_lvq=False, n_jobs=1), X, y, s)
                for s in SEMILLAS
            ]
            fila[f'f1_s{str(sigma).replace(".", "")}'] = round(float(np.mean(f1s)), 4)

        # Delta vs mejor fijo
        mejor_fijo = max(fila[f'f1_s{str(s).replace(".", "")}'] for s in SIGMA_FIJOS)
        fila['mejor_fijo']  = round(mejor_fijo, 4)
        fila['delta_f1']    = round(fila['f1_adaptativo'] - mejor_fijo, 4)
        fila['equivalente'] = abs(fila['delta_f1']) <= 0.02

        filas.append(fila)

    df = pd.DataFrame(filas).sort_values('ratio').reset_index(drop=True)
    guardar_tabla(df, 'proto_adaptativo', TAB_DIR)
    return df


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def graficar_curva_k():
    """Curva analítica k adaptativo vs ratio, con scatter de los 12 datasets."""
    ratios = np.linspace(0, 0.25, 300)
    ks = [formula_k(r) for r in ratios]

    fig, ax = nueva_figura()
    ax.plot(ratios, ks, color=COLOR_ADAPT, linewidth=2.5, label='k adaptativo (fórmula)')
    for k_f in K_FIJOS:
        ax.axhline(k_f, color=COLOR_ANN, linewidth=1.0, linestyle='--', alpha=0.7, label=f'k fijo = {k_f}')
    ax.set_xlabel('Ratio features/muestras (d/n)')
    ax.set_ylabel('k seleccionado')
    ax.set_title('Fórmula adaptativa de k — ANN\n'
                 r'$k = \lfloor 15 / (1 + \mathrm{ratio} \times 5) \rfloor$, clip [5, 15]')
    ax.legend()
    guardar_figura(fig, 'curva_k_adaptativo', FIG_DIR)


def graficar_curva_sigma():
    """Curva analítica sigma_eff vs ratio, con líneas de los valores fijos."""
    ratios = np.linspace(0, 0.25, 300)
    sigmas = [formula_sigma(r) for r in ratios]

    fig, ax = nueva_figura()
    ax.plot(ratios, sigmas, color=COLOR_PADAPT, linewidth=2.5, label='sigma adaptativo (fórmula)')
    for s_f in SIGMA_FIJOS:
        ax.axhline(s_f, color=COLOR_PROTO, linewidth=1.0, linestyle='--', alpha=0.7, label=f'sigma fijo = {s_f}')
    ax.set_xlabel('Ratio features/muestras (d/n)')
    ax.set_ylabel('sigma efectivo')
    ax.set_title('Fórmula adaptativa de sigma — Proto\n'
                 r'$\sigma_{eff} = \mathrm{clip}(0.10 \times (1 + \ln(\mathrm{ratio}/0.05)),\ 0.10,\ 0.25)$')
    ax.legend()
    guardar_figura(fig, 'curva_sigma_adaptativo', FIG_DIR)


def graficar_delta(df, algoritmo, col_adapt, cols_fijos, color_adapt, color_fijo):
    """
    Gráfica de barras: F1 adaptativo vs F1 de cada valor fijo, por dataset
    ordenado por ratio creciente.
    """
    datasets = df['dataset'].tolist()
    x = np.arange(len(datasets))
    n_fijos = len(cols_fijos)
    ancho = 0.15

    fig, ax = nueva_figura(tamano=(12, 4.5))

    # Barras fijas
    for i, col in enumerate(cols_fijos):
        offset = (i - n_fijos / 2) * ancho
        ax.bar(x + offset, df[col], width=ancho, color=color_fijo,
               alpha=0.6 + 0.1 * i, label=col.replace('f1_', ''))

    # Barra adaptativa
    ax.bar(x + (n_fijos / 2) * ancho, df[col_adapt], width=ancho,
           color=color_adapt, label='adaptativo', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row['dataset']}\n(r={row['ratio']:.3f})" for _, row in df.iterrows()],
        rotation=30, ha='right', fontsize=8,
    )
    ax.set_ylabel('F1 medio (5-fold CV, 5 semillas)')
    ax.set_ylim(bottom=max(0, df[[col_adapt] + cols_fijos].min().min() - 0.05))
    ax.set_title(f'{algoritmo}: F1 adaptativo vs fijos — datasets ordenados por ratio d/n')
    ax.legend(fontsize=9)
    fig.tight_layout()
    nombre_fig = f'delta_f1_{algoritmo.lower()}'
    guardar_figura(fig, nombre_fig, FIG_DIR)
    logger.info("Figura guardada: %s.png", nombre_fig)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()

    logger.info("Cargando datasets...")
    datasets = obtener_todos(semilla=42)

    # Curvas analíticas (no requieren datos)
    graficar_curva_k()
    graficar_curva_sigma()

    # ANN
    df_ann = experimento_ann(datasets)
    cols_fijos_ann = [f'f1_k{k}' for k in K_FIJOS]
    graficar_delta(df_ann, 'ANN', 'f1_adaptativo', cols_fijos_ann, COLOR_ADAPT, COLOR_ANN)

    n_equiv_ann = df_ann['equivalente'].sum()
    logger.info("ANN: %d/%d datasets equivalentes al mejor fijo (Δ ≤ 0.02)",
                n_equiv_ann, len(df_ann))
    logger.info("ANN delta medio: %.4f ± %.4f",
                df_ann['delta_f1'].mean(), df_ann['delta_f1'].std())

    # Proto
    df_proto = experimento_proto(datasets)
    cols_fijos_proto = [f'f1_s{str(s).replace(".", "")}' for s in SIGMA_FIJOS]
    graficar_delta(df_proto, 'Proto', 'f1_adaptativo', cols_fijos_proto, COLOR_PADAPT, COLOR_PROTO)

    n_equiv_proto = df_proto['equivalente'].sum()
    logger.info("Proto: %d/%d datasets equivalentes al mejor fijo (Δ ≤ 0.02)",
                n_equiv_proto, len(df_proto))
    logger.info("Proto delta medio: %.4f ± %.4f",
                df_proto['delta_f1'].mean(), df_proto['delta_f1'].std())

    logger.info("Experimento 12 completado en %.1f min", (time.time() - t0) / 60)
