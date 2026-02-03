"""
Test Simple con Dataset Real - Wine Quality
=============================================

Dataset: Wine Quality (red wine)
- 1599 muestras
- 11 features químicas (acidez, sulfatos, alcohol, etc.)
- Objetivo: Clasificar vino como "bueno" (quality >= 6) o "malo"

Autor: Nicolás Aller
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import time

# Setup
def setup():
    current = Path(__file__).parent.resolve()
    root = current
    while root != root.parent:
        if (root / 'src').exists():
            break
        root = root.parent
    sys.path.insert(0, str(root / 'src'))

setup()

from relieff_improved.ann.ann_v2_optimized import ANN
from relieff_improved.proto.proto_v2_optimized import Proto
from relieff_improved.base.relieff_original import ReliefF_Original


def load_wine_quality():
    """
    Carga Wine Quality desde sklearn o descarga si no existe
    """
    try:
        from sklearn.datasets import fetch_openml
        print("Descargando Wine Quality dataset...")
        wine = fetch_openml('wine-quality-red', version=1, as_frame=True, parser='auto')
        X = wine.data.values.astype(np.float32)
        y = wine.target.values.astype(int)
        feature_names = wine.feature_names
    except Exception as e:
        print(f"Error descargando: {e}")
        print("Usando dataset alternativo...")
        # Fallback a otro dataset interesante
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = data.data.astype(np.float32)
        y = data.target.astype(int)
        feature_names = data.feature_names

    # Convertir a clasificación binaria
    if len(np.unique(y)) > 2:
        median = np.median(y)
        y_binary = (y >= median).astype(int)
        print(f"Convertido a binario: calidad >= {median} = 'bueno' (1), < {median} = 'malo' (0)")
    else:
        y_binary = y

    return X, y_binary, feature_names


def add_noise_to_features(X, noise_ratio=0.1, random_state=42):
    """Añade ruido gaussiano a las features"""
    np.random.seed(random_state)
    noise = np.random.randn(*X.shape) * np.std(X, axis=0) * noise_ratio
    return X + noise


def eval_selector(selector, name, Xtr, Xte, ytr, yte):
    """Evalúa selector"""
    t0 = time.time()
    selector.fit(Xtr, ytr)
    t_fit = time.time() - t0

    # Seleccionar top features
    Xtr_sel = selector.transform(Xtr)
    Xte_sel = selector.transform(Xte)

    # Clasificador
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    clf.fit(Xtr_sel, ytr)

    y_pred = clf.predict(Xte_sel)
    acc = accuracy_score(yte, y_pred)
    f1 = f1_score(yte, y_pred, average='binary')

    # Top features
    rank = selector.rank()
    top5 = rank[:5]

    return {
        'name': name,
        'f1': f1,
        'acc': acc,
        'time': t_fit,
        'top5': top5
    }


# =============================================================================
# MAIN
# =============================================================================

print("="*70)
print("TEST CON DATASET REAL - Wine Quality")
print("="*70)

# Cargar datos
X, y, feature_names = load_wine_quality()
print(f"\nDataset cargado:")
print(f"  Samples: {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Distribución clases: {np.bincount(y)}")
print(f"  Feature names: {list(feature_names[:5])}...")

# Split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Escalar
scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

print(f"\nSplit: Train={Xtr.shape}, Test={Xte.shape}\n")

# Algoritmos
n_select = min(10, X.shape[1] // 2)
algos = [
    (ReliefF_Original(n_neighbors=10, n_features_to_select=n_select), 'ReliefF'),
    (ANN(n_features_to_select=n_select), 'ANN'),
    (Proto(n_features_to_select=n_select), 'Proto'),
]

# =============================================================================
# TEST 1: BASELINE
# =============================================================================
print("-"*70)
print("TEST 1: BASELINE (dataset limpio)")
print("-"*70)

results_baseline = []
for selector, name in algos:
    print(f"{name:15s} ... ", end="", flush=True)
    res = eval_selector(selector, name, Xtr, Xte, ytr, yte)
    results_baseline.append(res)

    # Mostrar nombres de top-5 features
    top_names = [str(feature_names[i])[:20] for i in res['top5']]

    print(f"F1={res['f1']:.3f}, Acc={res['acc']:.3f}, Time={res['time']:.1f}s")
    print(f"{'':17s} Top-5: {top_names}")

# =============================================================================
# TEST 2: RUIDO EN FEATURES (no en labels)
# =============================================================================
print("\n" + "-"*70)
print("TEST 2: RUIDO EN FEATURES (simulando errores de medición)")
print("-"*70)
print("\nAñadiendo ruido gaussiano a las mediciones...\n")

noise_ratios = [0.05, 0.10, 0.20, 0.30, 0.50]
results_noise = {name: [] for _, name in algos}

for noise_ratio in noise_ratios:
    print(f"Noise ratio={noise_ratio:.2f} (×std):")

    Xtr_n = add_noise_to_features(Xtr, noise_ratio, random_state=42)
    Xte_n = add_noise_to_features(Xte, noise_ratio, random_state=43)

    for selector, name in algos:
        res = eval_selector(selector, name, Xtr_n, Xte_n, ytr, yte)
        results_noise[name].append(res['f1'])
        print(f"  {name:15s}: F1={res['f1']:.3f}, Acc={res['acc']:.3f}")
    print()

# =============================================================================
# TEST 3: FEATURE DILUTION
# =============================================================================
print("-"*70)
print("TEST 3: DILUCIÓN (features irrelevantes)")
print("-"*70)
print("\nAñadiendo features aleatorias...\n")

n_random_list = [5, 10, 20, 50, 100]
results_dilution = {name: [] for _, name in algos}

for n_rand in n_random_list:
    print(f"+ {n_rand} features aleatorias (total: {X.shape[1] + n_rand}):")

    np.random.seed(42 + n_rand)
    rand_tr = np.random.randn(Xtr.shape[0], n_rand)
    rand_te = np.random.randn(Xte.shape[0], n_rand)

    Xtr_aug = np.hstack([Xtr, rand_tr])
    Xte_aug = np.hstack([Xte, rand_te])

    for selector, name in algos:
        res = eval_selector(selector, name, Xtr_aug, Xte_aug, ytr, yte)
        results_dilution[name].append(res['f1'])
        print(f"  {name:15s}: F1={res['f1']:.3f}, Acc={res['acc']:.3f}")
    print()

# =============================================================================
# GRÁFICOS
# =============================================================================
print("-"*70)
print("Generando gráficos...")
print("-"*70)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

colors = {'ReliefF': '#95a5a6', 'ANN': '#3498db', 'Proto': '#e74c3c'}

# Ruido en features
ax = axes[0]
for name in results_noise.keys():
    color = colors.get(name, '#34495e')
    ax.plot(noise_ratios, results_noise[name], marker='o', label=name, 
           linewidth=2.5, markersize=8, color=color)
ax.set_xlabel('Noise Ratio (×std)', fontsize=11)
ax.set_ylabel('F1-Score', fontsize=11)
ax.set_title('Robustez a Ruido en Mediciones', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([min(min(results_noise.values(), key=lambda x: min(x))) - 0.05, 1.0])

# Dilución
ax = axes[1]
for name in results_dilution.keys():
    color = colors.get(name, '#34495e')
    ax.plot(n_random_list, results_dilution[name], marker='s', label=name,
           linewidth=2.5, markersize=8, color=color)
ax.set_xlabel('# Features Aleatorias Añadidas', fontsize=11)
ax.set_ylabel('F1-Score', fontsize=11)
ax.set_title('Robustez a Aumento Dimensional', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([min(min(results_dilution.values(), key=lambda x: min(x))) - 0.05, 1.0])

plt.tight_layout()
plt.savefig('wine_quality_test.png', dpi=120, bbox_inches='tight')
print("\nGráfico: wine_quality_test.png")

# Guardar CSVs
df_noise = pd.DataFrame({
    'noise_ratio': noise_ratios,
    **{name: results_noise[name] for name in results_noise}
})
df_dilution = pd.DataFrame({
    'n_random': n_random_list,
    **{name: results_dilution[name] for name in results_dilution}
})

df_noise.to_csv('wine_noise_results.csv', index=False)
df_dilution.to_csv('wine_dilution_results.csv', index=False)
print("\nCSVs: wine_noise_results.csv, wine_dilution_results.csv")

# =============================================================================
# ANÁLISIS
# =============================================================================
print("\n" + "="*70)
print("ANÁLISIS")
print("="*70)

print("\nBASELINE (ordenado por F1):")
print("-"*70)
sorted_baseline = sorted(results_baseline, key=lambda x: x['f1'], reverse=True)
for res in sorted_baseline:
    print(f"{res['name']:15s}: F1={res['f1']:.3f}, Acc={res['acc']:.3f}, Time={res['time']:.1f}s")

print("\nROBUSTEZ A RUIDO:")
print("-"*70)
for name in results_noise.keys():
    f1_init = results_noise[name][0]
    f1_final = results_noise[name][-1]
    if f1_init > 0:
        degradation = ((f1_init - f1_final) / f1_init) * 100
        emoji = "OK" if degradation < 10 else ("OK" if degradation < 20 else "BAD")
        print(f"{name:15s}: {f1_init:.3f} → {f1_final:.3f} (↓{degradation:.1f}%) {emoji}")

print("\nROBUSTEZ A DILUCIÓN:")
print("-"*70)
for name in results_dilution.keys():
    f1_init = results_dilution[name][0]
    f1_final = results_dilution[name][-1]
    if f1_init > 0:
        degradation = ((f1_init - f1_final) / f1_init) * 100
        emoji = "OK" if degradation < 15 else ("OK" if degradation < 30 else "BAD")
        print(f"{name:15s}: {f1_init:.3f} → {f1_final:.3f} (↓{degradation:.1f}%) {emoji}")

print("\n" + "="*70)
print("COMPLETADO")
print("="*70)
print("\nArchivos generados:")
print("  - wine_quality_test.png")
print("  - wine_noise_results.csv")
print("  - wine_dilution_results.csv")
