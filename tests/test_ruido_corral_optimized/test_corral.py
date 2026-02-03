"""
Test Simple Corral - Ver cómo funcionan los algoritmos paso a paso
===================================================================

Autor: Nicolás Aller
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

# Dataset Corral simple
def make_corral(n=800, noise=0.0):
    """y = sign(X1*X2 + X3*X4 - X5*X6)"""
    np.random.seed(42)
    X = np.random.randn(n, 20).astype(np.float32)

    y = (
        X[:, 0] * X[:, 1] +
        X[:, 2] * X[:, 3] -
        X[:, 4] * X[:, 5]
    )

    if noise > 0:
        y += np.random.randn(n) * noise

    y = (y > 0).astype(int)
    return X, y

# Evaluar simple
def eval_selector(selector, name, Xtr, Xte, ytr, yte):
    """Evalúa y devuelve F1 y accuracy"""
    t0 = time.time()
    selector.fit(Xtr, ytr)
    t_fit = time.time() - t0

    # Ranking
    rank = selector.rank()
    top6 = set(rank[:6])
    true = {0, 1, 2, 3, 4, 5}

    tp = len(top6 & true)
    f1 = tp / 6  # Simplificado: F1 = recall (las 6 primeras)
    perfect = (top6 == true)

    # Accuracy
    Xtr_sel = selector.transform(Xtr)
    Xte_sel = selector.transform(Xte)
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf.fit(Xtr_sel, ytr)
    acc = accuracy_score(yte, clf.predict(Xte_sel))

    return {
        'name': name,
        'f1': f1,
        'acc': acc,
        'time': t_fit,
        'perfect': perfect,
        'top6': list(rank[:6])
    }

# =============================================================================
# MAIN
# =============================================================================

print("="*70)
print("TEST CORRAL SIMPLE - Paso a paso")
print("="*70)
print("\nDataset: 800 samples, 20 features (6 informativas: 0,1,2,3,4,5)")
print("Regla: y = sign(X₁·X₂ + X₃·X₄ - X₅·X₆)\n")

# Preparar datos
X, y = make_corral(800, noise=0.0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

print(f"Train: {Xtr.shape}, Test: {Xte.shape}\n")

# Algoritmos (CONFIGURACIÓN SIMPLE Y ESTABLE)
algos = [
    (ReliefF_Original(n_neighbors=10, n_features_to_select=10), 'ReliefF'),
    (ANN(n_features_to_select=10), 'ANN'),
    (Proto(n_features_to_select=10), 'Proto'),
]

# =============================================================================
# TEST 1: BASELINE (sin ruido)
# =============================================================================
print("-"*70)
print("TEST 1: BASELINE (dataset limpio)")
print("-"*70)

results_baseline = []
for selector, name in algos:
    print(f"{name:15s} ... ", end="", flush=True)
    res = eval_selector(selector, name, Xtr, Xte, ytr, yte)
    results_baseline.append(res)

    perf_str = "✓" if res['perfect'] else "✗"
    print(f"F1={res['f1']:.2f}, Acc={res['acc']:.3f}, Perfect={perf_str}, Time={res['time']:.1f}s")
    print(f"{'':17s} Top-6: {res['top6']}")

# =============================================================================
# TEST 2: RUIDO GAUSSIANO (gradual)
# =============================================================================
print("\n" + "-"*70)
print("TEST 2: RUIDO GAUSSIANO (inyección gradual)")
print("-"*70)
print("\nInyectando ruido en la regla de decisión...\n")

noise_levels = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
results_noise = {name: [] for _, name in algos}

for noise in noise_levels:
    print(f"Noise σ={noise:.1f}:")

    Xn, yn = make_corral(800, noise=noise)
    Xtr_n, Xte_n, ytr_n, yte_n = train_test_split(Xn, yn, test_size=0.25, random_state=42, stratify=yn)
    sc = StandardScaler()
    Xtr_n = sc.fit_transform(Xtr_n)
    Xte_n = sc.transform(Xte_n)

    for selector, name in algos:
        res = eval_selector(selector, name, Xtr_n, Xte_n, ytr_n, yte_n)
        results_noise[name].append(res['f1'])
        print(f"  {name:15s}: F1={res['f1']:.2f}, Acc={res['acc']:.3f}")
    print()

# =============================================================================
# TEST 3: DILUCIÓN (añadir features aleatorias)
# =============================================================================
print("-"*70)
print("TEST 3: DILUCIÓN DIMENSIONAL (features aleatorias)")
print("-"*70)
print("\nAñadiendo features aleatorias...\n")

n_random_list = [10, 20, 50, 100, 200]
results_dilution = {name: [] for _, name in algos}

for n_rand in n_random_list:
    print(f"+ {n_rand} features (total: {20 + n_rand}):")

    np.random.seed(42 + n_rand)
    rand_tr = np.random.randn(Xtr.shape[0], n_rand)
    rand_te = np.random.randn(Xte.shape[0], n_rand)

    Xtr_aug = np.hstack([Xtr, rand_tr])
    Xte_aug = np.hstack([Xte, rand_te])

    for selector, name in algos:
        res = eval_selector(selector, name, Xtr_aug, Xte_aug, ytr, yte)
        results_dilution[name].append(res['f1'])
        print(f"  {name:15s}: F1={res['f1']:.2f}, Acc={res['acc']:.3f}")
    print()

# =============================================================================
# GRÁFICOS SIMPLES
# =============================================================================
print("-"*70)
print("Generando gráficos...")
print("-"*70)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gráfico 1: Ruido gaussiano
ax = axes[0]
for name in results_noise.keys():
    ax.plot(noise_levels, results_noise[name], marker='o', label=name, linewidth=2)
ax.set_xlabel('Noise Level (σ)')
ax.set_ylabel('F1')
ax.set_title('Robustez a Ruido Gaussiano')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Gráfico 2: Dilución
ax = axes[1]
for name in results_dilution.keys():
    ax.plot(n_random_list, results_dilution[name], marker='s', label=name, linewidth=2)
ax.set_xlabel('# Features Aleatorias Añadidas')
ax.set_ylabel('F1')
ax.set_title('Robustez a Aumento Dimensional')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('corral_simple.png', dpi=120)
print("\n✓ Gráfico guardado: corral_simple.png")

# Guardar CSV
df_noise = pd.DataFrame({
    'noise_level': noise_levels,
    **{name: results_noise[name] for name in results_noise}
})
df_dilution = pd.DataFrame({
    'n_random': n_random_list,
    **{name: results_dilution[name] for name in results_dilution}
})

df_noise.to_csv('noise_results.csv', index=False)
df_dilution.to_csv('dilution_results.csv', index=False)
print("✓ CSVs: noise_results.csv, dilution_results.csv")

print("\n" + "="*70)
print("COMPLETADO")
print("="*70)
print("\nResumen:")
print(f"  Baseline: Todos F1~{results_baseline[0]['f1']:.2f}")
print(f"  Ruido máximo: ReliefF={results_noise['ReliefF'][-1]:.2f}, " +
      f"ANN={results_noise['ANN'][-1]:.2f}, Proto={results_noise['Proto'][-1]:.2f}")
print(f"  Dilución máxima: ReliefF={results_dilution['ReliefF'][-1]:.2f}, " +
      f"ANN={results_dilution['ANN'][-1]:.2f}, Proto={results_dilution['Proto'][-1]:.2f}")
print("\nRevisa corral_simple.png para ver las curvas de degradación.")
