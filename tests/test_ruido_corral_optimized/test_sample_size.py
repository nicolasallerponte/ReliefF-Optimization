"""
Test Simple: Sample Size (Pocos Datos)
=======================================

Evalúa cómo funcionan los algoritmos con datasets pequeños.
¿Necesitan muchos datos o convergen rápido?

Autor: Nicolás Aller
"""

import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

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


def make_corral(n=1000):
    """Dataset Corral"""
    np.random.seed(42)
    X = np.random.randn(n, 20).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] - X[:, 4] * X[:, 5] > 0).astype(int)
    return X, y


def eval_selector(selector, Xtr, Xte, ytr, yte):
    """Evalúa selector"""
    try:
        selector.fit(Xtr, ytr)
        Xtr_sel = selector.transform(Xtr)
        Xte_sel = selector.transform(Xte)

        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        clf.fit(Xtr_sel, ytr)

        f1 = f1_score(yte, clf.predict(Xte_sel), average='binary')
        return f1
    except:
        return 0.0  # Si falla con muy pocos datos


# =============================================================================
# MAIN
# =============================================================================

print("="*70)
print("TEST: SAMPLE SIZE (Pocos Datos)")
print("="*70)
print("\n¿Funcionan con datasets pequeños o necesitan muchos datos?\n")

# Datos (generamos un pool grande)
X, y = make_corral(1000)
_, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler_test = StandardScaler()
Xte = scaler_test.fit_transform(Xte)

print(f"Pool completo: 1000 samples")
print(f"Test set (fijo): {Xte.shape[0]} samples")
print(f"Submuestrearemos el train set\n")

# Algoritmos
algos = [
    (ReliefF_Original(n_neighbors=10, n_features_to_select=10), 'ReliefF'),
    (ANN(n_features_to_select=10), 'ANN'),
    (Proto(n_features_to_select=10), 'Proto'),
]

# EXPERIMENTO
print("-"*70)
print("EXPERIMENTO: Reduciendo tamaño del training set")
print("-"*70)

sample_sizes = [50, 100, 200, 300, 500, 800]
results = {name: [] for _, name in algos}

X_pool, y_pool = make_corral(1000)

for n_samples in sample_sizes:
    print(f"\nTrain size: {n_samples} samples:")

    # Submuestrear
    np.random.seed(42)
    indices = np.random.choice(len(X_pool), n_samples, replace=False)
    Xtr_sub = X_pool[indices]
    ytr_sub = y_pool[indices]

    # Escalar
    scaler = StandardScaler()
    Xtr_sub = scaler.fit_transform(Xtr_sub)
    Xte_scaled = scaler.transform(Xte)

    for selector, name in algos:
        f1 = eval_selector(selector, Xtr_sub, Xte_scaled, ytr_sub, yte)
        results[name].append(f1)
        print(f"  {name:15s}: F1={f1:.3f}")

# GRÁFICO
print("\n" + "-"*70)
print("Generando gráfico...")
print("-"*70)

plt.figure(figsize=(10, 5))
colors = {'ReliefF': '#95a5a6', 'ANN': '#3498db', 'Proto': '#e74c3c'}

for name in results.keys():
    color = colors.get(name, '#34495e')
    plt.plot(sample_sizes, results[name],
             marker='d', label=name, linewidth=2.5, markersize=8, color=color)

plt.xlabel('Training Set Size', fontsize=11)
plt.ylabel('F1-Score', fontsize=11)
plt.title('Convergencia con Pocos Datos', fontweight='bold', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim([0.4, 1.05])
plt.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5, label='Target 0.90')
plt.tight_layout()
plt.savefig('test_sample_size.png', dpi=120)
print("✓ Gráfico: test_sample_size.png")

# ANÁLISIS
print("\n" + "="*70)
print("ANÁLISIS")
print("="*70)

print("\nConvergencia (samples para F1 > 0.85):")
for name in results.keys():
    for i, (n, f1) in enumerate(zip(sample_sizes, results[name])):
        if f1 >= 0.85:
            emoji = "OK" if n <= 200 else ("OK" if n <= 500 else "NOK")
            print(f"{name:15s}: {n} samples {emoji}")
            break
    else:
        print(f"{name:15s}: No converge")

print("\nCOMPLETADO")
print("\nInterpretación:")
print("   • Convergencia rápida = funciona con pocos datos")
print("   • Crítico en medicina, materiales raros, etc.")
