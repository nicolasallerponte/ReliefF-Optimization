"""
Test Simple: Label Noise (Etiquetas Corruptas)
===============================================

Simula errores de anotación humana cambiando aleatoriamente
un porcentaje de las etiquetas.

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


def make_corral(n=800, noise=0.0):
    """Dataset Corral"""
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


def flip_labels(y, flip_ratio=0.1, random_state=42):
    """Corrompe flip_ratio% de las etiquetas"""
    np.random.seed(random_state)
    n_flips = int(len(y) * flip_ratio)
    flip_indices = np.random.choice(len(y), n_flips, replace=False)

    y_noisy = y.copy()
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]  # 0→1, 1→0

    print(f"    Corrupted: {n_flips}/{len(y)} labels ({flip_ratio*100:.1f}%)")
    return y_noisy


def eval_selector(selector, Xtr, Xte, ytr, yte):
    """Evalúa selector"""
    selector.fit(Xtr, ytr)  # ← Entrena con labels (posiblemente corruptas)

    Xtr_sel = selector.transform(Xtr)
    Xte_sel = selector.transform(Xte)

    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf.fit(Xtr_sel, ytr)

    # Evaluar en test set LIMPIO
    f1 = f1_score(yte, clf.predict(Xte_sel), average='binary')
    return f1


# =============================================================================
# MAIN
# =============================================================================

print("="*70)
print("TEST: LABEL NOISE (Etiquetas Corruptas)")
print("="*70)
print("\n¿Pueden los algoritmos encontrar features importantes cuando")
print("las etiquetas de entrenamiento están INCORRECTAS?\n")

# Datos
X, y = make_corral(800, noise=0.0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

print(f"Dataset: {Xtr.shape[0]} train, {Xte.shape[0]} test")
print(f"Ground truth: 6 features informativas (0,1,2,3,4,5)\n")

# Algoritmos
algos = [
    (ReliefF_Original(n_neighbors=10, n_features_to_select=10), 'ReliefF'),
    (ANN(n_features_to_select=10), 'ANN'),
    (Proto(n_features_to_select=10), 'Proto'),
]

# EXPERIMENTO
print("-"*70)
print("EXPERIMENTO: Inyectando label noise gradual")
print("-"*70)

flip_ratios = [0.0, 0.05, 0.10, 0.15, 0.20, 0.40, 0.50]
results = {name: [] for _, name in algos}

for flip in flip_ratios:
    print(f"\nLabel noise: {flip*100:.0f}%")

    # Corromper labels de train (test queda limpio)
    ytr_noisy = flip_labels(ytr, flip, random_state=41 + int(flip*100))

    for selector, name in algos:
        f1 = eval_selector(selector, Xtr, Xte, ytr_noisy, yte)
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
    plt.plot([f*100 for f in flip_ratios], results[name], 
             marker='o', label=name, linewidth=2.5, markersize=8, color=color)

plt.xlabel('Label Noise (%)', fontsize=11)
plt.ylabel('F1-Score', fontsize=11)
plt.title('Robustez a Etiquetas Corruptas', fontweight='bold', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim([0.4, 1.05])
plt.tight_layout()
plt.savefig('test_label_noise.png', dpi=120)
print("✓ Gráfico: test_label_noise.png")

# ANÁLISIS
print("\n" + "="*70)
print("ANÁLISIS")
print("="*70)

for name in results.keys():
    f1_clean = results[name][0]
    f1_30 = results[name][-1]
    deg = ((f1_clean - f1_30) / f1_clean) * 100
    emoji = "OK" if deg < 15 else ("OK" if deg < 25 else "BAD")
    print(f"{name:15s}: {f1_clean:.3f} → {f1_30:.3f} (↓{deg:.1f}%) {emoji}")

print("\nCOMPLETADO")
print("\nInterpretación:")
print("   Menor degradación = más robusto a errores de anotación")
print("   Importante en datasets con etiquetado humano imperfecto")
