"""
Test Simple: Features Redundantes (Copias Correlacionadas)
===========================================================

Añade features que son copias (con ruido) de las existentes.
¿Pueden los algoritmos distinguir originales de copias?

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

from relieff_improved.ann.ann_v2 import ANN
from relieff_improved.proto.proto_v2 import Proto
from relieff_improved.base.relieff_original import ReliefF_Original


def make_corral(n=800):
    """Dataset Corral"""
    np.random.seed(42)
    X = np.random.randn(n, 20).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] - X[:, 4] * X[:, 5] > 0).astype(int)
    return X, y


def add_redundant_features(X, n_redundant=10, correlation=0.9):
    """Añade features correlacionadas con las existentes"""
    np.random.seed(42)
    redundant = []

    for i in range(n_redundant):
        # Copia una feature informativa al azar
        source_idx = np.random.choice([0, 1, 2, 3, 4, 5])  # Solo las importantes

        # Crea copia con ruido controlado
        std = np.std(X[:, source_idx])
        noise = np.random.randn(X.shape[0]) * std * np.sqrt(1 - correlation**2)
        redundant_feature = X[:, source_idx] * correlation + noise
        redundant.append(redundant_feature)

    return np.hstack([X, np.column_stack(redundant)])


def eval_selector(selector, Xtr, Xte, ytr, yte):
    """Evalúa selector"""
    selector.fit(Xtr, ytr)
    Xtr_sel = selector.transform(Xtr)
    Xte_sel = selector.transform(Xte)

    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf.fit(Xtr_sel, ytr)

    f1 = f1_score(yte, clf.predict(Xte_sel), average='binary')
    return f1


# =============================================================================
# MAIN
# =============================================================================

print("="*70)
print("TEST: FEATURES REDUNDANTES (Copias Correlacionadas)")
print("="*70)
print("\n¿Pueden distinguir features originales de copias casi idénticas?\n")

# Datos
X, y = make_corral(800)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

print(f"Dataset base: {Xtr.shape}")
print(f"Features informativas: 6 (0,1,2,3,4,5)")
print(f"Correlación de copias: 0.85 (muy alta)\n")

# Algoritmos
algos = [
    (ReliefF_Original(n_neighbors=10, n_features_to_select=10), 'ReliefF'),
    (ANN(n_neighbors=10, n_features_to_select=10, iter_ratio=1.0,
            metric='euclidean'), 'ANN v2'),
    (Proto(n_features_to_select=10, sigma=0.1, k_protos=5,
              use_lvq=False, metric='euclidean'), 'Proto v2'),
]

# EXPERIMENTO
print("-"*70)
print("EXPERIMENTO: Añadiendo features redundantes gradualmente")
print("-"*70)

n_redundant_list = [0, 5, 10, 20, 30, 50]
results = {name: [] for _, name in algos}

for n_red in n_redundant_list:
    print(f"\n+ {n_red} features redundantes (total: {20 + n_red}):")

    if n_red == 0:
        Xtr_aug, Xte_aug = Xtr, Xte
    else:
        Xtr_aug = add_redundant_features(Xtr, n_red, correlation=0.85)
        Xte_aug = add_redundant_features(Xte, n_red, correlation=0.85)

    for selector, name in algos:
        f1 = eval_selector(selector, Xtr_aug, Xte_aug, ytr, yte)
        results[name].append(f1)
        print(f"  {name:15s}: F1={f1:.3f}")

# GRÁFICO
print("\n" + "-"*70)
print("Generando gráfico...")
print("-"*70)

plt.figure(figsize=(10, 5))
colors = {'ReliefF': '#95a5a6', 'ANN v2': '#3498db', 'Proto v2': '#e74c3c'}

for name in results.keys():
    color = colors.get(name, '#34495e')
    plt.plot(n_redundant_list, results[name],
             marker='s', label=name, linewidth=2.5, markersize=8, color=color)

plt.xlabel('# Features Redundantes Añadidas', fontsize=11)
plt.ylabel('F1-Score', fontsize=11)
plt.title('Robustez a Features Correlacionadas', fontweight='bold', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim([0.4, 1.05])
plt.tight_layout()
plt.savefig('test_redundant_features.png', dpi=120)
print("✓ Gráfico: test_redundant_features.png")

# ANÁLISIS
print("\n" + "="*70)
print("📊 ANÁLISIS")
print("="*70)

for name in results.keys():
    f1_clean = results[name][0]
    f1_50 = results[name][-1]
    deg = ((f1_clean - f1_50) / f1_clean) * 100
    emoji = "🏆" if deg < 10 else ("✓" if deg < 20 else "⚠️")
    print(f"{name:15s}: {f1_clean:.3f} → {f1_50:.3f} (↓{deg:.1f}%) {emoji}")

print("\n✅ COMPLETADO")
print("\n💡 Interpretación:")
print("   • Menor degradación = mejor discrimina originales de copias")
print("   • Importante cuando múltiples sensores miden lo mismo")
