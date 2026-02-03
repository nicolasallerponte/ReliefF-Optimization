"""
Comprehensive Hyperparameter Search - PRODUCTION VERSION
=========================================================

Benchmark exhaustivo con:
- 10+ datasets reales (sklearn + UCI)
- Múltiples tamaños (100 a 10000+ samples)
- Grid denso de hiperparámetros
- Análisis estadístico completo
- Gráficos del Top-5
- Duración: ~15-30 minutos

Autor: Nicolás Aller
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.datasets import (
    load_breast_cancer, load_digits, load_wine,
    make_classification, make_moons, make_circles
)
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time
import warnings
warnings.filterwarnings('ignore')

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


# =============================================================================
# DATASETS
# =============================================================================

def load_all_datasets():
    """Carga 10+ datasets reales y sintéticos"""
    datasets = {}

    print("Cargando datasets...")

    # 1. BREAST CANCER (sklearn)
    try:
        data = load_breast_cancer()
        X, y = data.data, data.target
        datasets['Breast_Cancer'] = (X, y, 'real', 'medium')
        print(f"  ✓ Breast Cancer: {X.shape}")
    except:
        pass

    # 2. WINE (sklearn)
    try:
        data = load_wine()
        X, y = data.data, (data.target > 0).astype(int)  # Binario
        datasets['Wine'] = (X, y, 'real', 'small')
        print(f"  ✓ Wine: {X.shape}")
    except:
        pass

    # 3. DIGITS (sklearn) - Binario
    try:
        data = load_digits()
        # Convertir a binario: 0-4 vs 5-9
        X, y = data.data, (data.target >= 5).astype(int)
        datasets['Digits'] = (X, y, 'real', 'large')
        print(f"  ✓ Digits: {X.shape}")
    except:
        pass

    # 4-6. SINTÉTICOS CLÁSICOS
    np.random.seed(42)

    # Corral
    X = np.random.randn(800, 20).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] - X[:, 4] * X[:, 5] > 0).astype(int)
    datasets['Corral'] = (X, y, 'synthetic', 'medium')
    print(f"  ✓ Corral: {X.shape}")

    # XOR
    X = np.random.randn(1000, 10).astype(np.float32)
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
    datasets['XOR'] = (X, y, 'synthetic', 'medium')
    print(f"  ✓ XOR: {X.shape}")

    # Moons
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    datasets['Moons'] = (X, y, 'synthetic', 'medium')
    print(f"  ✓ Moons: {X.shape}")

    # Circles
    X, y = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42)
    datasets['Circles'] = (X, y, 'synthetic', 'medium')
    print(f"  ✓ Circles: {X.shape}")

    # 7-10. SINTÉTICOS VARIADOS

    # High-dimensional (muchas features)
    X, y = make_classification(
        n_samples=500, n_features=100, n_informative=10, n_redundant=20,
        n_repeated=10, n_clusters_per_class=2, random_state=42
    )
    datasets['HighDim'] = (X, y, 'synthetic', 'high_dim')
    print(f"  ✓ High Dimensional: {X.shape}")

    # Imbalanced
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        weights=[0.7, 0.3], flip_y=0.1, random_state=42
    )
    datasets['Imbalanced'] = (X, y, 'synthetic', 'imbalanced')
    print(f"  ✓ Imbalanced: {X.shape}")

    # Noisy
    X, y = make_classification(
        n_samples=800, n_features=25, n_informative=10,
        n_redundant=5, flip_y=0.2, random_state=42
    )
    datasets['Noisy'] = (X, y, 'synthetic', 'noisy')
    print(f"  ✓ Noisy: {X.shape}")

    # Large (muchas muestras)
    X, y = make_classification(
        n_samples=5000, n_features=30, n_informative=20,
        n_redundant=5, random_state=42
    )
    datasets['Large'] = (X, y, 'synthetic', 'large')
    print(f"  ✓ Large: {X.shape}")

    print(f"\n→ Total: {len(datasets)} datasets cargados\n")
    return datasets


# =============================================================================
# EVALUACIÓN ROBUSTA
# =============================================================================

def eval_config_cv(selector, X, y, n_splits=3):
    """Evaluación con cross-validation"""
    scores_f1 = []
    scores_acc = []
    times = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        # Escalar
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        try:
            # Fit selector
            t0 = time.time()
            selector.fit(Xtr, ytr)
            t_fit = time.time() - t0

            # Transform
            Xtr_sel = selector.transform(Xtr)
            Xte_sel = selector.transform(Xte)

            # Clasificador
            clf = RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=42
            )
            clf.fit(Xtr_sel, ytr)

            # Métricas
            y_pred = clf.predict(Xte_sel)
            f1 = f1_score(yte, y_pred, average='binary')
            acc = accuracy_score(yte, y_pred)

            scores_f1.append(f1)
            scores_acc.append(acc)
            times.append(t_fit)

        except Exception as e:
            # Si falla, penalizar
            scores_f1.append(0.0)
            scores_acc.append(0.0)
            times.append(999.0)

    return {
        'f1_mean': np.mean(scores_f1),
        'f1_std': np.std(scores_f1),
        'acc_mean': np.mean(scores_acc),
        'acc_std': np.std(scores_acc),
        'time_mean': np.mean(times),
        'time_std': np.std(times)
    }


# =============================================================================
# GRID SEARCH EXHAUSTIVO
# =============================================================================

def grid_search_ann_comprehensive(datasets):
    """Grid search denso para ANN"""
    print("="*70)
    print("COMPREHENSIVE GRID SEARCH: ANN")
    print("="*70)

    # Grid más denso
    configs = []

    # Combinatoria de hiperparámetros principales
    for n_neigh in [5, 10, 15, 20, 25]:
        for metric in ['euclidean', 'manhattan']:
            for iter_ratio in [0.3, 0.5, 0.8, 1.0, 'auto']:
                configs.append({
                    'n_neighbors': n_neigh,
                    'metric': metric,
                    'iter_ratio': iter_ratio,
                    'M': 16,  # Fijo por ahora
                    'ef_construction': 200  # Fijo
                })

    # Variar M y ef_construction con mejores configs
    for M, ef in [(24, 200), (32, 300), (16, 100)]:
        configs.append({
            'n_neighbors': 10,
            'metric': 'euclidean',
            'iter_ratio': 'auto',
            'M': M,
            'ef_construction': ef
        })

    print(f"\nTotal configuraciones: {len(configs)}\n")

    results = []
    total = len(configs)

    for i, config in enumerate(configs):
        print(f"Config {i+1}/{total}: n_neigh={config['n_neighbors']}, "
              f"metric={config['metric']}, iter={config['iter_ratio']}")

        row = config.copy()

        # Evaluar en TODOS los datasets
        for ds_name, (X, y, ds_type, ds_cat) in datasets.items():
            ann = ANN(n_features_to_select=min(10, X.shape[1]//2), **config)
            metrics = eval_config_cv(ann, X, y, n_splits=3)

            row[f'{ds_name}_f1'] = metrics['f1_mean']
            row[f'{ds_name}_f1_std'] = metrics['f1_std']
            row[f'{ds_name}_time'] = metrics['time_mean']

            print(f"  {ds_name:15s}: F1={metrics['f1_mean']:.3f}±{metrics['f1_std']:.3f}, "
                  f"Time={metrics['time_mean']:.2f}s")

        # Promedios
        f1_cols = [col for col in row.keys() if col.endswith('_f1')]
        row['Avg_F1'] = np.mean([row[col] for col in f1_cols])
        row['Std_F1'] = np.std([row[col] for col in f1_cols])

        time_cols = [col for col in row.keys() if col.endswith('_time')]
        row['Avg_Time'] = np.mean([row[col] for col in time_cols])

        results.append(row)
        print()

    # DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('Avg_F1', ascending=False)

    print("\n" + "="*70)
    print("TOP 10 CONFIGURACIONES:")
    print("="*70)
    cols_to_show = ['n_neighbors', 'metric', 'iter_ratio', 'M', 'Avg_F1', 'Std_F1', 'Avg_Time']
    print(df[cols_to_show].head(10).to_string(index=False))

    # Guardar
    df.to_csv('ann_comprehensive_search.csv', index=False)
    print("\n✓ Guardado: ann_comprehensive_search.csv")

    return df


def grid_search_proto_comprehensive(datasets):
    """Grid search denso para Proto"""
    print("\n\n" + "="*70)
    print("COMPREHENSIVE GRID SEARCH: PROTO")
    print("="*70)

    # Grid más denso
    configs = []

    # Combinatoria de hiperparámetros principales
    for k_proto in [3, 5, 10, 15, 20]:
        for sigma in [0.05, 0.1, 0.2, 0.3, 0.5]:
            for use_lvq in [True, False]:
                for metric in ['euclidean', 'manhattan']:
                    configs.append({
                        'k_protos': k_proto,
                        'sigma': sigma,
                        'use_lvq': use_lvq,
                        'metric': metric
                    })

    print(f"\nTotal configuraciones: {len(configs)}\n")

    results = []
    total = len(configs)

    for i, config in enumerate(configs):
        print(f"Config {i+1}/{total}: k={config['k_protos']}, "
              f"σ={config['sigma']}, lvq={config['use_lvq']}, "
              f"metric={config['metric']}")

        row = config.copy()

        # Evaluar en TODOS los datasets
        for ds_name, (X, y, ds_type, ds_cat) in datasets.items():
            proto = Proto(n_features_to_select=min(10, X.shape[1]//2), **config)
            metrics = eval_config_cv(proto, X, y, n_splits=3)

            row[f'{ds_name}_f1'] = metrics['f1_mean']
            row[f'{ds_name}_f1_std'] = metrics['f1_std']
            row[f'{ds_name}_time'] = metrics['time_mean']

            print(f"  {ds_name:15s}: F1={metrics['f1_mean']:.3f}±{metrics['f1_std']:.3f}, "
                  f"Time={metrics['time_mean']:.2f}s")

        # Promedios
        f1_cols = [col for col in row.keys() if col.endswith('_f1')]
        row['Avg_F1'] = np.mean([row[col] for col in f1_cols])
        row['Std_F1'] = np.std([row[col] for col in f1_cols])

        time_cols = [col for col in row.keys() if col.endswith('_time')]
        row['Avg_Time'] = np.mean([row[col] for col in time_cols])

        results.append(row)
        print()

    # DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('Avg_F1', ascending=False)

    print("\n" + "="*70)
    print("TOP 10 CONFIGURACIONES:")
    print("="*70)
    cols_to_show = ['k_protos', 'sigma', 'use_lvq', 'metric', 'Avg_F1', 'Std_F1', 'Avg_Time']
    print(df[cols_to_show].head(10).to_string(index=False))

    # Guardar
    df.to_csv('proto_comprehensive_search.csv', index=False)
    print("\n✓ Guardado: proto_comprehensive_search.csv")

    return df


# =============================================================================
# VISUALIZACIONES
# =============================================================================

def plot_top5_comparison(df_ann, df_proto, datasets):
    """Gráficos del Top-5"""
    print("\n" + "="*70)
    print("Generando gráficos...")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Top-5 ANN: F1 por dataset
    ax = axes[0, 0]
    top5_ann = df_ann.head(5)
    ds_names = list(datasets.keys())
    f1_cols = [f'{ds}_f1' for ds in ds_names if f'{ds}_f1' in df_ann.columns]

    x = np.arange(len(f1_cols))
    width = 0.15

    for i, idx in enumerate(top5_ann.index[:5]):
        row = top5_ann.loc[idx]
        f1_values = [row[col] for col in f1_cols]
        label = f"Config {i+1}"
        ax.bar(x + i*width, f1_values, width, label=label, alpha=0.8)

    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('F1-Score', fontsize=10)
    ax.set_title('Top-5 ANN: F1 por Dataset', fontweight='bold', fontsize=11)
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(ds_names, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    # 2. Top-5 Proto: F1 por dataset
    ax = axes[0, 1]
    top5_proto = df_proto.head(5)

    for i, idx in enumerate(top5_proto.index[:5]):
        row = top5_proto.loc[idx]
        f1_values = [row[col] for col in f1_cols]
        label = f"Config {i+1}"
        ax.bar(x + i*width, f1_values, width, label=label, alpha=0.8)

    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('F1-Score', fontsize=10)
    ax.set_title('Top-5 Proto: F1 por Dataset', fontweight='bold', fontsize=11)
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(ds_names, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    # 3. Trade-off: F1 vs Time (ANN)
    ax = axes[1, 0]
    ax.scatter(df_ann['Avg_Time'], df_ann['Avg_F1'], alpha=0.5, s=30, c='blue', label='All configs')
    top10_ann = df_ann.head(10)
    ax.scatter(top10_ann['Avg_Time'], top10_ann['Avg_F1'], 
               s=100, c='red', marker='*', label='Top-10', zorder=5)
    ax.set_xlabel('Avg Time (s)', fontsize=10)
    ax.set_ylabel('Avg F1-Score', fontsize=10)
    ax.set_title('ANN: Trade-off F1 vs Tiempo', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Trade-off: F1 vs Time (Proto)
    ax = axes[1, 1]
    ax.scatter(df_proto['Avg_Time'], df_proto['Avg_F1'], alpha=0.5, s=30, c='green', label='All configs')
    top10_proto = df_proto.head(10)
    ax.scatter(top10_proto['Avg_Time'], top10_proto['Avg_F1'],
               s=100, c='red', marker='*', label='Top-10', zorder=5)
    ax.set_xlabel('Avg Time (s)', fontsize=10)
    ax.set_ylabel('Avg F1-Score', fontsize=10)
    ax.set_title('Proto: Trade-off F1 vs Tiempo', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hyperparam_search_comprehensive.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico: hyperparam_search_comprehensive.png")

    # Gráfico adicional: Heatmap de hiperparámetros
    plot_heatmaps(df_ann, df_proto)


def plot_heatmaps(df_ann, df_proto):
    """Heatmaps de impacto de hiperparámetros"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ANN: n_neighbors vs metric
    ax = axes[0]
    pivot = df_ann.groupby(['n_neighbors', 'metric'])['Avg_F1'].mean().unstack()
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': 'Avg F1'}, vmin=0.5, vmax=1.0)
    ax.set_title('ANN: n_neighbors vs metric', fontweight='bold')
    ax.set_xlabel('Metric')
    ax.set_ylabel('n_neighbors')

    # Proto: k_protos vs sigma
    ax = axes[1]
    pivot = df_proto.groupby(['k_protos', 'sigma'])['Avg_F1'].mean().unstack()
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': 'Avg F1'}, vmin=0.5, vmax=1.0)
    ax.set_title('Proto: k_protos vs sigma', fontweight='bold')
    ax.set_xlabel('Sigma')
    ax.set_ylabel('k_protos')

    plt.tight_layout()
    plt.savefig('hyperparam_heatmaps.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico: hyperparam_heatmaps.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE HYPERPARAMETER SEARCH")
    print("="*70)
    print("\nBenchmark exhaustivo para encontrar mejores defaults.\n")

    t0 = time.time()

    # Cargar datasets
    datasets = load_all_datasets()

    # Grid search ANN
    df_ann = grid_search_ann_comprehensive(datasets)

    # Grid search Proto
    df_proto = grid_search_proto_comprehensive(datasets)

    # Visualizaciones
    plot_top5_comparison(df_ann, df_proto, datasets)

    t_total = time.time() - t0

    # Resumen final
    print("\n\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)

    print("\n🏆 MEJOR CONFIGURACIÓN ANN:")
    best_ann = df_ann.iloc[0]
    print(f"  n_neighbors: {best_ann['n_neighbors']}")
    print(f"  metric: {best_ann['metric']}")
    print(f"  iter_ratio: {best_ann['iter_ratio']}")
    print(f"  M: {best_ann['M']}")
    print(f"  ef_construction: {best_ann['ef_construction']}")
    print(f"  → Avg F1: {best_ann['Avg_F1']:.4f} ± {best_ann['Std_F1']:.4f}")
    print(f"  → Avg Time: {best_ann['Avg_Time']:.2f}s")

    print("\n🏆 MEJOR CONFIGURACIÓN PROTO:")
    best_proto = df_proto.iloc[0]
    print(f"  k_protos: {best_proto['k_protos']}")
    print(f"  sigma: {best_proto['sigma']}")
    print(f"  use_lvq: {best_proto['use_lvq']}")
    print(f"  metric: {best_proto['metric']}")
    print(f"  → Avg F1: {best_proto['Avg_F1']:.4f} ± {best_proto['Std_F1']:.4f}")
    print(f"  → Avg Time: {best_proto['Avg_Time']:.2f}s")

    print(f"\n⏱️  Tiempo total: {t_total/60:.1f} minutos")
    print(f"📊 Datasets evaluados: {len(datasets)}")
    print(f"🔬 Configuraciones ANN: {len(df_ann)}")
    print(f"🔬 Configuraciones Proto: {len(df_proto)}")

    print("\n✅ COMPLETADO")
    print("\nArchivos generados:")
    print("  • ann_comprehensive_search.csv")
    print("  • proto_comprehensive_search.csv")
    print("  • hyperparam_search_comprehensive.png")
    print("  • hyperparam_heatmaps.png")
