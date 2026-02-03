# corral_100_noise_robustness_top5_full.py

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


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

try:
    from skrebate import ReliefF as ReliefF_Original
    relieff_available = True
except ImportError:
    relieff_available = False


def generate_corrAL100(seed=42, n_replicas=50):
    """
    CorrAL-100 con réplicas:
      f0..f3: relevantes
      f4: irrelevante
      f5: correlacionada 75%
    """
    rng = np.random.RandomState(seed)
    n_base = 32

    # 32 combinaciones base para f0..f3
    f0_base = np.array([(i >> 0) & 1 for i in range(n_base)], dtype=np.int32)
    f1_base = np.array([(i >> 1) & 1 for i in range(n_base)], dtype=np.int32)
    f2_base = np.array([(i >> 2) & 1 for i in range(n_base)], dtype=np.int32)
    f3_base = np.array([(i >> 3) & 1 for i in range(n_base)], dtype=np.int32)

    # Target base
    y_base = ((f0_base & f1_base) | (f2_base & f3_base)).astype(np.int32)

    # Réplicas
    f0 = np.tile(f0_base, n_replicas)
    f1 = np.tile(f1_base, n_replicas)
    f2 = np.tile(f2_base, n_replicas)
    f3 = np.tile(f3_base, n_replicas)
    y = np.tile(y_base, n_replicas)
    n_samples = len(y)

    # f4 irrelevante
    f4 = rng.binomial(1, 0.5, size=n_samples).astype(np.int32)

    # f5 correlacionada 75%
    f5 = y.copy()
    n_flip = int(0.25 * n_samples)
    if n_flip > 0:
        flip_idx = rng.choice(n_samples, n_flip, replace=False)
        f5[flip_idx] = 1 - f5[flip_idx]

    # 93 irrelevantes extra
    noise = rng.binomial(1, 0.5, size=(n_samples, 93)).astype(np.int32)

    X = np.column_stack([f0, f1, f2, f3, f4, f5, noise]).astype(np.float32)
    return X, y


def add_gaussian_noise(X, std, seed):
    if std == 0:
        return X.copy()
    rng = np.random.RandomState(seed)
    X_noisy = X.copy()
    X_noisy[:, :4] += rng.normal(0, std, size=(X.shape[0], 4))
    return X_noisy


def add_label_noise(y, prob, seed):
    if prob == 0:
        return y.copy()
    rng = np.random.RandomState(seed)
    y_noisy = y.copy()
    n_flip = int(prob * len(y))
    if n_flip > 0:
        flip_idx = rng.choice(len(y), n_flip, replace=False)
        y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return y_noisy


def add_feature_noise(X, prob, seed):
    if prob == 0:
        return X.copy()
    rng = np.random.RandomState(seed)
    X_noisy = X.copy()
    for i in range(4):  # solo relevantes
        mask = rng.random(X.shape[0]) < prob
        X_noisy[mask, i] = 1 - X_noisy[mask, i]
    return X_noisy


def add_missing_values(X, prob, seed):
    if prob == 0:
        return X.copy()
    rng = np.random.RandomState(seed)
    X_missing = X.copy()
    for i in range(4):  # solo relevantes
        mask = rng.random(X.shape[0]) < prob
        X_missing[mask, i] = np.nan
        if np.any(np.isnan(X_missing[:, i])):
            X_missing[np.isnan(X_missing[:, i]), i] = np.nanmean(X_missing[:, i])
    return X_missing


def evaluate_method(method_name, X, y):
    """
    Métricas clave:
      - n_top5_rel_corr: nº de {f0,f1,f2,f3,f5} que están en Top5 (0..5)
      - perfect_top5_rel_corr: 1 si las cinco (4 relevantes + correlacionada) están en Top5
      - además se guarda Top10 y avg_pos por si quieres comparar.
    """
    try:
        if method_name == 'ANN':
            model = ANN(
                n_neighbors=10,
                n_features_to_select=10,
                discrete_threshold=10,
                random_state=42
            )
        elif method_name == 'Proto':
            model = Proto(
                n_features_to_select=10,
                sigma=0.15,
                discrete_threshold=10,
                k_protos=5,
                use_lvq=False,
                n_jobs=1
            )
        elif method_name == 'ReliefF' and relieff_available:
            model = ReliefF_Original(n_features_to_select=10, n_neighbors=10)
        else:
            return None

        model.fit(X, y)
        ranking = model.rank() if hasattr(model, 'rank') else np.argsort(-model.feature_importances_)

        # Posiciones de f0..f5 (1-based)
        positions = {i: int(np.where(ranking == i)[0][0] + 1) for i in range(6)}

        # Relevantes en Top10 (como antes)
        rel_top10 = sum(1 for i in [0, 1, 2, 3] if positions[i] <= 10)

        # Conjunto objetivo para Top5: 4 relevantes + correlacionada
        target_feats = [0, 1, 2, 3, 5]

        # nº de {f0,f1,f2,f3,f5} que están en Top5
        n_top5_rel_corr = sum(1 for i in target_feats if positions[i] <= 5)

        # ¿están las cinco en Top5?
        perfect_top5_rel_corr = int(all(positions[i] <= 5 for i in target_feats))

        # Posición media de las 4 relevantes
        avg_pos_rel = float(np.mean([positions[i] for i in [0, 1, 2, 3]]))

        return {
            'rel_top10': rel_top10,
            'n_top5_rel_corr': n_top5_rel_corr,
            'perfect_top5_rel_corr': perfect_top5_rel_corr,
            'avg_pos_rel': avg_pos_rel,
            'score_top10': rel_top10 / 4.0,          # proporción de relevantes en Top10
            'score_top5_rel_corr': n_top5_rel_corr / 5.0,  # proporción (de 5) en Top5
            'pos_corr': positions[5],
            'positions': positions
        }
    except Exception as e:
        print(f"    ✗ {method_name}: {e}")
        return None


def run_experiment(noise_type, noise_levels, n_replicas=50):
    print(f"\n{'='*80}")
    print(f"EXPERIMENTO: {noise_type.upper()} NOISE (Top5 relevantes+correlacionada)")
    print(f"{'='*80}\n")

    methods = ['ANN', 'Proto']
    if relieff_available:
        methods.append('ReliefF')

    X_base, y_base = generate_corrAL100(seed=42, n_replicas=n_replicas)

    print(f"Dataset: {X_base.shape}, clases: {np.bincount(y_base)}")
    print(f"Correlación f5: {np.mean(X_base[:, 5] == y_base):.1%}")
    print(f"(32 combinaciones × {n_replicas} réplicas = {len(y_base)} muestras)\n")

    results = []

    for i, level in enumerate(noise_levels):
        print(f"  Ruido {level:.3f}:")
        seed = 1000 + i

        if noise_type == 'gaussian':
            X, y = add_gaussian_noise(X_base, level, seed), y_base.copy()
        elif noise_type == 'label':
            X, y = X_base.copy(), add_label_noise(y_base, level, seed)
        elif noise_type == 'feature':
            X, y = add_feature_noise(X_base, level, seed), y_base.copy()
        elif noise_type == 'missing':
            X, y = add_missing_values(X_base, level, seed), y_base.copy()
        else:
            continue

        for method in methods:
            metrics = evaluate_method(method, X, y)
            if metrics:
                row = {
                    'noise_type': noise_type,
                    'noise_level': level,
                    'method': method,
                    **metrics
                }
                results.append(row)

                if abs(level) < 1e-6:
                    pos_str = ", ".join([f"f{i}:{metrics['positions'][i]}"
                                         for i in [0, 1, 2, 3, 5]])
                    print(
                        f"    {method}: Top5_rel+corr={metrics['n_top5_rel_corr']}/5, "
                        f"Top10_rel={metrics['rel_top10']}/4 | {pos_str}"
                    )
                else:
                    print(
                        f"    {method}: Top5_rel+corr={metrics['n_top5_rel_corr']}/5 "
                        f"(score={metrics['score_top5_rel_corr']:.2f}), "
                        f"Top10_rel={metrics['rel_top10']}/4 "
                        f"(score={metrics['score_top10']:.2f})"
                    )

    return pd.DataFrame(results)


def plot_results(df, output_dir='./results'):
    import os
    os.makedirs(output_dir, exist_ok=True)

    for noise_type in df['noise_type'].unique():
        df_sub = df[df['noise_type'] == noise_type]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f'Robustez ante {noise_type.upper()} Noise - CorrAL-100\n'
            f'(Top5: 4 relevantes + correlacionada)',
            fontsize=14, fontweight='bold'
        )

        # Izquierda: proporción de {4 rel + corr} en Top5
        ax = axes[0]
        for method in df_sub['method'].unique():
            data = df_sub[df_sub['method'] == method]
            ax.plot(
                data['noise_level'],
                data['score_top5_rel_corr'],
                marker='o',
                label=method,
                linewidth=2,
                markersize=6
            )
        ax.set_xlabel('Nivel de ruido', fontsize=11)
        ax.set_ylabel('Score Top5 (0-1)', fontsize=11)
        ax.set_title('Proporción (4 relevantes + correlacionada) en Top5', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1)

        # Derecha: posición media de las 4 relevantes
        ax = axes[1]
        for method in df_sub['method'].unique():
            data = df_sub[df_sub['method'] == method]
            ax.plot(
                data['noise_level'],
                data['avg_pos_rel'],
                marker='s',
                label=method,
                linewidth=2,
                markersize=6
            )
        ax.set_xlabel('Nivel de ruido', fontsize=11)
        ax.set_ylabel('Posición promedio f0..f3', fontsize=11)
        ax.set_title('Posición Promedio de Features Relevantes', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        ax.axhline(y=5.5, color='green', linestyle='--', alpha=0.3, linewidth=1)

        plt.tight_layout()
        fname = f'{output_dir}/corral100_{noise_type}_top5_rel_corr.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  ✓ Guardado: {fname}")
        plt.close()


def main():
    print("="*80)
    print("EVALUACIÓN DE ROBUSTEZ - CorrAL-100 (Top5: relevantes+correlacionada)")
    print("="*80)

    N_REPLICAS = 150

    configs = {
        'gaussian': np.linspace(0, 0.25, 10),
        'label':    np.linspace(0, 0.25, 10),
        'feature':  np.linspace(0, 0.25, 10),
        'missing':  np.linspace(0, 0.25, 10),
    }

    all_results = []
    for noise_type, levels in configs.items():
        df = run_experiment(noise_type, levels, n_replicas=N_REPLICAS)
        all_results.append(df)

    df_all = pd.concat(all_results, ignore_index=True)

    import os
    os.makedirs('./results', exist_ok=True)
    csv_path = './results/corral100_noise_top5_rel_corr.csv'
    df_all.to_csv(csv_path, index=False)
    print(f"\n✓ Resultados en: {csv_path}")

    print("\nGenerando gráficos...")
    plot_results(df_all)

    print("\n" + "="*80)
    print("RESUMEN (Top5: 4 relevantes + correlacionada)")
    print("="*80)
    for noise_type in configs.keys():
        print(f"\n{noise_type.upper()}:")
        df_sub = df_all[df_all['noise_type'] == noise_type]
        for method in df_sub['method'].unique():
            data = df_sub[df_sub['method'] == method]
            s0 = data[data['noise_level'] == 0]['score_top5_rel_corr'].values[0]
            smax = data[data['noise_level'] == data['noise_level'].max()]['score_top5_rel_corr'].values[0]
            deg = s0 - smax
            pct = (deg / s0 * 100) if s0 > 0 else 0
            print(
                f"  {method:8s}: sin ruido(Top5 rel+corr)={s0:.2f}, "
                f"max ruido={smax:.2f}, degradación={deg:.2f} ({pct:.0f}%)"
            )

    print("\n" + "="*80)
    print("✅ Completado. Revisa ./results/")
    print("="*80)


if __name__ == "__main__":
    main()
