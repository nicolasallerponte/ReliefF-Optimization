# corral_100_scaling_ann_proto.py
# Escalado de ANN y Proto en CorrAL-100 grande (hasta millones de muestras)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Setup imports (ajusta la ruta según tu repo)
# ---------------------------------------------------------------------
def setup():
    current = Path(__file__).parent.resolve()
    root = current
    while root != root.parent:
        if (root / "src").exists():
            break
        root = root.parent
    sys.path.insert(0, str(root / "src"))

setup()

from relieff_improved.ann.ann_v2_optimized import ANN
from relieff_improved.proto.proto_v2_optimized import Proto


# ---------------------------------------------------------------------
# Generación de CorrAL-100 grande
# ---------------------------------------------------------------------
def generate_large_corral(n_samples: int, n_features: int = 100, seed: int = 42):
    """
    Genera un CorrAL-100 "grande" con n_samples filas y n_features columnas.
    
    Estructura:
      - f0..f3: relevantes
      - f4: irrelevante
      - f5: correlacionada con Y al 75%
      - f6..f99: irrelevantes
    Regla:
      Y = (f0 AND f1) OR (f2 AND f3)
    """
    rng = np.random.RandomState(seed)

    # f0..f3 relevantes (Bernoulli(0.5))
    F = rng.binomial(1, 0.5, size=(n_samples, 4)).astype(np.int32)
    f0, f1, f2, f3 = [F[:, i] for i in range(4)]

    # Target
    y = ((f0 & f1) | (f2 & f3)).astype(np.int32)

    # f4 irrelevante
    f4 = rng.binomial(1, 0.5, size=n_samples).astype(np.int32)

    # f5 correlacionada al 75% con Y
    f5 = y.copy()
    n_flip = int(0.25 * n_samples)
    if n_flip > 0:
        flip_idx = rng.choice(n_samples, n_flip, replace=False)
        f5[flip_idx] = 1 - f5[flip_idx]

    # resto de irrelevantes
    n_noise = n_features - 6
    noise = rng.binomial(1, 0.5, size=(n_samples, n_noise)).astype(np.int32)

    X = np.column_stack([f0, f1, f2, f3, f4, f5, noise]).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------
# Evaluación de un selector (ANN o Proto)
# ---------------------------------------------------------------------
def evaluate_selector(selector_cls, selector_kwargs, X, y):
    """
    Ejecuta fit + rank de un selector de características y mide tiempo.
    Devuelve tiempos y posiciones de f0..f5.
    """
    t0 = time.perf_counter()
    selector = selector_cls(**selector_kwargs)
    selector.fit(X, y)
    ranking = selector.rank()
    t1 = time.perf_counter()
    elapsed = t1 - t0

    # Posición (1-based) de las 6 primeras features
    positions = {i: int(np.where(ranking == i)[0][0] + 1) for i in range(6)}

    # Número de relevantes (f0..f3) en Top 10
    rel_top10 = sum(1 for i in [0, 1, 2, 3] if positions[i] <= 10)

    # Posición media de relevantes
    avg_pos_rel = float(np.mean([positions[i] for i in [0, 1, 2, 3]]))

    return {
        "time": elapsed,
        "positions": positions,
        "rel_top10": rel_top10,
        "avg_pos_rel": avg_pos_rel,
    }


# ---------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------
def run_scaling_experiment(
    sample_sizes=None,
    n_features=100,
    base_seed=42,
    output_csv="corral100_scaling_ann_proto.csv",
    output_dir="results",
):
    if sample_sizes is None:
        # Ajusta según la RAM que tengas; 5M puede ser muy pesado
        sample_sizes = [
            2_000,
            10_000,
            50_000,
            100_000,
            200_000,
            500_000,
            1_000_000,
            # 2_000_000,
            # 5_000_000,
        ]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []

    for n in sample_sizes:
        print("=" * 80)
        print(f"Evaluando con n_samples = {n}")
        print("=" * 80)

        # Generar dataset
        X, y = generate_large_corral(n_samples=n, n_features=n_features, seed=base_seed)
        print(f"  Dataset: X.shape={X.shape}, y.shape={y.shape}")

        # ANN
        print("  -> ANN...")
        ann_metrics = evaluate_selector(
            ANN,
            dict(
                n_neighbors=10,
                n_features_to_select=10,
                discrete_threshold=10,
                random_state=42,
            ),
            X,
            y,
        )
        ann_row = {
            "method": "ANN",
            "n_samples": n,
            "n_features": n_features,
            "time": ann_metrics["time"],
            "rel_top10": ann_metrics["rel_top10"],
            "avg_pos_rel": ann_metrics["avg_pos_rel"],
            "pos_f0": ann_metrics["positions"][0],
            "pos_f1": ann_metrics["positions"][1],
            "pos_f2": ann_metrics["positions"][2],
            "pos_f3": ann_metrics["positions"][3],
            "pos_f4": ann_metrics["positions"][4],  # irrelevante
            "pos_f5": ann_metrics["positions"][5],  # correlacionada
        }
        results.append(ann_row)
        print(
            f"    ANN: tiempo={ann_row['time']:.2f}s, "
            f"relevantes_top10={ann_row['rel_top10']}/4, "
            f"pos_f0..f3={[ann_row['pos_f0'], ann_row['pos_f1'], ann_row['pos_f2'], ann_row['pos_f3']]}, "
            f"pos_f4(irr)={ann_row['pos_f4']}, pos_f5(corr)={ann_row['pos_f5']}"
        )

        # Proto
        print("  -> Proto...")
        proto_metrics = evaluate_selector(
            Proto,
            dict(
                n_features_to_select=10,
                sigma=0.15,
                discrete_threshold=10,
                k_protos=5,
                use_lvq=False,
                n_jobs=1,
            ),
            X,
            y,
        )
        proto_row = {
            "method": "Proto",
            "n_samples": n,
            "n_features": n_features,
            "time": proto_metrics["time"],
            "rel_top10": proto_metrics["rel_top10"],
            "avg_pos_rel": proto_metrics["avg_pos_rel"],
            "pos_f0": proto_metrics["positions"][0],
            "pos_f1": proto_metrics["positions"][1],
            "pos_f2": proto_metrics["positions"][2],
            "pos_f3": proto_metrics["positions"][3],
            "pos_f4": proto_metrics["positions"][4],
            "pos_f5": proto_metrics["positions"][5],
        }
        results.append(proto_row)
        print(
            f"    Proto: tiempo={proto_row['time']:.2f}s, "
            f"relevantes_top10={proto_row['rel_top10']}/4, "
            f"pos_f0..f3={[proto_row['pos_f0'], proto_row['pos_f1'], proto_row['pos_f2'], proto_row['pos_f3']]}, "
            f"pos_f4(irr)={proto_row['pos_f4']}, pos_f5(corr)={proto_row['pos_f5']}"
        )

    # Guardar resultados
    df = pd.DataFrame(results)
    csv_path = Path(output_dir) / output_csv
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 80)
    print(f"Resultados guardados en: {csv_path}")
    print("=" * 80)

    # Plots
    plot_scaling_results(df, output_dir=output_dir)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_scaling_results(df: pd.DataFrame, output_dir="results"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    methods = ["ANN", "Proto"]

    # Tiempo vs nº muestras (log-log)
    plt.figure(figsize=(7, 5))
    for m in methods:
        sub = df[df["method"] == m].sort_values("n_samples")
        plt.plot(sub["n_samples"], sub["time"], marker="o", label=m)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Número de muestras (log)")
    plt.ylabel("Tiempo (s, log)")
    plt.title("Tiempo de ejecución vs nº de muestras (CorrAL-100)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "corral100_time_scaling.png", dpi=300)
    plt.close()

    # Nº de relevantes en Top 10 vs nº muestras
    plt.figure(figsize=(7, 5))
    for m in methods:
        sub = df[df["method"] == m].sort_values("n_samples")
        plt.plot(sub["n_samples"], sub["rel_top10"], marker="o", label=m)
    plt.xscale("log")
    plt.xlabel("Número de muestras (log)")
    plt.ylabel("Relevantes en Top 10 (de 4)")
    plt.ylim(-0.1, 4.1)
    plt.title("Relevantes en Top 10 vs nº de muestras")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "corral100_rel_top10_scaling.png", dpi=300)
    plt.close()

    # Posición promedio de relevantes vs nº muestras
    plt.figure(figsize=(7, 5))
    for m in methods:
        sub = df[df["method"] == m].sort_values("n_samples")
        plt.plot(sub["n_samples"], sub["avg_pos_rel"], marker="o", label=m)
    plt.xscale("log")
    plt.xlabel("Número de muestras (log)")
    plt.ylabel("Posición promedio f0..f3")
    plt.title("Posición promedio de relevantes vs nº de muestras")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "corral100_avg_pos_rel_scaling.png", dpi=300)
    plt.close()

    # Posición de f4 (irrelevante) y f5 (correlacionada)
    plt.figure(figsize=(7, 5))
    for m, color, ls in [("ANN", "tab:blue", "-"), ("Proto", "tab:orange", "--")]:
        sub = df[df["method"] == m].sort_values("n_samples")
        plt.plot(
            sub["n_samples"],
            sub["pos_f4"],
            marker="o",
            color=color,
            linestyle=ls,
            label=f"{m} f4 irrelevante",
        )
        plt.plot(
            sub["n_samples"],
            sub["pos_f5"],
            marker="s",
            color=color,
            linestyle=":",
            label=f"{m} f5 correlacionada",
        )
    plt.xscale("log")
    plt.xlabel("Número de muestras (log)")
    plt.ylabel("Posición en ranking")
    plt.title("Posición de f4 (irrelevante) y f5 (correlacionada)")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "corral100_f4_f5_positions.png", dpi=300)
    plt.close()

    print(f"Plots guardados en: {output_dir}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_scaling_experiment()
