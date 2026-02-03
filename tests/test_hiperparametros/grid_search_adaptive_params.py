"""
Grid Search para Parámetros Óptimos - FIXED VERSION
===================================================

Implementación inline de Proto/ANN con configuración fija de parámetros.
No depende de imports externos.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.cluster import MiniBatchKMeans
from itertools import product
import time

# ============================================================================
# DATASET CORRAL
# ============================================================================

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

# ============================================================================
# PROTO SIMPLIFICADO (INLINE, CONFIGURACIÓN FIJA)
# ============================================================================

class ProtoSimple:
    """Proto simplificado con sigma configurable"""

    def __init__(self, sigma=0.10, k_protos=10):
        self.sigma = sigma
        self.k_protos = k_protos
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int32)
        n_samples, n_features = X.shape
        classes = np.unique(y)

        # Normalización simple
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Generar prototipos por clustering
        prototypes = {}
        for c in classes:
            X_c = X[y == c]
            count = len(X_c)

            n_clusters = int(max(1, min(200, count * self.sigma)))

            if count <= n_clusters:
                prototypes[c] = X_c
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=min(1024, len(X_c)),
                    n_init=3,
                    random_state=42
                ).fit(X_c)
                prototypes[c] = kmeans.cluster_centers_.astype(np.float32)

        # Scoring simple (sin paralelización para debugging)
        p_classes = {c: np.sum(y == c) / n_samples for c in classes}
        scores = np.zeros(n_features, dtype=np.float32)

        for c in classes:
            mask = (y == c)
            X_c = X[mask]
            my_protos = prototypes[c]

            if len(X_c) == 0 or len(my_protos) == 0:
                continue

            # Hit phase
            dists = pairwise_distances(X_c, my_protos, metric='euclidean')
            k_eff = min(self.k_protos, len(my_protos))
            nearest_idx = np.argpartition(dists, k_eff - 1, axis=1)[:, :k_eff]

            for i, row in enumerate(X_c):
                chosen = my_protos[nearest_idx[i]]
                diffs = np.abs(row - chosen)
                scores -= np.mean(diffs, axis=0) * p_classes[c] / len(X_c)

            # Miss phase
            prob_not_c = 1.0 - p_classes[c]
            if prob_not_c == 0:
                prob_not_c = 1.0

            for other_c in classes:
                if other_c == c:
                    continue

                other_protos = prototypes[other_c]
                if len(other_protos) == 0:
                    continue

                dists_m = pairwise_distances(X_c, other_protos, metric='euclidean')
                k_eff_m = min(self.k_protos, len(other_protos))
                nearest_idx_m = np.argpartition(dists_m, k_eff_m - 1, axis=1)[:, :k_eff_m]

                for i, row in enumerate(X_c):
                    chosen_m = other_protos[nearest_idx_m[i]]
                    diffs = np.abs(row - chosen_m)
                    weight = p_classes[other_c] / prob_not_c
                    scores += np.mean(diffs, axis=0) * weight * p_classes[c] / len(X_c)

        self.feature_importances_ = scores
        return self

    def rank(self):
        return np.argsort(-self.feature_importances_)

    def transform(self, X):
        idx = self.rank()[:10]
        return X[:, idx]


# ============================================================================
# ANN SIMPLIFICADO (INLINE, CONFIGURACIÓN FIJA)
# ============================================================================

class ANNSimple:
    """ANN simplificado con k configurable"""

    def __init__(self, k=10):
        self.k = k
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        n_samples, n_features = X.shape

        # Normalización
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Calcular distancias (O(N^2), solo para datasets pequeños)
        distance = pairwise_distances(X, metric='euclidean')
        np.fill_diagonal(distance, np.inf)

        # Scoring
        classes = np.unique(y)
        p_c = np.bincount(y) / n_samples
        scores = np.zeros(n_features, dtype=np.float32)

        for idx in range(n_samples):
            # Encontrar k vecinos más cercanos de cada clase
            sorted_indices = np.argsort(distance[idx])

            hits = []
            misses = {c: [] for c in classes}

            for i in sorted_indices:
                if y[i] == y[idx] and len(hits) < self.k:
                    hits.append(i)
                elif y[i] != y[idx] and len(misses[y[i]]) < self.k:
                    misses[y[i]].append(i)

                if len(hits) == self.k and all(len(v) >= self.k for v in misses.values()):
                    break

            # Hit diff
            if hits:
                hit_diff = np.mean([np.abs(X[idx] - X[i]) for i in hits], axis=0)
                scores -= hit_diff

            # Miss diff
            prob_target = p_c[y[idx]]
            denom = 1.0 - prob_target
            if denom == 0:
                denom = 1.0

            for c in classes:
                if c == y[idx] or len(misses[c]) == 0:
                    continue
                weight = p_c[c] / denom
                miss_diff = np.mean([np.abs(X[idx] - X[i]) for i in misses[c]], axis=0)
                scores += miss_diff * weight

        self.feature_importances_ = scores / n_samples
        return self

    def rank(self):
        return np.argsort(-self.feature_importances_)

    def transform(self, X):
        idx = self.rank()[:10]
        return X[:, idx]


# ============================================================================
# WRAPPERS CON FUNCIÓN CONTINUA
# ============================================================================

class ProtoGridSearch:
    """Proto con función continua ajustable"""
    def __init__(self, base_sigma=0.08, baseline_ratio=0.05):
        self.base_sigma = base_sigma
        self.baseline_ratio = baseline_ratio

    def _compute_sigma(self, ratio, is_synthetic):
        """Función continua"""
        # Override: sintético pequeño
        if is_synthetic and ratio < 0.05:
            return 0.02

        if ratio <= self.baseline_ratio:
            return self.baseline_ratio

        log_scale = np.log(ratio / self.baseline_ratio)
        sigma = self.base_sigma * (1 + log_scale)
        return np.clip(sigma, 0.02, 0.25)

    def fit(self, X, y):
        n_features = X.shape[1]
        n_samples = X.shape[0]
        ratio = n_features / n_samples
        is_synthetic = n_features < 50 and n_samples < 2000

        computed_sigma = self._compute_sigma(ratio, is_synthetic)

        self.proto = ProtoSimple(sigma=computed_sigma, k_protos=10)
        self.proto.fit(X, y)
        return self

    def rank(self):
        return self.proto.rank()

    def transform(self, X):
        return self.proto.transform(X)


class ANNGridSearch:
    """ANN con función continua ajustable"""
    def __init__(self, base_k=15, scale_factor=7.0):
        self.base_k = base_k
        self.scale_factor = scale_factor

    def _compute_k(self, ratio, is_synthetic):
        """Función continua"""
        # Override: sintético pequeño
        if is_synthetic and ratio < 0.05:
            return 7

        k = self.base_k / (1 + ratio * self.scale_factor)
        return int(np.clip(k, 5, 15))

    def fit(self, X, y):
        n_features = X.shape[1]
        n_samples = X.shape[0]
        ratio = n_features / n_samples
        is_synthetic = n_features < 50 and n_samples < 2000

        computed_k = self._compute_k(ratio, is_synthetic)

        self.ann = ANNSimple(k=computed_k)
        self.ann.fit(X, y)
        return self

    def rank(self):
        return self.ann.rank()

    def transform(self, X):
        return self.ann.transform(X)


# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluate_f1(selector, Xtr, Xte, ytr, yte):
    """Evalúa F1 (ranking top-6)"""
    try:
        selector.fit(Xtr, ytr)
        rank = selector.rank()
        top6 = set(rank[:6])
        true = {0, 1, 2, 3, 4, 5}
        f1 = len(top6 & true) / 6
        return f1
    except Exception as e:
        print(f"\nError en evaluate_f1: {e}")
        return 0.0


# ============================================================================
# GRID SEARCH
# ============================================================================

def run_grid_search():
    """Ejecuta búsqueda exhaustiva"""

    print("="*70)
    print("GRID SEARCH: Parámetros Óptimos (FIXED)")
    print("="*70)

    # Preparar datos base
    X, y = make_corral(800, noise=0.0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # Grid de parámetros (reducido para testing rápido)
    proto_base_sigmas = [0.05, 0.08, 0.10, 0.12]
    proto_baseline_ratios = [0.04, 0.05]

    ann_base_ks = [12, 15]
    ann_scale_factors = [5.0, 7.0, 10.0]

    # Configuraciones de dilución
    dilution_configs = [
        (0, "baseline"),
        (10, "+10"),
        (20, "+20"),
        (50, "+50"),
        (100, "+100"),
        (200, "+200")
    ]

    # ========================================================================
    # PROTO GRID SEARCH
    # ========================================================================

    print("\n" + "-"*70)
    print("PROTO: Buscando base_sigma y baseline_ratio óptimos")
    print("-"*70)

    proto_results = []
    total_proto = len(proto_base_sigmas) * len(proto_baseline_ratios)
    current = 0

    for base_sigma, baseline_ratio in product(proto_base_sigmas, proto_baseline_ratios):
        current += 1
        print(f"[{current}/{total_proto}] Testing base_sigma={base_sigma:.2f}, baseline_ratio={baseline_ratio:.2f}")

        f1_scores = []

        for n_rand, name in dilution_configs:
            # Preparar datos diluidos
            if n_rand == 0:
                Xtr_test = Xtr.copy()
                Xte_test = Xte.copy()
            else:
                np.random.seed(42 + n_rand)
                rand_tr = np.random.randn(Xtr.shape[0], n_rand)
                rand_te = np.random.randn(Xte.shape[0], n_rand)
                Xtr_test = np.hstack([Xtr, rand_tr])
                Xte_test = np.hstack([Xte, rand_te])

            # Evaluar
            selector = ProtoGridSearch(base_sigma=base_sigma, baseline_ratio=baseline_ratio)
            f1 = evaluate_f1(selector, Xtr_test, Xte_test, ytr, yte)
            f1_scores.append(f1)
            print(f"  {name:10s}: F1={f1:.2f}")

        # F1 promedio ponderado
        weights = [2.0, 1.0, 1.5, 1.5, 1.0, 0.5]
        weighted_f1 = np.average(f1_scores, weights=weights)

        proto_results.append({
            'base_sigma': base_sigma,
            'baseline_ratio': baseline_ratio,
            'f1_baseline': f1_scores[0],
            'f1_+10': f1_scores[1],
            'f1_+20': f1_scores[2],
            'f1_+50': f1_scores[3],
            'f1_+100': f1_scores[4],
            'f1_+200': f1_scores[5],
            'f1_avg_weighted': weighted_f1
        })
        print(f"  Weighted Avg: {weighted_f1:.3f}\n")

    df_proto = pd.DataFrame(proto_results)
    df_proto = df_proto.sort_values('f1_avg_weighted', ascending=False)

    print("\n🏆 TOP 3 PROTO CONFIGS:")
    print(df_proto.head(3).to_string(index=False))

    best_proto = df_proto.iloc[0]
    print(f"\n✅ BEST PROTO: base_sigma={best_proto['base_sigma']:.2f}, baseline_ratio={best_proto['baseline_ratio']:.2f}")
    print(f"   F1 Weighted Avg: {best_proto['f1_avg_weighted']:.3f}")

    # ========================================================================
    # ANN GRID SEARCH
    # ========================================================================

    print("\n" + "-"*70)
    print("ANN: Buscando base_k y scale_factor óptimos")
    print("-"*70)

    ann_results = []
    total_ann = len(ann_base_ks) * len(ann_scale_factors)
    current = 0

    for base_k, scale_factor in product(ann_base_ks, ann_scale_factors):
        current += 1
        print(f"[{current}/{total_ann}] Testing base_k={base_k}, scale_factor={scale_factor:.1f}")

        f1_scores = []

        for n_rand, name in dilution_configs:
            if n_rand == 0:
                Xtr_test = Xtr.copy()
                Xte_test = Xte.copy()
            else:
                np.random.seed(42 + n_rand)
                rand_tr = np.random.randn(Xtr.shape[0], n_rand)
                rand_te = np.random.randn(Xte.shape[0], n_rand)
                Xtr_test = np.hstack([Xtr, rand_tr])
                Xte_test = np.hstack([Xte, rand_te])

            selector = ANNGridSearch(base_k=base_k, scale_factor=scale_factor)
            f1 = evaluate_f1(selector, Xtr_test, Xte_test, ytr, yte)
            f1_scores.append(f1)
            print(f"  {name:10s}: F1={f1:.2f}")

        weights = [2.0, 1.0, 1.5, 1.5, 1.0, 0.5]
        weighted_f1 = np.average(f1_scores, weights=weights)

        ann_results.append({
            'base_k': base_k,
            'scale_factor': scale_factor,
            'f1_baseline': f1_scores[0],
            'f1_+10': f1_scores[1],
            'f1_+20': f1_scores[2],
            'f1_+50': f1_scores[3],
            'f1_+100': f1_scores[4],
            'f1_+200': f1_scores[5],
            'f1_avg_weighted': weighted_f1
        })
        print(f"  Weighted Avg: {weighted_f1:.3f}\n")

    df_ann = pd.DataFrame(ann_results)
    df_ann = df_ann.sort_values('f1_avg_weighted', ascending=False)

    print("\n🏆 TOP 3 ANN CONFIGS:")
    print(df_ann.head(3).to_string(index=False))

    best_ann = df_ann.iloc[0]
    print(f"\n✅ BEST ANN: base_k={int(best_ann['base_k'])}, scale_factor={best_ann['scale_factor']:.1f}")
    print(f"   F1 Weighted Avg: {best_ann['f1_avg_weighted']:.3f}")

    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================

    df_proto.to_csv('proto_grid_search_fixed.csv', index=False)
    df_ann.to_csv('ann_grid_search_fixed.csv', index=False)

    print("\n" + "="*70)
    print("RESULTADOS GUARDADOS")
    print("="*70)
    print("✓ proto_grid_search_fixed.csv")
    print("✓ ann_grid_search_fixed.csv")

    return best_proto, best_ann


if __name__ == "__main__":
    start = time.time()
    best_proto, best_ann = run_grid_search()
    elapsed = time.time() - start
    print(f"\n⏱️  Tiempo total: {elapsed/60:.1f} minutos")
