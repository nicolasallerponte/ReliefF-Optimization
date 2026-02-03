"""
Test Debug: Verificar por qué Proto falla en baseline
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances

# ============================================================================
# PROTO DEBUG: Versión con prints de debugging
# ============================================================================

class ProtoDebug:
    """Proto con debugging para encontrar bug"""

    def __init__(self, sigma=0.10, k_protos=10):
        self.sigma = sigma
        self.k_protos = k_protos
        self.feature_importances_ = None
        print(f"[ProtoDebug] __init__: sigma={sigma}, k_protos={k_protos}")

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int32)
        n_samples, n_features = X.shape
        classes = np.unique(y)

        print(f"[ProtoDebug] fit: X.shape={X.shape}, n_classes={len(classes)}")
        print(f"[ProtoDebug] sigma={self.sigma}")

        # Normalización
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Generar prototipos
        prototypes = {}
        for c in classes:
            X_c = X[y == c]
            count = len(X_c)

            n_clusters = int(max(1, min(200, count * self.sigma)))
            print(f"[ProtoDebug] Clase {c}: samples={count}, sigma={self.sigma}, n_clusters={n_clusters}")

            if count <= n_clusters:
                prototypes[c] = X_c
                print(f"  → Usando todas las muestras ({count} prototipos)")
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=min(1024, len(X_c)),
                    n_init=3,
                    random_state=42
                ).fit(X_c)
                prototypes[c] = kmeans.cluster_centers_.astype(np.float32)
                print(f"  → Clustering: {n_clusters} prototipos")

        # Scoring
        p_classes = {c: np.sum(y == c) / n_samples for c in classes}
        scores = np.zeros(n_features, dtype=np.float32)

        for c in classes:
            mask = (y == c)
            X_c = X[mask]
            my_protos = prototypes[c]

            if len(X_c) == 0 or len(my_protos) == 0:
                continue

            # Hit phase (simplified)
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

        # Debug: mostrar top features
        rank = np.argsort(-scores)
        print(f"[ProtoDebug] Top-6 features: {rank[:6].tolist()}")
        print(f"[ProtoDebug] Feature importances (top-6): {scores[rank[:6]]}")

        return self

    def rank(self):
        return np.argsort(-self.feature_importances_)

# ============================================================================
# TEST RÁPIDO
# ============================================================================

def make_corral_baseline():
    """Genera Corral baseline (20 features)"""
    np.random.seed(42)
    X = np.random.randn(800, 20).astype(np.float32)
    y = (
        X[:, 0] * X[:, 1] +
        X[:, 2] * X[:, 3] -
        X[:, 4] * X[:, 5]
    )
    y = (y > 0).astype(int)
    return X, y

def test_proto_baseline():
    """Test Proto con sigma=0.02 (baseline sintético)"""
    print("="*70)
    print("TEST: Proto en Corral Baseline")
    print("="*70)

    X, y = make_corral_baseline()
    print(f"\nDataset: {X.shape}, {np.bincount(y)}")

    # Test 1: sigma=0.02 (esperado para sintético)
    print("\n" + "-"*70)
    print("TEST 1: sigma=0.02 (sintético)")
    print("-"*70)
    proto1 = ProtoDebug(sigma=0.02, k_protos=10)
    proto1.fit(X[:600], y[:600])
    rank1 = proto1.rank()
    top6_1 = set(rank1[:6])
    f1_1 = len(top6_1 & {0,1,2,3,4,5}) / 6
    print(f"\n✅ F1 = {f1_1:.2f}")

    # Test 2: sigma=0.10 (medio)
    print("\n" + "-"*70)
    print("TEST 2: sigma=0.10 (medio)")
    print("-"*70)
    proto2 = ProtoDebug(sigma=0.10, k_protos=10)
    proto2.fit(X[:600], y[:600])
    rank2 = proto2.rank()
    top6_2 = set(rank2[:6])
    f1_2 = len(top6_2 & {0,1,2,3,4,5}) / 6
    print(f"\n✅ F1 = {f1_2:.2f}")

    # Test 3: sigma=0.20 (alto)
    print("\n" + "-"*70)
    print("TEST 3: sigma=0.20 (alto)")
    print("-"*70)
    proto3 = ProtoDebug(sigma=0.20, k_protos=10)
    proto3.fit(X[:600], y[:600])
    rank3 = proto3.rank()
    top6_3 = set(rank3[:6])
    f1_3 = len(top6_3 & {0,1,2,3,4,5}) / 6
    print(f"\n✅ F1 = {f1_3:.2f}")

    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    print(f"sigma=0.02: F1={f1_1:.2f} {'✓' if f1_1 == 1.0 else '❌'}")
    print(f"sigma=0.10: F1={f1_2:.2f} {'✓' if f1_2 >= 0.83 else '❌'}")
    print(f"sigma=0.20: F1={f1_3:.2f} {'✓' if f1_3 >= 0.67 else '❌'}")

if __name__ == "__main__":
    test_proto_baseline()
