import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from tqdm import trange

class ReliefF_Original(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, discrete_threshold=10, n_features_to_select=None):
        self.n_neighbors = n_neighbors
        self.discrete_threshold = discrete_threshold
        self.feature_importances_ = None
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Determine discrete features
        discrete_mask = np.array([len(np.unique(X[:, i])) <= self.discrete_threshold for i in range(n_features)])
        continuous_mask = ~discrete_mask

        # Normalize continuous features
        X_proc = X.copy()
        if np.any(continuous_mask):
            scaler = MinMaxScaler()
            X_proc[:, continuous_mask] = scaler.fit_transform(X[:, continuous_mask])

        # Compute pairwise Manhattan distance
        # Nota: Esto es costoso en memoria O(N^2), correcto para el "base"
        distance = pairwise_distances(X_proc, metric='manhattan')
        np.fill_diagonal(distance, np.inf)

        score = np.zeros(n_features)
        class_probs = np.bincount(y) / len(y)

        # Loop principal (sin tqdm para no ensuciar logs de experimentos masivos, o opcional)
        for idx in range(n_samples):
            sorted_indices = np.lexsort((np.arange(n_samples), distance[idx]))

            hits, misses = [], defaultdict(list)
            for i in sorted_indices:
                if y[i] == y[idx] and len(hits) < self.n_neighbors:
                    hits.append(i)
                elif y[i] != y[idx] and len(misses[y[i]]) < self.n_neighbors:
                    misses[y[i]].append(i)

                if len(hits) == self.n_neighbors and all(len(v) >= self.n_neighbors for v in misses.values()):
                    break

            # Compute hit difference
            hit_diff = np.zeros(n_features)
            if hits:
                for i in hits:
                    hit_diff[discrete_mask] += (X_proc[idx, discrete_mask] != X_proc[i, discrete_mask])
                    hit_diff[continuous_mask] += np.abs(X_proc[idx, continuous_mask] - X_proc[i, continuous_mask])
                hit_diff /= len(hits)
                score -= hit_diff

            # Compute miss difference
            for label, miss_indices in misses.items():
                if label == y[idx] or not miss_indices:
                    continue
                p_label = class_probs[label] / (1 - class_probs[y[idx]] + 1e-8)
                miss_diff = np.zeros(n_features)
                for i in miss_indices:
                    miss_diff[discrete_mask] += (X_proc[idx, discrete_mask] != X_proc[i, discrete_mask])
                    miss_diff[continuous_mask] += np.abs(X_proc[idx, continuous_mask] - X_proc[i, continuous_mask])
                miss_diff /= len(miss_indices)
                score += (miss_diff * p_label)

        self.feature_importances_ = score
        return self

    def rank(self):
        if self.feature_importances_ is None:
            raise ValueError("Fit the model first.")
        return np.argsort(-self.feature_importances_)        

    def transform(self, X):
        """Selecciona las top-k features más importantes"""
        if self.feature_importances_ is None:
            raise ValueError("Fit the model first.")
        
        return X[:, self.rank()[:self.n_features_to_select]]
