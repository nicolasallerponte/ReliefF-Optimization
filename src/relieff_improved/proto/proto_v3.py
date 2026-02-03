"""
Proto ReliefF v3.1 - FIXED (Sin memory leaks)
==============================================

Mantiene las mejoras de v3 pero elimina los memory leaks.

Cambios sobre v3.0:
- ✅ Elimina copias innecesarias
- ✅ No usa LVQ (causa memory corruption)
- ✅ Simplifica clustering
- ✅ Limpia objetos correctamente

Autor: Nicolás Aller
Fecha: 2026-01-26
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

class Proto_v3(BaseEstimator, TransformerMixin):
    """
    Prototype-based ReliefF v3.1 - FIXED

    Usa clustering para generar prototipos representativos de cada clase,
    luego calcula importancias usando softmax weighting adaptativo.

    Parámetros
    ----------
    n_features_to_select : int, default=10
        Número de características a seleccionar
    sigma : float, default=0.1
        Control de número de prototipos: n_protos = max(1, int(n_samples_class * sigma))
    discrete_threshold : int, default=10
        Features con ≤ discrete_threshold valores únicos se tratan como discretas
    k_protos : int, default=5
        Número de prototipos más cercanos a considerar
    force_real_centers : bool, default=True
        Si True, mueve prototipos a muestras reales más cercanas
    use_lvq : bool, default=False
        LVQ DESACTIVADO para evitar crashes
    metric : str, default='euclidean'
        Métrica de distancia ('euclidean', 'manhattan')
    scaler : bool, default=True
        Si True, aplica MinMaxScaler a features continuas
    use_idw : bool, default=True
        Si True, usa softmax weighting
    gamma : float or 'auto' or 'adaptive', default='adaptive'
        Control de temperatura en softmax
    gamma_strategy : str, default='variance'
        Estrategia para gamma adaptativo ('variance', 'range')
    n_jobs : int, default=1
        Paralelización (solo 1 soportado actualmente)
    verbose : bool, default=False
        Mostrar mensajes
    random_state : int, default=42
        Semilla para reproducibilidad
    """

    def __init__(self,
                 n_features_to_select=10,
                 sigma=0.1,
                 discrete_threshold=10,
                 k_protos=5,
                 force_real_centers=True,
                 use_lvq=False,
                 metric='euclidean',
                 scaler=True,
                 use_idw=True,
                 gamma='adaptive',
                 gamma_strategy='variance',
                 n_jobs=1,
                 verbose=False,
                 random_state=42):

        self.n_features_to_select = n_features_to_select
        self.sigma = sigma
        self.discrete_threshold = discrete_threshold
        self.k_protos = k_protos
        self.force_real_centers = force_real_centers
        self.use_lvq = use_lvq
        self.metric = metric
        self.scaler = scaler
        self.use_idw = use_idw
        self.gamma = gamma
        self.gamma_strategy = gamma_strategy
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.feature_importances_ = None
        self.prototypes_ = None

    def _dist_func(self, X, Y):
        """Calcula distancias"""
        if self.metric == 'manhattan':
            return manhattan_distances(X, Y)
        return euclidean_distances(X, Y)

    def _compute_adaptive_softmax_weights(self, dists):
        """Softmax con temperatura adaptativa"""
        mean_dist = np.mean(dists, axis=1, keepdims=True)
        std_dist = np.std(dists, axis=1, keepdims=True) + 1e-8

        # Determinar gamma
        if isinstance(self.gamma, (int, float)):
            adaptive_gamma = self.gamma
        elif self.gamma == 'auto':
            adaptive_gamma = 1.0 / (std_dist + 1e-8)
        elif self.gamma == 'adaptive':
            if self.gamma_strategy == 'variance':
                adaptive_gamma = 1.0 / (std_dist + 1e-8)
            elif self.gamma_strategy == 'range':
                dist_range = np.ptp(dists, axis=1, keepdims=True) + 1e-8
                adaptive_gamma = 1.0 / dist_range
            else:
                adaptive_gamma = 1.0
        else:
            adaptive_gamma = 1.0

        # Normalización para estabilidad
        shift = np.min(dists, axis=1, keepdims=True)
        normalized_dists = (dists - shift) / (std_dist + 1e-8)

        # Softmax
        exp_dists = np.exp(-adaptive_gamma * normalized_dists)
        weights = exp_dists / (np.sum(exp_dists, axis=1, keepdims=True) + 1e-8)

        return weights[:, :, np.newaxis]

    def _generate_prototypes(self, X, y, c, count):
        """Generación de prototipos por clase"""
        X_subset = X[y == c]
        n_clusters = max(1, int(count * self.sigma))
        n_clusters = min(n_clusters, len(X_subset), 100)  # Cap a 100

        if count <= n_clusters or len(X_subset) < n_clusters:
            return X_subset

        # Clustering simple con KMeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=3,
            max_iter=100,
            random_state=self.random_state
        )
        kmeans.fit(X_subset)
        centers = kmeans.cluster_centers_.astype(np.float32)

        # Snap to real samples si se solicita
        if self.force_real_centers:
            dists = self._dist_func(X_subset, centers)
            real_indices = np.argmin(dists, axis=0)
            return X_subset[real_indices]

        return centers

    def fit(self, X, y):
        """
        Ajusta el modelo

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Datos de entrenamiento
        y : array-like of shape (n_samples,)
            Etiquetas

        Returns
        -------
        self
        """
        # Conversión simple (sin copias innecesarias)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        n_samples, n_features = X.shape
        classes = np.unique(y)

        # Detectar features discretas
        discrete_mask = np.zeros(n_features, dtype=bool)
        if self.discrete_threshold > 0:
            for i in range(n_features):
                if len(np.unique(X[:, i])) <= self.discrete_threshold:
                    discrete_mask[i] = True

        # Escalar features continuas IN-PLACE
        if self.scaler and np.any(~discrete_mask):
            scaler = MinMaxScaler()
            X[:, ~discrete_mask] = scaler.fit_transform(X[:, ~discrete_mask])

        # Calcular probabilidades de clase
        class_counts = {c: np.sum(y == c) for c in classes}
        p_classes = {c: count / n_samples for c, count in class_counts.items()}

        # Generar prototipos por clase
        prototypes = {}
        for c in classes:
            count = class_counts[c]
            prototypes[c] = self._generate_prototypes(X, y, c, count)

        self.prototypes_ = prototypes

        # Calcular importancias
        scores = np.zeros(n_features, dtype=np.float32)

        for c in classes:
            mask = (y == c)
            X_c_indices = np.where(mask)[0]
            if len(X_c_indices) == 0:
                continue

            n_c = len(X_c_indices)
            my_protos = prototypes[c]

            # FASE HIT (scoring interno)
            sum_diffs_hit = np.zeros(n_features, dtype=np.float32)

            for idx in X_c_indices:
                x = X[idx:idx+1]
                dists = self._dist_func(x, my_protos)

                k_eff = min(self.k_protos, len(my_protos))
                nearest_idx = np.argpartition(dists[0], k_eff - 1)[:k_eff]
                nearest_dists = dists[0, nearest_idx]
                chosen = my_protos[nearest_idx]

                diffs = np.abs(x - chosen)

                if self.use_idw:
                    weights = self._compute_adaptive_softmax_weights(nearest_dists.reshape(1, -1))
                    mean_diffs = np.sum(diffs * weights[0], axis=0)
                else:
                    mean_diffs = np.mean(diffs, axis=0)

                sum_diffs_hit += mean_diffs

            scores -= (sum_diffs_hit / n_c) * p_classes[c]

            # FASE MISS (scoring externo)
            prob_not_c = 1.0 - p_classes[c]
            if prob_not_c == 0:
                prob_not_c = 1.0

            for other_c in classes:
                if other_c == c:
                    continue

                other_protos = prototypes[other_c]
                sum_diffs_miss = np.zeros(n_features, dtype=np.float32)

                for idx in X_c_indices:
                    x = X[idx:idx+1]
                    dists_m = self._dist_func(x, other_protos)

                    k_eff_m = min(self.k_protos, len(other_protos))
                    nearest_idx_m = np.argpartition(dists_m[0], k_eff_m - 1)[:k_eff_m]
                    nearest_dists_m = dists_m[0, nearest_idx_m]
                    chosen_m = other_protos[nearest_idx_m]

                    diffs = np.abs(x - chosen_m)

                    if self.use_idw:
                        weights = self._compute_adaptive_softmax_weights(nearest_dists_m.reshape(1, -1))
                        mean_diffs = np.sum(diffs * weights[0], axis=0)
                    else:
                        mean_diffs = np.mean(diffs, axis=0)

                    sum_diffs_miss += mean_diffs

                weight = (p_classes[other_c] / prob_not_c)
                scores += (sum_diffs_miss / n_c) * weight * p_classes[c]

        self.feature_importances_ = scores

        return self

    def rank(self):
        """Retorna índices de características ordenadas por importancia"""
        if self.feature_importances_ is None:
            raise ValueError("Modelo no ajustado. Llama a fit() primero.")
        return np.argsort(-self.feature_importances_)

    def transform(self, X):
        """Transforma X seleccionando las mejores características"""
        if self.feature_importances_ is None:
            raise ValueError("Modelo no ajustado. Llama a fit() primero.")

        X = np.asarray(X, dtype=np.float32)
        idx = self.rank()[:self.n_features_to_select]
        return X[:, idx]
