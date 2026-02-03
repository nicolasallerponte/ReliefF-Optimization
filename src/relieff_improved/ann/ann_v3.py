"""
ANN ReliefF v3.1 - FIXED (Sin memory leaks)
===========================================

Mantiene las mejoras de v3 pero elimina los memory leaks que causaban crashes.

Cambios sobre v3.0:
- ✅ Elimina copias innecesarias de X
- ✅ No usa check_X_y (hace copia)
- ✅ No acumula historial de convergencia
- ✅ Limpia objetos hnswlib correctamente
- ✅ Simplifica gestión de memoria

Autor: Nicolás Aller
Fecha: 2026-01-26
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import hnswlib
from numba import njit, prange

@njit(fastmath=True, parallel=True)
def _compute_scores_numba_batched(X_sample, X_full, neighbor_indices, neighbor_labels,
                                   sample_labels, discrete_mask, p_c, k, batch_size=512):
    """Versión optimizada con batch processing"""
    n_samples_iter = X_sample.shape[0]
    n_features = X_sample.shape[1]
    scores = np.zeros(n_features, dtype=np.float32)

    for batch_start in range(0, n_samples_iter, batch_size):
        batch_end = min(batch_start + batch_size, n_samples_iter)
        batch_scores = np.zeros(n_features, dtype=np.float32)

        for i in prange(batch_start, batch_end):
            target_y = sample_labels[i]
            target_row = X_sample[i]

            hits_diff = np.zeros(n_features, dtype=np.float32)
            miss_diff = np.zeros((len(p_c), n_features), dtype=np.float32)
            hits_count = 0.0
            miss_counts = np.zeros(len(p_c), dtype=np.float32)

            for neighbor_idx in neighbor_indices[i]:
                if neighbor_idx == -1:
                    continue

                n_y = neighbor_labels[neighbor_idx]
                row_neighbor = X_full[neighbor_idx]
                diff = np.abs(target_row - row_neighbor)

                for f in range(n_features):
                    if discrete_mask[f]:
                        diff[f] = 1.0 if target_row[f] != row_neighbor[f] else 0.0

                if n_y == target_y:
                    hits_diff += diff
                    hits_count += 1
                else:
                    miss_diff[n_y] += diff
                    miss_counts[n_y] += 1

            sample_score = np.zeros(n_features, dtype=np.float32)
            if hits_count > 0:
                sample_score -= (hits_diff / hits_count)

            prob_target = p_c[target_y]
            denom = 1.0 - prob_target
            if denom == 0:
                denom = 1.0

            for c_idx in range(len(p_c)):
                if c_idx == target_y:
                    continue
                if miss_counts[c_idx] > 0:
                    weight = p_c[c_idx] / denom
                    sample_score += (miss_diff[c_idx] / miss_counts[c_idx]) * weight

            batch_scores += sample_score

        scores += batch_scores

    return scores


class ANN_v3(BaseEstimator, TransformerMixin):
    """
    ANN-based ReliefF v3.1 - FIXED (sin memory leaks)

    Parámetros
    ----------
    n_neighbors : int, default=10
        Número de vecinos a buscar por clase
    n_features_to_select : int, default=10
        Número de características a seleccionar
    discrete_threshold : int, default=10
        Umbral para considerar una característica como discreta
    iter_ratio : float or 'auto', default='auto'
        Proporción de muestras a iterar
    n_jobs : int, default=1
        Número de cores (actualmente solo 1 soportado)
    random_state : int, default=42
        Semilla para reproducibilidad
    metric : str, default='euclidean'
        Métrica de distancia ('euclidean', 'manhattan')
    M : int, default=16
        Parámetro M de HNSW
    ef_construction : int, default=200
        Parámetro ef durante construcción del índice HNSW
    ef_search : int or 'auto', default='auto'
        Parámetro ef durante búsqueda
    scaler : bool, default=True
        Si True, aplica MinMaxScaler a features continuas
    verbose : bool, default=False
        Mostrar mensajes
    """

    def __init__(self,
                 n_neighbors=10,
                 n_features_to_select=10,
                 discrete_threshold=10,
                 iter_ratio='auto',
                 n_jobs=1,
                 random_state=42,
                 metric='euclidean',
                 M=16,
                 ef_construction=200,
                 ef_search='auto',
                 scaler=True,
                 verbose=False):

        self.n_neighbors = n_neighbors
        self.n_features_to_select = n_features_to_select
        self.discrete_threshold = discrete_threshold
        self.iter_ratio = iter_ratio
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.scaler = scaler
        self.verbose = verbose

        self.feature_importances_ = None

    def _get_adaptive_params(self, n_samples):
        """Determina parámetros adaptativos"""
        if self.iter_ratio == 'auto':
            if n_samples < 1000:
                effective_ratio = 1.0
            elif n_samples < 5000:
                effective_ratio = 0.7
            elif n_samples < 10000:
                effective_ratio = 0.5
            else:
                effective_ratio = 0.3
        else:
            effective_ratio = float(self.iter_ratio)

        # HNSW solo para datasets grandes
        use_hnsw = n_samples >= 5000  # Aumentado de 3000 para mayor estabilidad

        final_M = self.M
        final_ef = self.ef_construction

        if use_hnsw and self.M == 16 and self.ef_construction == 200:
            if n_samples >= 50000:
                final_M, final_ef = 24, 180
            elif n_samples >= 10000:
                final_M, final_ef = 20, 150

        return effective_ratio, use_hnsw, final_M, final_ef

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

        # Parámetros adaptativos
        effective_ratio, use_hnsw, M_param, ef_construction = self._get_adaptive_params(n_samples)

        # Detectar features discretas
        discrete_mask = np.zeros(n_features, dtype=bool)
        for i in range(n_features):
            if len(np.unique(X[:, i])) <= self.discrete_threshold:
                discrete_mask[i] = True

        # Escalar features continuas IN-PLACE (sin copias)
        if self.scaler and np.any(~discrete_mask):
            scaler = MinMaxScaler()
            X[:, ~discrete_mask] = scaler.fit_transform(X[:, ~discrete_mask])

        # Seleccionar queries
        n_iters = int(n_samples * effective_ratio)
        if n_iters < self.n_neighbors + 2:
            n_iters = n_samples

        np.random.seed(self.random_state)
        if effective_ratio < 1.0:
            query_indices = np.random.choice(n_samples, n_iters, replace=False)
        else:
            query_indices = np.arange(n_samples)

        X_query = X[query_indices]
        y_query = y[query_indices]

        # Calcular probabilidades de clase
        classes = np.unique(y)
        class_counts = np.bincount(y)
        p_c = (class_counts / n_samples).astype(np.float32)

        # Configurar métrica
        hnsw_space = 'l2' if self.metric in ['euclidean', 'l2'] else 'l1'
        sklearn_p = 2 if self.metric in ['euclidean', 'l2'] else 1

        # Construir índices por clase
        indices_by_class = {}
        hnsw_indices = []  # Lista para limpiar después

        for c in classes:
            idx_c = np.where(y == c)[0]
            if len(idx_c) <= 1:
                continue

            try:
                if use_hnsw:
                    p = hnswlib.Index(space=hnsw_space, dim=n_features)
                    p.init_index(
                        max_elements=len(idx_c),
                        ef_construction=ef_construction,
                        M=M_param
                    )
                    p.add_items(X[idx_c], idx_c)

                    # Configurar ef_search
                    if self.ef_search == 'auto':
                        ef_val = max(self.n_neighbors * 2, 100)
                    else:
                        ef_val = int(self.ef_search)
                    p.set_ef(ef_val)

                    indices_by_class[c] = ('hnsw', p, idx_c)
                    hnsw_indices.append(p)  # Guardar referencia
                else:
                    raise Exception("Use exact search")

            except Exception:
                # Fallback a búsqueda exacta
                nn = NearestNeighbors(
                    n_neighbors=min(self.n_neighbors + 1, len(idx_c)),
                    algorithm='brute',
                    p=sklearn_p,
                    n_jobs=1
                )
                nn.fit(X[idx_c])
                indices_by_class[c] = ('sklearn', nn, idx_c)

        # Buscar vecinos
        max_neighbors = min(len(classes) * self.n_neighbors, n_samples - 1)
        final_neighbors = np.full((n_iters, max_neighbors), -1, dtype=np.int32)

        for i, q_idx in enumerate(query_indices):
            current_y = y[q_idx]
            insert_pos = 0

            for c in classes:
                if c not in indices_by_class:
                    continue

                index_type, index_obj, real_indices = indices_by_class[c]
                k_search = min(
                    self.n_neighbors + 1 if c == current_y else self.n_neighbors,
                    len(real_indices)
                )

                if index_type == 'hnsw':
                    labels, _ = index_obj.knn_query(X[q_idx].reshape(1, -1), k=k_search)
                    found = labels[0]
                else:
                    _, rel_ind = index_obj.kneighbors(X[q_idx].reshape(1, -1), n_neighbors=k_search)
                    found = real_indices[rel_ind[0]]

                for f_idx in found:
                    if f_idx != q_idx and insert_pos < max_neighbors:
                        final_neighbors[i, insert_pos] = f_idx
                        insert_pos += 1
                    if c == current_y and insert_pos >= self.n_neighbors:
                        break

        # Calcular scores
        self.feature_importances_ = _compute_scores_numba_batched(
            X_query, X, final_neighbors, y, y_query,
            discrete_mask, p_c, self.n_neighbors, batch_size=512
        )

        # Normalizar
        if n_iters > 0 and np.any(self.feature_importances_ != 0):
            self.feature_importances_ /= n_iters

        # LIMPIEZA EXPLÍCITA de objetos hnswlib
        for p in hnsw_indices:
            try:
                del p
            except:
                pass

        return self

    def rank(self):
        """Retorna índices de características ordenadas por importancia"""
        if self.feature_importances_ is None:
            raise ValueError("Modelo no ajustado. Llama a fit() primero.")
        return np.argsort(-self.feature_importances_)

    def transform(self, X, n_features_to_select=None):
        """Transforma X seleccionando las mejores características"""
        if self.feature_importances_ is None:
            raise ValueError("Modelo no ajustado. Llama a fit() primero.")

        X = np.asarray(X, dtype=np.float32)

        if n_features_to_select is None:
            n_features_to_select = self.n_features_to_select

        selected_indices = self.rank()[:n_features_to_select]
        return X[:, selected_indices]
