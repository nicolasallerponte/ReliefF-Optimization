import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import hnswlib
from numba import njit, prange

# --- NÚCLEO JIT (Sin cambios) ---
@njit(fastmath=True, parallel=True)
def _compute_scores_numba(X_sample, X_full, neighbor_indices, neighbor_labels, 
                          sample_labels, discrete_mask, p_c, k):
    n_samples_iter = X_sample.shape[0]
    n_features = X_sample.shape[1]
    scores_matrix = np.zeros((n_samples_iter, n_features), dtype=np.float32)
    
    for i in prange(n_samples_iter):
        target_y = sample_labels[i]
        target_row = X_sample[i]
        
        hits_diff = np.zeros(n_features, dtype=np.float32)
        miss_diff = np.zeros((len(p_c), n_features), dtype=np.float32)
        hits_count = 0.0
        miss_counts = np.zeros(len(p_c), dtype=np.float32)
        
        for neighbor_idx in neighbor_indices[i]:
            if neighbor_idx == -1: continue 
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
        if denom == 0: denom = 1.0
        
        for c_idx in range(len(p_c)):
            if c_idx == target_y: continue
            if miss_counts[c_idx] > 0:
                weight = p_c[c_idx] / denom
                sample_score += (miss_diff[c_idx] / miss_counts[c_idx]) * weight
        
        scores_matrix[i] = sample_score

    return np.sum(scores_matrix, axis=0)

# --- CLASE PRINCIPAL ---
class ANN(BaseEstimator, TransformerMixin):
    """
    ANN ReliefF Optimizado v3.3 (Nuclear-Ready: ef_search expuesto)
    """
    
    def __init__(self, n_neighbors=10, n_features_to_select=10, discrete_threshold=10, 
                 iter_ratio='auto', n_jobs=1, random_state=42,
                 convergence_threshold=1e-4, min_iterations=5,
                 metric='euclidean', M=16, ef_construction=200, ef_search=-1): # <--- NUEVO PARAM
        
        self.n_neighbors = n_neighbors
        self.n_features_to_select = n_features_to_select 
        self.discrete_threshold = discrete_threshold
        self.iter_ratio = iter_ratio
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.convergence_threshold = convergence_threshold
        self.min_iterations = min_iterations
        
        # Parámetros avanzados HNSW
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search # Control de precisión de búsqueda
        self.feature_importances_ = None
        
    def _get_adaptive_params(self, n_samples):
        effective_ratio = self.iter_ratio
        if self.iter_ratio == 'auto':
            if n_samples < 1000: effective_ratio = 1.0
            elif n_samples < 10000: effective_ratio = 0.5
            elif n_samples < 50000: effective_ratio = 0.3
            else: effective_ratio = 0.2
            
        use_hnsw = n_samples >= 1
        
        # Lógica de defaults solo si no se fuerzan en init
        # (Aquí simplificamos: si el usuario puso defaults en init, usamos esos)
        final_M = self.M
        final_ef = self.ef_construction
        
        # Si estamos en modo "auto" (valores por defecto bajos), aplicamos lógica adaptativa
        # Pero si el usuario puso M=100, respetamos M=100.
        # Detectamos si son los defaults de la firma:
        if use_hnsw and self.M == 16 and self.ef_construction == 200:
             if n_samples >= 50000:
                final_M, final_ef = 24, 150
             if n_samples >= 100000:
                final_M, final_ef = 32, 200

        return effective_ratio, use_hnsw, final_M, final_ef
        
    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        n_samples, n_features = X.shape
        
        effective_ratio, USE_HNSW, M_param, ef_construction = self._get_adaptive_params(n_samples)
        
        discrete_mask = np.zeros(n_features, dtype=bool)
        for i in range(n_features):
            if len(np.unique(X[:, i])) <= self.discrete_threshold:
                discrete_mask[i] = True
        
        if np.any(~discrete_mask):
            scaler = MinMaxScaler()
            X[:, ~discrete_mask] = scaler.fit_transform(X[:, ~discrete_mask])

        n_iters = int(n_samples * effective_ratio)
        if n_iters < self.n_neighbors + 2: n_iters = n_samples
        
        np.random.seed(self.random_state)
        if effective_ratio < 1.0:
            query_indices = np.random.choice(n_samples, n_iters, replace=False)
        else:
            query_indices = np.arange(n_samples)
            
        X_query = X[query_indices]
        y_query = y[query_indices]
        
        classes = np.unique(y)
        max_neighbors = min(len(classes) * self.n_neighbors, n_samples - 1)
        final_neighbors = np.full((n_iters, max_neighbors), -1, dtype=np.int32)
        
        # Configurar métrica
        hnsw_space = 'l2'
        sklearn_p = 2
        if self.metric == 'manhattan' or self.metric == 'l1':
            hnsw_space = 'l1'
            sklearn_p = 1
        
        indices_by_class = {}
        for c in classes:
            idx_c = np.where(y == c)[0]
            if len(idx_c) <= 1: continue
            
            if USE_HNSW:
                try:
                    p = hnswlib.Index(space=hnsw_space, dim=n_features)
                    p.init_index(max_elements=len(idx_c), 
                               ef_construction=ef_construction, 
                               M=M_param)
                    p.add_items(X[idx_c], idx_c)
                    
                    # CONFIGURACIÓN DE PRECISIÓN DE BÚSQUEDA
                    # Si ef_search es -1, usamos la heurística antigua. Si no, usamos el valor forzado.
                    if self.ef_search > 0:
                        p.set_ef(self.ef_search)
                    else:
                        p.set_ef(max(self.n_neighbors * 2, 100))
                        
                    indices_by_class[c] = ('hnsw', p, idx_c)
                except Exception:
                    USE_HNSW = False 
            
            if not USE_HNSW:
                nn = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, len(idx_c)), 
                                      algorithm='brute', p=sklearn_p, n_jobs=1)
                nn.fit(X[idx_c])
                indices_by_class[c] = ('sklearn', nn, idx_c)

        for i, q_idx in enumerate(query_indices):
            current_y = y[q_idx]
            insert_pos = 0
            for c in classes:
                if c not in indices_by_class: continue
                index_type, index_obj, real_indices = indices_by_class[c]
                k_search = min(self.n_neighbors + 1 if c == current_y else self.n_neighbors, len(real_indices))
                
                if index_type == 'hnsw':
                    labels, _ = index_obj.knn_query(X[q_idx].reshape(1, -1), k=k_search)
                    found = labels[0]
                else:
                    _, rel_ind = index_obj.kneighbors(X[q_idx].reshape(1, -1), n_neighbors=k_search)
                    found = real_indices[rel_ind[0]]
                
                for f_idx in found:
                    if f_idx != q_idx:  
                        if insert_pos < max_neighbors:
                            final_neighbors[i, insert_pos] = f_idx
                            insert_pos += 1
                        if c == current_y and insert_pos >= self.n_neighbors: break

        self.feature_importances_ = _compute_scores_numba(
            X_query, X, final_neighbors, y, y_query, 
            discrete_mask, (np.bincount(y) / n_samples).astype(np.float32), self.n_neighbors
        )
        if np.any(self.feature_importances_ != 0): self.feature_importances_ /= n_iters
        return self

    def rank(self):
        if self.feature_importances_ is None: raise ValueError("Fit first.")
        return np.argsort(-self.feature_importances_)
    
    def transform(self, X, n_features_to_select=None):
        if self.feature_importances_ is None: raise ValueError("Fit first.")
        if n_features_to_select is None: n_features_to_select = self.n_features_to_select
        if n_features_to_select is None: n_features_to_select = X.shape[1] // 2
        return X[:, self.rank()[:n_features_to_select]]