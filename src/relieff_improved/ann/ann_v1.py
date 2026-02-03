import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import hnswlib
from numba import njit, prange

# --- NÚCLEO JIT (Fuera de la clase para que Numba lo compile bien) ---
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
    def __init__(self, n_neighbors=10, discrete_threshold=10, 
                 iter_ratio=1.0, n_jobs=-1, random_state=42):
        self.n_neighbors = n_neighbors
        self.discrete_threshold = discrete_threshold
        self.iter_ratio = iter_ratio
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.feature_importances_ = None
        
    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        n_samples, n_features = X.shape
        
        # 1. LÓGICA INTELIGENTE DE PARÁMETROS
        # Corrección de Precisión: Si hay pocos datos, NO hacemos subsampling
        effective_ratio = self.iter_ratio
        if n_samples < 1000:
            effective_ratio = 1.0 # Forzar modo exacto completo para precisión
            
        # Corrección de Tiempo: Usar HNSW solo si el dataset es grande
        USE_HNSW = n_samples >= 20000

        # Preprocesamiento 
        discrete_mask = np.zeros(n_features, dtype=np.bool_)
        for i in range(n_features):
            if len(np.unique(X[:, i])) <= self.discrete_threshold:
                discrete_mask[i] = True
        
        if np.any(~discrete_mask):
            scaler = MinMaxScaler()
            X[:, ~discrete_mask] = scaler.fit_transform(X[:, ~discrete_mask])

        # Selección de Muestras (Query)
        n_iters = int(n_samples * effective_ratio)
        if n_iters < self.n_neighbors + 2: n_iters = n_samples
        
        np.random.seed(self.random_state)
        if effective_ratio < 1.0:
            query_indices = np.random.choice(n_samples, n_iters, replace=False)
        else:
            query_indices = np.arange(n_samples) # Usar todos en orden
            
        X_query = X[query_indices]
        y_query = y[query_indices]
        
        class_counts = np.bincount(y)
        p_c = (class_counts / n_samples).astype(np.float32)

        # Construcción de Índices
        max_neighbors = len(class_counts) * self.n_neighbors
        final_neighbors = np.full((n_iters, max_neighbors), -1, dtype=np.int32)
        classes = np.unique(y)
        indices_by_class = {}
        
        for c in classes:
            idx_c = np.where(y == c)[0]
            if len(idx_c) <= self.n_neighbors: continue
            
            if USE_HNSW:
                p = hnswlib.Index(space='l2', dim=n_features)
                # Ajustamos parámetros HNSW para construcción más rápida en rangos medios
                M_param = 16 if n_samples > 5000 else 8
                p.init_index(max_elements=len(idx_c), ef_construction=100, M=M_param)
                p.add_items(X[idx_c], idx_c) 
                p.set_ef(max(50, self.n_neighbors * 2))
                indices_by_class[c] = p
            else:
                # Fuerza bruta vectorizada rápida de Sklearn
                nn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto', n_jobs=self.n_jobs)
                nn.fit(X[idx_c])
                indices_by_class[c] = (nn, idx_c)

        # Búsqueda (Query)
        for i, q_idx in enumerate(query_indices):
            current_y = y[q_idx]
            insert_pos = 0
            
            # Hits
            if current_y in indices_by_class:
                if USE_HNSW:
                    idx_c = indices_by_class[current_y]
                    labels, _ = idx_c.knn_query(X[q_idx].reshape(1, -1), k=self.n_neighbors+1)
                    found = labels[0]
                else:
                    nn_model, real_indices = indices_by_class[current_y]
                    dists, rel_ind = nn_model.kneighbors(X[q_idx].reshape(1, -1), n_neighbors=min(len(real_indices), self.n_neighbors+1))
                    found = real_indices[rel_ind[0]]
                
                count = 0
                for f_idx in found:
                    if f_idx != q_idx and count < self.n_neighbors:
                        final_neighbors[i, insert_pos] = f_idx
                        insert_pos += 1
                        count += 1
            
            # Misses
            for c in classes:
                if c == current_y: continue
                if c not in indices_by_class: continue
                
                if USE_HNSW:
                    idx_c = indices_by_class[c]
                    labels, _ = idx_c.knn_query(X[q_idx].reshape(1, -1), k=self.n_neighbors)
                    found = labels[0]
                else:
                    nn_model, real_indices = indices_by_class[c]
                    dists, rel_ind = nn_model.kneighbors(X[q_idx].reshape(1, -1), n_neighbors=min(len(real_indices), self.n_neighbors))
                    found = real_indices[rel_ind[0]]
                
                for f_idx in found:
                    if insert_pos < max_neighbors:
                        final_neighbors[i, insert_pos] = f_idx
                        insert_pos += 1

        # Cálculo JIT
        self.feature_importances_ = _compute_scores_numba(
            X_query, X, final_neighbors, y, y_query, 
            discrete_mask, p_c, self.n_neighbors
        )
        return self

    def rank(self):
        if self.feature_importances_ is None:
            raise ValueError("Fit the model first.")
        return np.argsort(-self.feature_importances_)