import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

class Proto(BaseEstimator, TransformerMixin):
    """
    ReliefF Proto implementation (Optimized v2.0).
    Características:
    - Softmax Weighting (Estabilidad numérica y selectividad).
    - Gestión eficiente de memoria (Batches y Float32).
    - LVQ Vectorizado.
    """

    def __init__(self, n_features_to_select=10, sigma=0.05, discrete_threshold=10, 
                 k_protos=3, force_real_centers=True, use_lvq=False, lvq_epochs=5, lvq_lr=0.1,
                 metric='manhattan', lvq_batch_size=128,
                 scaler_type='robust', use_idw=True, gamma=1.0): 
        
        self.n_features_to_select = n_features_to_select
        self.feature_importances_ = None
        self.discrete_threshold = discrete_threshold
        self.sigma = sigma
        self.k_protos = k_protos
        self.force_real_centers = force_real_centers
        self.use_lvq = use_lvq
        self.lvq_epochs = lvq_epochs
        self.lvq_lr = lvq_lr
        self.metric = metric
        self.lvq_batch_size = lvq_batch_size
        self.scaler_type = scaler_type
        self.use_idw = use_idw # Si es True, usa Softmax. Si es False, usa Media simple.
        self.gamma = gamma     # Controla la agresividad del Softmax

    def _dist_func(self, X, Y):
        if self.metric == 'manhattan':
            return manhattan_distances(X, Y)
        return euclidean_distances(X, Y)

    def _refine_prototypes_lvq_vectorized(self, X, y, prototypes_dict):
        """Ajusta los prototipos para separar mejor las clases."""
        flat_protos = []
        proto_labels = []
        structure_map = {}
        sorted_classes = sorted(prototypes_dict.keys())
        
        for c in sorted_classes:
            protos = prototypes_dict[c]
            structure_map[c] = len(protos)
            for p in protos:
                flat_protos.append(p)
                proto_labels.append(c)
                
        flat_protos = np.array(flat_protos, dtype=np.float32)
        proto_labels = np.array(proto_labels)
        
        n_samples = len(X)
        limit = min(5000, n_samples) # Limitamos LVQ a 5000 muestras para velocidad
        
        for epoch in range(self.lvq_epochs):
            current_lr = self.lvq_lr * (1.0 - (epoch / self.lvq_epochs))
            indices = np.random.choice(n_samples, limit, replace=False)
            
            for i in range(0, limit, self.lvq_batch_size):
                batch_idx = indices[i : i + self.lvq_batch_size]
                if len(batch_idx) == 0: break
                
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                dists = self._dist_func(X_batch, flat_protos)
                winners_idx = np.argmin(dists, axis=1)
                winners_labels = proto_labels[winners_idx]
                
                diffs = X_batch - flat_protos[winners_idx]
                # Si acierta la clase, acerca el prototipo (+). Si falla, lo aleja (-).
                signs = np.where(winners_labels == y_batch, 1.0, -1.0)[:, np.newaxis]
                updates = current_lr * signs * diffs
                np.add.at(flat_protos, winners_idx, updates)

        new_prototypes = {}
        cursor = 0
        for c in sorted_classes:
            count = structure_map[c]
            new_prototypes[c] = flat_protos[cursor : cursor + count]
            cursor += count
        return new_prototypes

    def _get_real_prototypes(self, X_indices, X_full, abstract_centers):
        """Mueve los centroides abstractos a la muestra real más cercana (Snap-to-Grid)."""
        X_subset = X_full[X_indices]
        dists = self._dist_func(X_subset, abstract_centers)
        real_indices_local = np.argmin(dists, axis=0)
        return X_subset[real_indices_local]

    def _compute_softmax_weights(self, dists):
        """Calcula pesos estables usando Softmax."""
        # Shift para estabilidad numérica (evitar exp de números muy negativos)
        # Hacemos que la distancia mínima sea 0 para el exponente -> exp(0) = 1 (peso máximo)
        shift = np.min(dists, axis=1, keepdims=True)
        exp_dists = np.exp(-self.gamma * (dists - shift))
        weights = exp_dists / np.sum(exp_dists, axis=1, keepdims=True)
        return weights[:, :, np.newaxis] # Añadir dimensión para broadcasting

    def fit(self, X, y):
        # 1. Optimización de tipos
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int32)
        n_samples, n_features = X.shape
        classes = np.unique(y)

        # 2. Preprocesamiento (Scaling)
        discrete_mask = np.zeros(n_features, dtype=bool)
        if self.discrete_threshold > 0:
            for i in range(n_features):
                if len(np.unique(X[:, i])) <= self.discrete_threshold:
                    discrete_mask[i] = True
        
        if np.any(~discrete_mask):
            if self.scaler_type == 'robust':
                scaler = RobustScaler()
                X[:, ~discrete_mask] = scaler.fit_transform(X[:, ~discrete_mask])
                # Clip para evitar valores extremos que rompan las distancias
                X[:, ~discrete_mask] = np.clip(X[:, ~discrete_mask], -5.0, 5.0)
            else:
                scaler = MinMaxScaler()
                X[:, ~discrete_mask] = scaler.fit_transform(X[:, ~discrete_mask])
        
        gc.collect()

        # 3. Generación de Prototipos (Clustering)
        prototypes = {}
        class_counts = {}
        for c in classes:
            indices = np.where(y == c)[0]
            count = len(indices)
            class_counts[c] = count
            
            # Número dinámico de clusters según sigma
            n_clusters = int(max(1, min(200, count * self.sigma)))
            
            X_subset = X[indices]
            if count <= n_clusters:
                prototypes[c] = X_subset
            else:
                # Batch size grande para velocidad
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, n_init=3).fit(X_subset)
                prototypes[c] = kmeans.cluster_centers_.astype(np.float32)
            del X_subset # Liberar memoria
        
        gc.collect()

        # 4. Refinamiento LVQ (Opcional)
        if self.use_lvq:
            prototypes = self._refine_prototypes_lvq_vectorized(X, y, prototypes)
            gc.collect()

        # 5. Snap to Grid (Opcional: usar muestras reales en vez de medias)
        if self.force_real_centers:
            for c in classes:
                if class_counts[c] > len(prototypes[c]):
                    prototypes[c] = self._get_real_prototypes(np.where(y==c)[0], X, prototypes[c])
        gc.collect()

        # 6. Cálculo de Importancias (CORE DEL ALGORITMO)
        p_classes = {c: count / n_samples for c, count in class_counts.items()}
        scores = np.zeros(n_features, dtype=np.float32)
        SCORE_BATCH_SIZE = 256 # Batch más grande es seguro aquí
        
        for c in classes:
            mask = (y == c)
            X_c_indices = np.where(mask)[0]
            if len(X_c_indices) == 0: continue
            
            n_c = len(X_c_indices)
            my_protos = prototypes[c]
            
            # --- FASE HIT ---
            sum_mean_diffs = np.zeros(n_features, dtype=np.float32)
            for i in range(0, n_c, SCORE_BATCH_SIZE):
                end = min(i + SCORE_BATCH_SIZE, n_c)
                batch_idx = X_c_indices[i:end]
                X_batch = X[batch_idx] 
                
                dists_batch = self._dist_func(X_batch, my_protos)
                k_eff = min(self.k_protos, len(my_protos))
                
                # Top-k más cercanos
                nearest_idx = np.argpartition(dists_batch, k_eff - 1, axis=1)[:, :k_eff]
                nearest_dists = np.take_along_axis(dists_batch, nearest_idx, axis=1)
                chosen_batch = my_protos[nearest_idx] 
                
                diffs_batch = np.abs(X_batch[:, np.newaxis, :] - chosen_batch)
                
                if self.use_idw: # AHORA USA SOFTMAX
                    weights = self._compute_softmax_weights(nearest_dists)
                    mean_diffs_batch = np.sum(diffs_batch * weights, axis=1)
                else:
                    mean_diffs_batch = np.mean(diffs_batch, axis=1)
                
                sum_mean_diffs += np.sum(mean_diffs_batch, axis=0)
            
            scores -= (sum_mean_diffs / n_c) * p_classes[c]
            gc.collect()

            # --- FASE MISS ---
            prob_not_c = 1.0 - p_classes[c]
            if prob_not_c == 0: prob_not_c = 1.0

            for other_c in classes:
                if other_c == c: continue
                other_protos = prototypes[other_c]
                
                sum_mean_diffs_m = np.zeros(n_features, dtype=np.float32)

                for i in range(0, n_c, SCORE_BATCH_SIZE):
                    end = min(i + SCORE_BATCH_SIZE, n_c)
                    batch_idx = X_c_indices[i:end]
                    X_batch = X[batch_idx]
                    
                    dists_m_batch = self._dist_func(X_batch, other_protos)
                    k_eff_m = min(self.k_protos, len(other_protos))
                    
                    nearest_idx_m = np.argpartition(dists_m_batch, k_eff_m - 1, axis=1)[:, :k_eff_m]
                    nearest_dists_m = np.take_along_axis(dists_m_batch, nearest_idx_m, axis=1)
                    chosen_m_batch = other_protos[nearest_idx_m]
                    
                    diffs_batch = np.abs(X_batch[:, np.newaxis, :] - chosen_m_batch)
                    
                    if self.use_idw: # AHORA USA SOFTMAX
                        weights = self._compute_softmax_weights(nearest_dists_m)
                        mean_diffs_batch = np.sum(diffs_batch * weights, axis=1)
                    else:
                        mean_diffs_batch = np.mean(diffs_batch, axis=1)
                    
                    sum_mean_diffs_m += np.sum(mean_diffs_batch, axis=0)

                weight = p_classes[other_c] / prob_not_c
                scores += (sum_mean_diffs_m / n_c) * weight * p_classes[c]
                gc.collect()
        
        self.feature_importances_ = scores
        return self

    def transform(self, X):
        if self.feature_importances_ is None:
             raise ValueError("Fit the model first.")
        idx = np.argsort(self.feature_importances_)[::-1][:self.n_features_to_select]
        return X[:, idx]

    def rank(self):
        if self.feature_importances_ is None:
             raise ValueError("Fit the model first.")
        return np.argsort(-self.feature_importances_)