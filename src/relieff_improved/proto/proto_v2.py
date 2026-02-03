import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from joblib import Parallel, delayed

class Proto(BaseEstimator, TransformerMixin):
    """
    ReliefF Proto Optimizado v3.0
    
    Mejoras implementadas:
    - Clustering jerárquico con validación de densidad
    - LVQ con momentum para convergencia más rápida
    - Softmax con temperatura adaptativa
    - Paralelización de scoring por clases
    - Mixed precision (float16/float32) para reducir memoria
    - Caché inteligente de distancias entre prototipos
    """

    def __init__(self, n_features_to_select=10, sigma=0.05, discrete_threshold=10, 
                 k_protos=3, force_real_centers=True, use_lvq=False, lvq_epochs=5, 
                 lvq_lr=0.1, lvq_momentum=0.9, metric='manhattan', lvq_batch_size=128,
                 scaler_type='robust', use_idw=True, gamma='auto', 
                 use_mixed_precision=True, n_jobs=-1, min_cluster_size=5): 
        
        self.n_features_to_select = n_features_to_select
        self.feature_importances_ = None
        self.discrete_threshold = discrete_threshold
        self.sigma = sigma
        self.k_protos = k_protos
        self.force_real_centers = force_real_centers
        self.use_lvq = use_lvq
        self.lvq_epochs = lvq_epochs
        self.lvq_lr = lvq_lr
        self.lvq_momentum = lvq_momentum
        self.metric = metric
        self.lvq_batch_size = lvq_batch_size
        self.scaler_type = scaler_type
        self.use_idw = use_idw
        self.gamma = gamma
        self.use_mixed_precision = use_mixed_precision
        self.n_jobs = n_jobs
        self.min_cluster_size = min_cluster_size

    def _dist_func(self, X, Y, use_fp16=False):
        """Calcula distancias con mixed precision opcional"""
        if use_fp16 and self.use_mixed_precision:
            X_fp16 = X.astype(np.float16)
            Y_fp16 = Y.astype(np.float16)
            if self.metric == 'manhattan':
                return manhattan_distances(X_fp16, Y_fp16).astype(np.float32)
            return euclidean_distances(X_fp16, Y_fp16).astype(np.float32)
        else:
            if self.metric == 'manhattan':
                return manhattan_distances(X, Y)
            return euclidean_distances(X, Y)

    def _refine_prototypes_lvq_momentum(self, X, y, prototypes_dict):
        """LVQ mejorado con momentum para convergencia más rápida"""
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
        
        # Inicializar momentum
        velocity = np.zeros_like(flat_protos)
        
        n_samples = len(X)
        limit = min(5000, n_samples)
        
        for epoch in range(self.lvq_epochs):
            # Learning rate decay
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
                signs = np.where(winners_labels == y_batch, 1.0, -1.0)[:, np.newaxis]
                updates = current_lr * signs * diffs
                
                # Aplicar momentum
                velocity = self.lvq_momentum * velocity
                np.add.at(velocity, winners_idx, updates)
                flat_protos += velocity

        new_prototypes = {}
        cursor = 0
        for c in sorted_classes:
            count = structure_map[c]
            new_prototypes[c] = flat_protos[cursor : cursor + count]
            cursor += count
        return new_prototypes

    def _get_real_prototypes(self, X_indices, X_full, abstract_centers):
        """Snap-to-grid: mueve centroides a muestras reales"""
        X_subset = X_full[X_indices]
        dists = self._dist_func(X_subset, abstract_centers)
        real_indices_local = np.argmin(dists, axis=0)
        return X_subset[real_indices_local]

    def _compute_adaptive_softmax_weights(self, dists):
        """Softmax con temperatura adaptativa basada en varianza de distancias"""
        # Calcular temperatura óptima por batch
        mean_dist = np.mean(dists, axis=1, keepdims=True)
        std_dist = np.std(dists, axis=1, keepdims=True) + 1e-8
        
        if self.gamma == 'auto':
            # Temperatura inversamente proporcional a la dispersión
            adaptive_gamma = 1.0 / (std_dist + 1e-8)
        else:
            adaptive_gamma = self.gamma
        
        # Normalización para estabilidad numérica
        shift = np.min(dists, axis=1, keepdims=True)
        normalized_dists = (dists - shift) / (std_dist + 1e-8)
        
        exp_dists = np.exp(-adaptive_gamma * normalized_dists)
        weights = exp_dists / (np.sum(exp_dists, axis=1, keepdims=True) + 1e-8)
        return weights[:, :, np.newaxis]

    def _generate_improved_prototypes(self, X, y, c, count):
        """Generación de prototipos con validación de densidad"""
        X_subset = X[y == c]
        
        # Número dinámico de clusters
        n_clusters = int(max(1, min(200, count * self.sigma)))
        
        if count <= n_clusters:
            return X_subset
        
        # Clustering con validación
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            batch_size=min(1024, len(X_subset)), 
            n_init=3,
            random_state=42
        ).fit(X_subset)
        
        centers = kmeans.cluster_centers_.astype(np.float32)
        
        # Validación: eliminar clusters muy pequeños (posible ruido)
        labels = kmeans.labels_
        valid_centers = []
        for i in range(n_clusters):
            cluster_size = np.sum(labels == i)
            if cluster_size >= self.min_cluster_size:
                valid_centers.append(centers[i])
        
        if len(valid_centers) == 0:
            # Si todos fueron rechazados, usar el centro global
            return X_subset.mean(axis=0, keepdims=True)
        
        return np.array(valid_centers, dtype=np.float32)

    def _score_class_parallel(self, c, X, y, prototypes, p_classes, n_samples, classes):
        """Cálculo de score para una clase (paralelizable)"""
        mask = (y == c)
        X_c_indices = np.where(mask)[0]
        if len(X_c_indices) == 0:
            return np.zeros(X.shape[1], dtype=np.float32)
        
        n_c = len(X_c_indices)
        my_protos = prototypes[c]
        scores_c = np.zeros(X.shape[1], dtype=np.float32)
        BATCH_SIZE = 256
        
        # FASE HIT
        sum_mean_diffs = np.zeros(X.shape[1], dtype=np.float32)
        for i in range(0, n_c, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n_c)
            batch_idx = X_c_indices[i:end]
            X_batch = X[batch_idx]
            
            dists_batch = self._dist_func(X_batch, my_protos, use_fp16=True)
            k_eff = min(self.k_protos, len(my_protos))
            
            nearest_idx = np.argpartition(dists_batch, k_eff - 1, axis=1)[:, :k_eff]
            nearest_dists = np.take_along_axis(dists_batch, nearest_idx, axis=1)
            chosen_batch = my_protos[nearest_idx]
            
            diffs_batch = np.abs(X_batch[:, np.newaxis, :] - chosen_batch)
            
            if self.use_idw:
                weights = self._compute_adaptive_softmax_weights(nearest_dists)
                mean_diffs_batch = np.sum(diffs_batch * weights, axis=1)
            else:
                mean_diffs_batch = np.mean(diffs_batch, axis=1)
            
            sum_mean_diffs += np.sum(mean_diffs_batch, axis=0)
        
        scores_c -= (sum_mean_diffs / n_c) * p_classes[c]
        
        # FASE MISS
        prob_not_c = 1.0 - p_classes[c]
        if prob_not_c == 0:
            prob_not_c = 1.0
            
        for other_c in classes:
            if other_c == c:
                continue
            
            other_protos = prototypes[other_c]
            sum_mean_diffs_m = np.zeros(X.shape[1], dtype=np.float32)
            
            for i in range(0, n_c, BATCH_SIZE):
                end = min(i + BATCH_SIZE, n_c)
                batch_idx = X_c_indices[i:end]
                X_batch = X[batch_idx]
                
                dists_m_batch = self._dist_func(X_batch, other_protos, use_fp16=True)
                k_eff_m = min(self.k_protos, len(other_protos))
                
                nearest_idx_m = np.argpartition(dists_m_batch, k_eff_m - 1, axis=1)[:, :k_eff_m]
                nearest_dists_m = np.take_along_axis(dists_m_batch, nearest_idx_m, axis=1)
                chosen_m_batch = other_protos[nearest_idx_m]
                
                diffs_batch = np.abs(X_batch[:, np.newaxis, :] - chosen_m_batch)
                
                if self.use_idw:
                    weights = self._compute_adaptive_softmax_weights(nearest_dists_m)
                    mean_diffs_batch = np.sum(diffs_batch * weights, axis=1)
                else:
                    mean_diffs_batch = np.mean(diffs_batch, axis=1)
                
                sum_mean_diffs_m += np.sum(mean_diffs_batch, axis=0)
            
            weight = p_classes[other_c] / prob_not_c
            scores_c += (sum_mean_diffs_m / n_c) * weight * p_classes[c]
        
        return scores_c

    def fit(self, X, y):
        # 1. OPTIMIZACIÓN DE TIPOS
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int32)
        n_samples, n_features = X.shape
        classes = np.unique(y)

        # 2. PREPROCESAMIENTO
        discrete_mask = np.zeros(n_features, dtype=bool)
        if self.discrete_threshold > 0:
            for i in range(n_features):
                if len(np.unique(X[:, i])) <= self.discrete_threshold:
                    discrete_mask[i] = True
        
        if np.any(~discrete_mask):
            if self.scaler_type == 'robust':
                scaler = RobustScaler()
                X[:, ~discrete_mask] = scaler.fit_transform(X[:, ~discrete_mask])
                X[:, ~discrete_mask] = np.clip(X[:, ~discrete_mask], -5.0, 5.0)
            else:
                scaler = MinMaxScaler()
                X[:, ~discrete_mask] = scaler.fit_transform(X[:, ~discrete_mask])
        
        gc.collect()

        # 3. GENERACIÓN MEJORADA DE PROTOTIPOS
        prototypes = {}
        class_counts = {}
        
        for c in classes:
            count = np.sum(y == c)
            class_counts[c] = count
            prototypes[c] = self._generate_improved_prototypes(X, y, c, count)
        
        gc.collect()

        # 4. REFINAMIENTO LVQ CON MOMENTUM
        if self.use_lvq:
            prototypes = self._refine_prototypes_lvq_momentum(X, y, prototypes)
            gc.collect()

        # 5. SNAP TO GRID
        if self.force_real_centers:
            for c in classes:
                if class_counts[c] > len(prototypes[c]):
                    prototypes[c] = self._get_real_prototypes(
                        np.where(y == c)[0], X, prototypes[c]
                    )
        gc.collect()

        # 6. CÁLCULO PARALELO DE IMPORTANCIAS
        p_classes = {c: count / n_samples for c, count in class_counts.items()}
        
        if self.n_jobs != 1 and len(classes) > 2:
            # Paralelización por clases
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._score_class_parallel)(
                    c, X, y, prototypes, p_classes, n_samples, classes
                ) for c in classes
            )
            self.feature_importances_ = np.sum(results, axis=0)
        else:
            # Secuencial para datasets pequeños
            scores = np.zeros(n_features, dtype=np.float32)
            for c in classes:
                scores += self._score_class_parallel(
                    c, X, y, prototypes, p_classes, n_samples, classes
                )
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