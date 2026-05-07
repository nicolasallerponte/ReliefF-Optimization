"""
Proto-ReliefF: ReliefF con prototipos basados en clustering (LVQ opcional).

Sustituye los vecinos exactos de ReliefF por centroides de clase generados
con MiniBatchKMeans. El número de clusters escala adaptativamente con el ratio
features/muestras mediante la fórmula sigma:
    sigma_eff = clip(0.10 * (1 + log(ratio / 0.05)), 0.10, 0.25)

Parámetros principales (valores por defecto óptimos según grid search):
  k_protos  : 10   - prototipos por clase (adaptativo)
  sigma     : 0.15 - controla el número de clusters (adaptativo si usa default)
  use_lvq   : False - LVQ empeora ligeramente en los experimentos
  metric    : 'euclidean'
  scaler_type: 'robust'
"""

import gc
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from joblib import Parallel, delayed


class Proto(BaseEstimator, TransformerMixin):
    """
    ReliefF con Prototipos y clustering adaptativo.

    El número de clusters por clase se calcula como:
        n_clusters = int(max(1, min(200, count * sigma_eff)))
    donde sigma_eff sigue una curva logarítmica respecto al ratio features/muestras.
    """

    def __init__(
        self,
        n_features_to_select=10,
        sigma=0.15,
        discrete_threshold=10,
        k_protos=10,
        force_real_centers=True,
        use_lvq=False,
        lvq_epochs=5,
        lvq_lr=0.1,
        lvq_momentum=0.9,
        metric='euclidean',
        lvq_batch_size=128,
        scaler_type='robust',
        use_idw=True,
        gamma='auto',
        use_mixed_precision=True,
        n_jobs=-1,
        min_cluster_size=5,
    ):
        self.n_features_to_select = n_features_to_select
        self.sigma = sigma
        self.discrete_threshold = discrete_threshold
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
        self.feature_importances_ = None

    # ------------------------------------------------------------------
    # Métodos internos de adaptación
    # ------------------------------------------------------------------

    def _analizar_dataset(self, X, y):
        n, d = X.shape
        clases, conteos = np.unique(y, return_counts=True)
        return {
            'n_samples':      n,
            'n_features':     d,
            'n_classes':      len(clases),
            'min_class_size': conteos.min(),
            'max_class_size': conteos.max(),
            'ratio':          d / n,
            'is_large':       n >= 5000,
            'is_imbalanced':  (conteos.max() / conteos.min()) > 3,
        }

    def _sigma_adaptativo(self, ratio):
        """
        Curva logarítmica: sigma crece lentamente con el ratio features/muestras.
        Rango fijo [0.10, 0.25] - elimina la zona de bajo rendimiento < 0.10.
        """
        base, ratio_base = 0.10, 0.05
        if ratio <= ratio_base:
            sigma_eff = base
        else:
            sigma_eff = base * (1 + np.log(ratio / ratio_base))
        return float(np.clip(sigma_eff, 0.10, 0.25))

    def _params_adaptativos(self, X, y):
        info = self._analizar_dataset(X, y)
        min_clase = info['min_class_size']
        ratio     = info['ratio']

        sigma_eff = self._sigma_adaptativo(ratio) if self.sigma == 0.15 else self.sigma

        if self.k_protos == 10:
            if min_clase < 30:
                k_eff = 3
            elif min_clase < 100:
                k_eff = 5
            elif min_clase < 300:
                k_eff = 7
            elif info['is_large']:
                k_eff = 15
            else:
                k_eff = 10
            if info['is_imbalanced']:
                k_eff = max(3, k_eff - 2)
        else:
            k_eff = self.k_protos

        if self.n_jobs == -1:
            n_jobs_eff = 1 if (info['n_samples'] < 2000 or info['n_classes'] < 3) else -1
        else:
            n_jobs_eff = self.n_jobs

        return {'k_protos': k_eff, 'sigma': sigma_eff, 'n_jobs': n_jobs_eff, 'info': info}

    # ------------------------------------------------------------------
    # Cálculo de distancias
    # ------------------------------------------------------------------

    def _distancias(self, X, Y, usar_fp16=False):
        if usar_fp16 and self.use_mixed_precision:
            X = X.astype(np.float16)
            Y = Y.astype(np.float16)
        fn = manhattan_distances if self.metric == 'manhattan' else euclidean_distances
        resultado = fn(X, Y)
        return resultado.astype(np.float32)

    # ------------------------------------------------------------------
    # Generación y refinamiento de prototipos
    # ------------------------------------------------------------------

    def _generar_prototipos(self, X, y, c, count, sigma_eff):
        X_c = X[y == c]
        n_clusters = int(max(1, min(200, count * sigma_eff)))
        if count <= n_clusters:
            return X_c
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(1024, len(X_c)),
            n_init=3,
            random_state=42,
        ).fit(X_c)
        centros = km.cluster_centers_.astype(np.float32)
        etiquetas_km = km.labels_
        validos = [
            centros[i] for i in range(n_clusters)
            if np.sum(etiquetas_km == i) >= self.min_cluster_size
        ]
        return np.array(validos, dtype=np.float32) if validos else X_c.mean(axis=0, keepdims=True)

    def _refinar_lvq(self, X, y, prototipos):
        """LVQ con momentum para refinar los centros."""
        planos, etqs_planas, mapa = [], [], {}
        for c in sorted(prototipos):
            mapa[c] = len(prototipos[c])
            for p in prototipos[c]:
                planos.append(p)
                etqs_planas.append(c)
        P = np.array(planos, dtype=np.float32)
        E = np.array(etqs_planas)
        vel = np.zeros_like(P)

        n = len(X)
        limite = min(5000, n)
        for epoca in range(self.lvq_epochs):
            lr = self.lvq_lr * (1.0 - epoca / self.lvq_epochs)
            idx = np.random.choice(n, limite, replace=False)
            for i in range(0, limite, self.lvq_batch_size):
                batch_idx = idx[i: i + self.lvq_batch_size]
                if len(batch_idx) == 0:
                    break
                X_b = X[batch_idx]
                y_b = y[batch_idx]
                dists = self._distancias(X_b, P)
                ganadores = np.argmin(dists, axis=1)
                signos = np.where(E[ganadores] == y_b, 1.0, -1.0)[:, np.newaxis]
                updates = lr * signos * (X_b - P[ganadores])
                vel = self.lvq_momentum * vel
                np.add.at(vel, ganadores, updates)
        P += vel

        nuevos = {}
        cursor = 0
        for c in sorted(prototipos):
            n_c = mapa[c]
            nuevos[c] = P[cursor: cursor + n_c]
            cursor += n_c
        return nuevos

    def _snap_to_grid(self, idx_c, X, centros_abstractos):
        """Reemplaza centros abstractos por la muestra real más cercana."""
        X_c = X[idx_c]
        dists = self._distancias(X_c, centros_abstractos)
        idx_local = np.argmin(dists, axis=0)
        return X_c[idx_local]

    # ------------------------------------------------------------------
    # Cálculo de puntuaciones por clase
    # ------------------------------------------------------------------

    def _puntuar_clase(self, c, X, y, prototipos, p_clases, clases, k_eff):
        idx_c = np.where(y == c)[0]
        n_c = len(idx_c)
        if n_c == 0:
            return np.zeros(X.shape[1], dtype=np.float32)

        mis_protos = prototipos[c]
        puntuacion = np.zeros(X.shape[1], dtype=np.float32)
        BATCH = 256

        def _pesos_softmax(dists):
            media = np.mean(dists, axis=1, keepdims=True)
            std   = np.std(dists,  axis=1, keepdims=True) + 1e-8
            gamma = 1.0 / std if self.gamma == 'auto' else self.gamma
            shift = np.min(dists, axis=1, keepdims=True)
            norm  = (dists - shift) / std
            exp   = np.exp(-gamma * norm)
            return (exp / (np.sum(exp, axis=1, keepdims=True) + 1e-8))[:, :, np.newaxis]

        # Fase HIT
        suma_hit = np.zeros(X.shape[1], dtype=np.float32)
        for i in range(0, n_c, BATCH):
            X_b = X[idx_c[i: i + BATCH]]
            dists = self._distancias(X_b, mis_protos, usar_fp16=True)
            k_real = min(k_eff, len(mis_protos))
            idx_near = np.argpartition(dists, k_real - 1, axis=1)[:, :k_real]
            dists_near = np.take_along_axis(dists, idx_near, axis=1)
            near_protos = mis_protos[idx_near]
            diffs = np.abs(X_b[:, np.newaxis, :] - near_protos)
            pesos = _pesos_softmax(dists_near) if self.use_idw else None
            diffs_prom = np.sum(diffs * pesos, axis=1) if self.use_idw else np.mean(diffs, axis=1)
            suma_hit += np.sum(diffs_prom, axis=0)
        puntuacion -= (suma_hit / n_c) * p_clases[c]

        # Fase MISS
        denom = 1.0 - p_clases[c] or 1.0
        for otro in clases:
            if otro == c:
                continue
            otros_protos = prototipos[otro]
            suma_miss = np.zeros(X.shape[1], dtype=np.float32)
            for i in range(0, n_c, BATCH):
                X_b = X[idx_c[i: i + BATCH]]
                dists = self._distancias(X_b, otros_protos, usar_fp16=True)
                k_real = min(k_eff, len(otros_protos))
                idx_near = np.argpartition(dists, k_real - 1, axis=1)[:, :k_real]
                dists_near = np.take_along_axis(dists, idx_near, axis=1)
                near_protos = otros_protos[idx_near]
                diffs = np.abs(X_b[:, np.newaxis, :] - near_protos)
                pesos = _pesos_softmax(dists_near) if self.use_idw else None
                diffs_prom = np.sum(diffs * pesos, axis=1) if self.use_idw else np.mean(diffs, axis=1)
                suma_miss += np.sum(diffs_prom, axis=0)
            peso_clase = p_clases[otro] / denom
            puntuacion += (suma_miss / n_c) * peso_clase * p_clases[c]

        return puntuacion

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int32)
        n, d = X.shape
        clases = np.unique(y)

        params = self._params_adaptativos(X, y)
        k_eff, sigma_eff, n_jobs_eff = params['k_protos'], params['sigma'], params['n_jobs']

        # Preprocesamiento
        mascara_discreta = np.array([
            len(np.unique(X[:, i])) <= self.discrete_threshold
            for i in range(d)
        ], dtype=bool)

        if np.any(~mascara_discreta):
            scaler = RobustScaler() if self.scaler_type == 'robust' else MinMaxScaler()
            X[:, ~mascara_discreta] = scaler.fit_transform(X[:, ~mascara_discreta])
            if self.scaler_type == 'robust':
                X[:, ~mascara_discreta] = np.clip(X[:, ~mascara_discreta], -5.0, 5.0)

        gc.collect()

        # Generación de prototipos
        conteos = {c: int(np.sum(y == c)) for c in clases}
        prototipos = {
            c: self._generar_prototipos(X, y, c, conteos[c], sigma_eff)
            for c in clases
        }
        gc.collect()

        if self.use_lvq:
            prototipos = self._refinar_lvq(X, y, prototipos)
            gc.collect()

        if self.force_real_centers:
            for c in clases:
                if conteos[c] > len(prototipos[c]):
                    prototipos[c] = self._snap_to_grid(np.where(y == c)[0], X, prototipos[c])
            gc.collect()

        # Scoring
        p_clases = {c: conteos[c] / n for c in clases}

        if n_jobs_eff != 1 and len(clases) > 2:
            resultados = Parallel(n_jobs=n_jobs_eff)(
                delayed(self._puntuar_clase)(c, X, y, prototipos, p_clases, clases, k_eff)
                for c in clases
            )
            self.feature_importances_ = np.sum(resultados, axis=0)
        else:
            puntuaciones = np.zeros(d, dtype=np.float32)
            for c in clases:
                puntuaciones += self._puntuar_clase(c, X, y, prototipos, p_clases, clases, k_eff)
            self.feature_importances_ = puntuaciones

        return self

    def rank(self):
        """Devuelve los índices de features de mayor a menor importancia."""
        if self.feature_importances_ is None:
            raise ValueError("Primero llama a fit().")
        return np.argsort(-self.feature_importances_)

    def transform(self, X):
        """Selecciona las n_features_to_select más importantes."""
        if self.feature_importances_ is None:
            raise ValueError("Primero llama a fit().")
        idx = self.rank()[:self.n_features_to_select]
        return X[:, idx]
