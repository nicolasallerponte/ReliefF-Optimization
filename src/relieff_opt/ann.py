"""
ANN-ReliefF: ReliefF con índice de vecinos aproximados HNSW.

Selección de características mediante ReliefF acelerado con hnswlib.
Usa Numba JIT para el núcleo de cálculo de puntuaciones.

Parámetros principales (valores por defecto óptimos según grid search):
  n_neighbors      : 15   - ventana de vecindario base
  iter_ratio       : 'auto' - fracción de muestras usadas como query (adaptativo)
  metric           : 'euclidean'
  M / ef_construction: parámetros HNSW
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import hnswlib
from numba import njit, prange


# ---------------------------------------------------------------------------
# Núcleo JIT - se compila la primera vez que se llama
# ---------------------------------------------------------------------------

@njit(fastmath=True, parallel=True)
def _calcular_puntuaciones(
    X_muestra, X_completo, indices_vecinos, etiquetas_vecinos,
    etiquetas_muestra, mascara_discreta, p_c, k,
):
    n_iter = X_muestra.shape[0]
    n_feat = X_muestra.shape[1]
    matriz = np.zeros((n_iter, n_feat), dtype=np.float32)

    for i in prange(n_iter):
        y_i = etiquetas_muestra[i]
        fila_i = X_muestra[i]
        hits_diff  = np.zeros(n_feat, dtype=np.float32)
        miss_diff  = np.zeros((len(p_c), n_feat), dtype=np.float32)
        cnt_hits   = 0.0
        cnt_misses = np.zeros(len(p_c), dtype=np.float32)

        for idx_vec in indices_vecinos[i]:
            if idx_vec == -1:
                continue
            y_vec   = etiquetas_vecinos[idx_vec]
            fila_vec = X_completo[idx_vec]
            diff = np.abs(fila_i - fila_vec)

            for f in range(n_feat):
                if mascara_discreta[f]:
                    diff[f] = 1.0 if fila_i[f] != fila_vec[f] else 0.0

            if y_vec == y_i:
                hits_diff += diff
                cnt_hits  += 1
            else:
                miss_diff[y_vec] += diff
                cnt_misses[y_vec] += 1

        puntuacion = np.zeros(n_feat, dtype=np.float32)
        if cnt_hits > 0:
            puntuacion -= hits_diff / cnt_hits

        denom = 1.0 - p_c[y_i]
        if denom == 0:
            denom = 1.0

        for c in range(len(p_c)):
            if c == y_i:
                continue
            if cnt_misses[c] > 0:
                peso = p_c[c] / denom
                puntuacion += (miss_diff[c] / cnt_misses[c]) * peso

        matriz[i] = puntuacion

    return np.sum(matriz, axis=0)


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class ANN(BaseEstimator, TransformerMixin):
    """
    ReliefF con Vecinos Aproximados (HNSW) y Numba JIT.

    La función adaptativa de n_vecinos sigue la fórmula:
        k = 15 / (1 + ratio_features * 5.0)
    donde ratio_features = n_features / n_samples, con clipping [5, 15].

    Esto permite un vecindario amplio en datasets con pocas features relativas
    y más local cuando las features son muchas respecto a las muestras.
    """

    def __init__(
        self,
        n_neighbors=15,
        n_features_to_select=10,
        discrete_threshold=10,
        iter_ratio='auto',
        n_jobs=1,
        random_state=42,
        convergence_threshold=1e-4,
        min_iterations=5,
        metric='euclidean',
        M=16,
        ef_construction=200,
        ef_search=-1,
    ):
        self.n_neighbors = n_neighbors
        self.n_features_to_select = n_features_to_select
        self.discrete_threshold = discrete_threshold
        self.iter_ratio = iter_ratio
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.convergence_threshold = convergence_threshold
        self.min_iterations = min_iterations
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.feature_importances_ = None

    # ------------------------------------------------------------------
    # Métodos internos de adaptación
    # ------------------------------------------------------------------

    def _analizar_dataset(self, X, y):
        n, d = X.shape
        return {
            'n_samples':  n,
            'n_features': d,
            'n_classes':  len(np.unique(y)),
            'ratio':      d / n,
        }

    def _k_adaptativo(self, ratio, n_clases):
        """Fórmula continua: k decrece suavemente con el ratio features/muestras."""
        k = 15.0 / (1 + ratio * 5.0)
        k = int(np.clip(k, 5, 15))
        if n_clases > 5:
            k = min(k + 2, 15)
        return k

    def _params_adaptativos(self, X, y):
        info = self._analizar_dataset(X, y)
        n = info['n_samples']

        k = self._k_adaptativo(info['ratio'], info['n_classes']) if self.n_neighbors == 15 else self.n_neighbors

        if self.iter_ratio == 'auto':
            if n < 2000:
                ratio_iter = 1.0
            elif n < 5000:
                ratio_iter = 0.8
            elif n < 20000:
                ratio_iter = 0.5
            else:
                ratio_iter = 0.3
        else:
            ratio_iter = self.iter_ratio

        usar_hnsw = n >= 10000
        if usar_hnsw:
            if n >= 100000:
                M_final, ef_final = 32, 200
            elif n >= 50000:
                M_final, ef_final = 24, 150
            else:
                M_final, ef_final = 16, 100
        else:
            M_final, ef_final = 16, 200

        return {
            'k':         k,
            'ratio_iter': ratio_iter,
            'usar_hnsw': usar_hnsw,
            'M':         M_final,
            'ef':        ef_final,
            'info':      info,
        }

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        n, d = X.shape

        p = self._params_adaptativos(X, y)
        k          = p['k']
        ratio_iter = p['ratio_iter']
        usar_hnsw  = p['usar_hnsw']
        M_param    = p['M']
        ef_param   = p['ef']

        mascara_discreta = np.array([
            len(np.unique(X[:, i])) <= self.discrete_threshold
            for i in range(d)
        ], dtype=bool)

        if np.any(~mascara_discreta):
            scaler = MinMaxScaler()
            X[:, ~mascara_discreta] = scaler.fit_transform(X[:, ~mascara_discreta])

        n_iter = max(int(n * ratio_iter), k + 2)
        np.random.seed(self.random_state)
        idx_query = (
            np.random.choice(n, n_iter, replace=False) if ratio_iter < 1.0
            else np.arange(n)
        )

        X_query  = X[idx_query]
        y_query  = y[idx_query]
        clases   = np.unique(y)
        max_vecs = min(len(clases) * k, n - 1)
        vecinos  = np.full((n_iter, max_vecs), -1, dtype=np.int32)

        espacio_hnsw = 'l2' if self.metric in ('euclidean', 'l2') else 'l1'
        sklearn_p    = 2    if self.metric in ('euclidean', 'l2') else 1

        indices_por_clase = {}
        for c in clases:
            idx_c = np.where(y == c)[0]
            if len(idx_c) <= 1:
                continue

            if usar_hnsw:
                try:
                    indice = hnswlib.Index(space=espacio_hnsw, dim=d)
                    indice.init_index(
                        max_elements=len(idx_c),
                        ef_construction=ef_param,
                        M=M_param,
                    )
                    indice.add_items(X[idx_c], idx_c)
                    ef_busqueda = self.ef_search if self.ef_search > 0 else max(k * 2, 50)
                    indice.set_ef(ef_busqueda)
                    indices_por_clase[c] = ('hnsw', indice, idx_c)
                    continue
                except Exception:
                    usar_hnsw = False

            nn = NearestNeighbors(
                n_neighbors=min(k + 1, len(idx_c)),
                algorithm='brute',
                p=sklearn_p,
                n_jobs=1,
            )
            nn.fit(X[idx_c])
            indices_por_clase[c] = ('sklearn', nn, idx_c)

        for i, q_idx in enumerate(idx_query):
            y_q = y[q_idx]
            pos = 0
            for c in clases:
                if c not in indices_por_clase:
                    continue
                tipo, obj, idx_reales = indices_por_clase[c]
                k_buscar = min(k + 1 if c == y_q else k, len(idx_reales))

                if tipo == 'hnsw':
                    etiquetas, _ = obj.knn_query(X[q_idx].reshape(1, -1), k=k_buscar)
                    encontrados = etiquetas[0]
                else:
                    _, idx_rel = obj.kneighbors(X[q_idx].reshape(1, -1), n_neighbors=k_buscar)
                    encontrados = idx_reales[idx_rel[0]]

                for f_idx in encontrados:
                    if f_idx != q_idx and pos < max_vecs:
                        vecinos[i, pos] = f_idx
                        pos += 1
                        if c == y_q and pos >= k:
                            break

        self.feature_importances_ = _calcular_puntuaciones(
            X_query, X, vecinos, y, y_query,
            mascara_discreta,
            (np.bincount(y) / n).astype(np.float32),
            k,
        )

        if np.any(self.feature_importances_ != 0):
            self.feature_importances_ /= n_iter

        return self

    def rank(self):
        """Devuelve los índices de features de mayor a menor importancia."""
        if self.feature_importances_ is None:
            raise ValueError("Primero llama a fit().")
        return np.argsort(-self.feature_importances_)

    def transform(self, X, n_features_to_select=None):
        """Selecciona las n_features_to_select más importantes."""
        if self.feature_importances_ is None:
            raise ValueError("Primero llama a fit().")
        n = n_features_to_select or self.n_features_to_select or X.shape[1] // 2
        return X[:, self.rank()[:n]]
