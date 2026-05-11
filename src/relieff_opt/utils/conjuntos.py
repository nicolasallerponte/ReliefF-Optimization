"""
Fuente única de verdad para todos los datasets del TFG.

Cada dataset expone:
  - (X, y) reproducibles dado una semilla
  - features_relevantes: índices con ground truth (solo datasets sintéticos)
  - descripción breve
"""

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_classification,
    make_moons,
    make_circles,
)

# ---------------------------------------------------------------------------
# Catálogo de datasets
# ---------------------------------------------------------------------------

CATALOGO = {
    # --- Datasets reales (sin ground truth de features) ---
    'Breast_Cancer': {
        'descripcion': 'UCI Breast Cancer Wisconsin, 30 features, binario',
        'origen':      'sklearn',
        'relevantes':  None,
    },
    'Wine': {
        'descripcion': 'UCI Wine, 13 features, binario (clase > 0)',
        'origen':      'sklearn',
        'relevantes':  None,
    },
    'Digits': {
        'descripcion': 'NIST Digits, 64 features, binario (dígito >= 5)',
        'origen':      'sklearn',
        'relevantes':  None,
    },

    # --- Datasets sintéticos (con ground truth) ---
    'Corral': {
        'descripcion': 'Interacción multiplicativa de 6 features entre 20, 800 muestras',
        'origen':      'sintético',
        'relevantes':  [0, 1, 2, 3, 4, 5],
    },
    'CorrAL100': {
        'descripcion': 'Lógica AND/OR con 4 features relevantes entre 100, 1600 muestras',
        'origen':      'sintético',
        'relevantes':  [0, 1, 2, 3],          # f5 es correlacionada (no relevante pura)
        'correlacionada': [5],
    },
    'XOR': {
        'descripcion': 'Regla XOR entre 2 features entre 10, 1000 muestras',
        'origen':      'sintético',
        'relevantes':  [0, 1],
    },
    'Moons': {
        'descripcion': 'make_moons con ruido=0.3, 1000 muestras',
        'origen':      'sintético',
        'relevantes':  [0, 1],
    },
    'Circles': {
        'descripcion': 'make_circles con ruido=0.2, 1000 muestras',
        'origen':      'sintético',
        'relevantes':  [0, 1],
    },
    'AltaDim': {
        'descripcion': '100 features, 10 informativas, 500 muestras',
        'origen':      'sintético',
        'relevantes':  list(range(10)),
    },
    'Desbalanceado': {
        'descripcion': '20 features, 30% clase minoritaria, 1000 muestras',
        'origen':      'sintético',
        'relevantes':  list(range(15)),
    },
    'Ruidoso': {
        'descripcion': '25 features, 20% etiquetas corruptas, 800 muestras',
        'origen':      'sintético',
        'relevantes':  list(range(10)),
    },
    'Grande': {
        'descripcion': '30 features, 20 informativas, 5000 muestras',
        'origen':      'sintético',
        'relevantes':  list(range(20)),
    },

    # --- Datasets reales a gran escala (no incluidos en obtener_todos) ---
    'CoverType_10k': {
        'descripcion': 'Forest Cover Type (UCI/sklearn), clase 1 vs resto, 10000 muestras, 54 features',
        'origen':      'sklearn-fetch',
        'relevantes':  None,
    },
    'MNIST_10k': {
        'descripcion': 'MNIST (LeCun), dígito 0 vs resto, 10000 muestras, 784 features',
        'origen':      'openml',
        'relevantes':  None,
    },
}


def obtener_dataset(nombre: str, semilla: int = 42):
    """
    Devuelve (X, y, features_relevantes) para el dataset solicitado.

    X e y son arrays numpy float32/int32.
    features_relevantes es una lista de índices o None si no hay ground truth.
    """
    if nombre not in CATALOGO:
        raise ValueError(f"Dataset '{nombre}' desconocido. Disponibles: {list(CATALOGO)}")

    meta = CATALOGO[nombre]
    relevantes = meta.get('relevantes')
    rng = np.random.RandomState(semilla)

    # ------------------------------------------------------------------
    # Datasets reales
    # ------------------------------------------------------------------
    if nombre == 'Breast_Cancer':
        datos = load_breast_cancer()
        X, y = datos.data.astype(np.float32), datos.target.astype(np.int32)

    elif nombre == 'Wine':
        datos = load_wine()
        X = datos.data.astype(np.float32)
        y = (datos.target > 0).astype(np.int32)

    elif nombre == 'Digits':
        datos = load_digits()
        X = datos.data.astype(np.float32)
        y = (datos.target >= 5).astype(np.int32)

    # ------------------------------------------------------------------
    # Datasets sintéticos
    # ------------------------------------------------------------------
    elif nombre == 'Corral':
        X = rng.randn(800, 20).astype(np.float32)
        y = (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] - X[:, 4] * X[:, 5] > 0).astype(np.int32)

    elif nombre == 'CorrAL100':
        X, y = _generar_corral100(semilla=semilla, n_replicas=50)

    elif nombre == 'XOR':
        X = rng.randn(1000, 10).astype(np.float32)
        y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(np.int32)

    elif nombre == 'Moons':
        X, y = make_moons(n_samples=1000, noise=0.3, random_state=semilla)
        X = X.astype(np.float32)
        y = y.astype(np.int32)

    elif nombre == 'Circles':
        X, y = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=semilla)
        X = X.astype(np.float32)
        y = y.astype(np.int32)

    elif nombre == 'AltaDim':
        X, y = make_classification(
            n_samples=500, n_features=100, n_informative=10,
            n_redundant=20, n_repeated=10, n_clusters_per_class=2,
            random_state=semilla,
        )
        X = X.astype(np.float32)
        y = y.astype(np.int32)

    elif nombre == 'Desbalanceado':
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            weights=[0.7, 0.3], flip_y=0.1, random_state=semilla,
        )
        X = X.astype(np.float32)
        y = y.astype(np.int32)

    elif nombre == 'Ruidoso':
        X, y = make_classification(
            n_samples=800, n_features=25, n_informative=10,
            n_redundant=5, flip_y=0.2, random_state=semilla,
        )
        X = X.astype(np.float32)
        y = y.astype(np.int32)

    elif nombre == 'Grande':
        X, y = make_classification(
            n_samples=5000, n_features=30, n_informative=20,
            n_redundant=5, random_state=semilla,
        )
        X = X.astype(np.float32)
        y = y.astype(np.int32)

    elif nombre == 'CoverType_10k':
        from sklearn.datasets import fetch_covtype
        datos = fetch_covtype(as_frame=False)
        X_full = datos.data.astype(np.float32)
        y_full = (datos.target == 1).astype(np.int32)   # clase 1 vs resto (binario)
        rng_sel = np.random.RandomState(semilla)
        idx = rng_sel.choice(len(X_full), 10000, replace=False)
        X, y = X_full[idx], y_full[idx]

    elif nombre == 'MNIST_10k':
        from sklearn.datasets import fetch_openml
        datos = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X_full = datos.data.astype(np.float32)
        y_full = (datos.target.astype(int) == 0).astype(np.int32)  # dígito 0 vs resto
        rng_sel = np.random.RandomState(semilla)
        idx = rng_sel.choice(len(X_full), 10000, replace=False)
        X, y = X_full[idx], y_full[idx]

    else:
        raise NotImplementedError(f"Cargador no implementado para '{nombre}'")

    return X, y, relevantes


def obtener_todos(semilla: int = 42) -> dict:
    """Devuelve un dict {nombre: (X, y, relevantes)} para los 11 datasets estándar."""
    nombres = [
        'Breast_Cancer', 'Wine', 'Digits',
        'Corral', 'CorrAL100', 'XOR', 'Moons', 'Circles',
        'AltaDim', 'Desbalanceado', 'Ruidoso', 'Grande',
    ]
    return {n: obtener_dataset(n, semilla=semilla) for n in nombres}


# ---------------------------------------------------------------------------
# Generador interno de CorrAL-100
# ---------------------------------------------------------------------------

def _generar_corral100(semilla: int = 42, n_replicas: int = 50):
    """
    CorrAL-100:
      f0..f3 : relevantes  (lógica AND/OR)
      f4     : irrelevante
      f5     : correlacionada con y al 75%
      f6..f99: ruido Bernoulli(0.5)
    """
    rng = np.random.RandomState(semilla)
    n_base = 32

    f0_base = np.array([(i >> 0) & 1 for i in range(n_base)], dtype=np.int32)
    f1_base = np.array([(i >> 1) & 1 for i in range(n_base)], dtype=np.int32)
    f2_base = np.array([(i >> 2) & 1 for i in range(n_base)], dtype=np.int32)
    f3_base = np.array([(i >> 3) & 1 for i in range(n_base)], dtype=np.int32)
    y_base  = ((f0_base & f1_base) | (f2_base & f3_base)).astype(np.int32)

    f0 = np.tile(f0_base, n_replicas)
    f1 = np.tile(f1_base, n_replicas)
    f2 = np.tile(f2_base, n_replicas)
    f3 = np.tile(f3_base, n_replicas)
    y  = np.tile(y_base,  n_replicas)
    n  = len(y)

    f4 = rng.binomial(1, 0.5, size=n).astype(np.int32)

    f5 = y.copy()
    n_flip = int(0.25 * n)
    if n_flip > 0:
        idx_flip = rng.choice(n, n_flip, replace=False)
        f5[idx_flip] = 1 - f5[idx_flip]

    ruido = rng.binomial(1, 0.5, size=(n, 93)).astype(np.int32)

    X = np.column_stack([f0, f1, f2, f3, f4, f5, ruido]).astype(np.float32)
    return X, y
