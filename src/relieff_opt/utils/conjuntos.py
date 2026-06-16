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

    # --- Datasets reales masivos (validación de escalabilidad al límite) ---
    'CoverType': {
        'descripcion': 'Forest Cover Type completo, clase 1 vs resto, 581012 muestras, 54 features',
        'origen':      'sklearn-fetch',
        'relevantes':  None,
    },
    'SUSY': {
        'descripcion': 'SUSY (física de partículas), binario, ~5M muestras, 18 features',
        'origen':      'uci',
        'relevantes':  None,
    },
    'HIGGS': {
        'descripcion': 'HIGGS (física de partículas), binario, ~11M muestras, 28 features',
        'origen':      'openml',
        'relevantes':  None,
    },
    'KDDCup99': {
        'descripcion': 'KDD Cup 99 (intrusión de red), normal vs ataque, ~4.9M muestras, 41 features (121 tras one-hot)',
        'origen':      'openml',
        'relevantes':  None,
    },
    'HEPMASS': {
        'descripcion': 'HEPMASS (física de partículas), binario, ~10.5M muestras, 28 features',
        'origen':      'uci',
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

    elif nombre == 'CoverType':
        from sklearn.datasets import fetch_covtype
        datos = fetch_covtype(as_frame=False)
        X = datos.data.astype(np.float32)
        y = (datos.target == 1).astype(np.int32)   # clase 1 vs resto (binario)

    elif nombre == 'SUSY':
        # OpenML no tiene SUSY; se descarga de UCI #279 (5M x 18) y se cachea.
        # CSV.gz sin cabecera: col 0 = etiqueta (0/1), cols 1-18 = features.
        import urllib.request
        import pandas as pd
        from pathlib import Path
        from sklearn.datasets import get_data_home
        cache = Path(get_data_home()) / 'susy'
        cache.mkdir(parents=True, exist_ok=True)
        gz = cache / 'SUSY.csv.gz'
        if not gz.exists():
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz'
            tmp = gz.with_suffix('.part')
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(gz)   # rename solo si la descarga completó (evita caché corrupta)
        datos = pd.read_csv(gz, header=None, dtype=np.float32)
        y = datos.iloc[:, 0].to_numpy().astype(np.int32)
        X = datos.iloc[:, 1:].to_numpy().astype(np.float32)
        del datos

    elif nombre == 'HIGGS':
        from sklearn.datasets import fetch_openml
        datos = fetch_openml(data_id=45570, as_frame=False, parser='auto')  # HIGGS completo: 11M x 28
        X = np.nan_to_num(datos.data.astype(np.float32))
        y = np.asarray(datos.target).astype(float).astype(np.int32)

    elif nombre == 'KDDCup99':
        # KDD Cup 99 completo (4.9M x 41) desde OpenML (data_id=42746).
        # Objetivo binario: normal (0) vs ataque (1). Las 3 columnas categóricas
        # (protocol_type, service, flag) se codifican one-hot -> 121 features.
        import pandas as pd
        from sklearn.datasets import fetch_openml
        datos = fetch_openml(data_id=42746, as_frame=True, parser='auto')
        df = datos.frame
        target_col = datos.target.name
        y = (~df[target_col].astype(str).str.contains('normal')).astype(np.int32).to_numpy()
        X_df = pd.get_dummies(df.drop(columns=[target_col]), dummy_na=False)
        del df
        X = X_df.to_numpy(dtype=np.float32)
        del X_df

    elif nombre == 'HEPMASS':
        # HEPMASS completo (10.5M x 28) desde UCI #347: all_train.csv.gz (7M) +
        # all_test.csv.gz (3.5M), concatenados y cacheados. Col 0 = '# label'
        # (0/1), cols 1-28 = features (f0..f26 + mass). Binario y balanceado.
        import urllib.request
        import pandas as pd
        from pathlib import Path
        from sklearn.datasets import get_data_home
        cache = Path(get_data_home()) / 'hepmass'
        cache.mkdir(parents=True, exist_ok=True)
        base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00347/'
        partes = []
        for fichero in ('all_train.csv.gz', 'all_test.csv.gz'):
            gz = cache / fichero
            if not gz.exists():
                tmp = gz.with_suffix('.part')
                urllib.request.urlretrieve(base + fichero, tmp)
                tmp.rename(gz)   # rename solo si completó (evita caché corrupta)
            partes.append(pd.read_csv(gz, dtype=np.float32))
        datos = pd.concat(partes, ignore_index=True)
        del partes
        y = datos.iloc[:, 0].to_numpy().astype(np.int32)
        X = datos.iloc[:, 1:].to_numpy().astype(np.float32)
        del datos

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
