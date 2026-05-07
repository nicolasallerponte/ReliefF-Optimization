"""
Métricas de evaluación compartidas para todos los experimentos del TFG.

Centralizar aquí evita inconsistencias entre scripts.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import wilcoxon, friedmanchisquare, rankdata


def precision_top_k(pesos: np.ndarray, relevantes: list[int], k: int) -> float:
    """
    Proporción de features relevantes que aparecen entre los k primeros.

    pesos     : array de importancias (mayor = más relevante)
    relevantes: índices de features que son realmente relevantes
    k         : número de features a considerar en el top
    """
    if not relevantes:
        return float('nan')
    ranking = np.argsort(-pesos)[:k]
    encontrados = sum(1 for r in relevantes if r in ranking)
    return encontrados / len(relevantes)


def ranking_posicional(pesos: np.ndarray, relevantes: list[int]) -> float:
    """
    Posición media (1-based) de las features relevantes en el ranking.
    Menor es mejor.
    """
    if not relevantes:
        return float('nan')
    orden = np.argsort(-pesos)
    posiciones = [int(np.where(orden == r)[0][0]) + 1 for r in relevantes]
    return float(np.mean(posiciones))


def intervalo_confianza_bootstrap(
    valores: np.ndarray,
    nivel: float = 0.95,
    n_muestras: int = 2000,
    semilla: int = 0,
) -> tuple[float, float]:
    """
    IC bootstrap para la media.
    Devuelve (limite_inferior, limite_superior).
    """
    rng = np.random.default_rng(semilla)
    medias = np.array([
        rng.choice(valores, size=len(valores), replace=True).mean()
        for _ in range(n_muestras)
    ])
    alpha = (1 - nivel) / 2
    return (float(np.quantile(medias, alpha)),
            float(np.quantile(medias, 1 - alpha)))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Tamaño de efecto d de Cohen entre dos muestras.
    Valores orientativos: pequeño=0.2, medio=0.5, grande=0.8.
    """
    n_a, n_b = len(a), len(b)
    var_pooled = ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
    return float((np.mean(a) - np.mean(b)) / np.sqrt(var_pooled + 1e-12))


def tabla_resumen(
    datos: dict[str, np.ndarray],
    ic: bool = True,
) -> pd.DataFrame:
    """
    Construye tabla resumen a partir de un dict {algoritmo: array_de_scores}.

    Columnas: algoritmo, media, std, ic_95_inf, ic_95_sup
    """
    filas = []
    for nombre, valores in datos.items():
        arr = np.array(valores, dtype=float)
        fila = {
            'algoritmo': nombre,
            'media':     float(np.mean(arr)),
            'std':       float(np.std(arr, ddof=1)),
            'min':       float(np.min(arr)),
            'max':       float(np.max(arr)),
        }
        if ic:
            inf, sup = intervalo_confianza_bootstrap(arr)
            fila['ic_95_inf'] = inf
            fila['ic_95_sup'] = sup
        filas.append(fila)
    return pd.DataFrame(filas)


def tests_significancia(
    datos: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Ejecuta Wilcoxon pareado (todos los pares) y Friedman (global).

    Incluye p-valor y tamaño de efecto d de Cohen.
    Devuelve DataFrame con los resultados.
    """
    nombres  = list(datos.keys())
    filas = []

    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            a_nombre, b_nombre = nombres[i], nombres[j]
            a, b = np.array(datos[a_nombre]), np.array(datos[b_nombre])
            try:
                stat, p = wilcoxon(a, b)
            except Exception:
                stat, p = float('nan'), float('nan')
            filas.append({
                'comparacion':  f'{a_nombre} vs {b_nombre}',
                'test':         'Wilcoxon',
                'estadistico':  round(stat, 4),
                'p_valor':      round(p, 6),
                'significativo': p < 0.05,
                'cohen_d':      round(cohen_d(a, b), 4),
            })

    if len(nombres) >= 3:
        try:
            stat_f, p_f = friedmanchisquare(*[datos[n] for n in nombres])
        except Exception:
            stat_f, p_f = float('nan'), float('nan')
        filas.append({
            'comparacion':  ' vs '.join(nombres),
            'test':         'Friedman',
            'estadistico':  round(stat_f, 4),
            'p_valor':      round(p_f, 6),
            'significativo': p_f < 0.05,
            'cohen_d':      float('nan'),
        })

    df = pd.DataFrame(filas)

    # Corrección de Bonferroni para comparaciones múltiples
    n_tests = (df['test'] == 'Wilcoxon').sum()
    if n_tests > 1:
        df['p_bonferroni'] = df['p_valor'].apply(
            lambda p: min(p * n_tests, 1.0) if not np.isnan(p) else float('nan')
        )
        df['sig_bonferroni'] = df['p_bonferroni'] < 0.05
    return df


def nemenyi_post_hoc(datos: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Test de Nemenyi post-hoc tras el test de Friedman.

    Solo es aplicable cuando Friedman es significativo.
    Implementación basada en la diferencia crítica (CD) con α=0.05.
    """
    nombres = list(datos.keys())
    k = len(nombres)
    n = len(next(iter(datos.values())))

    matriz = np.array([datos[n_] for n_ in nombres], dtype=float)
    rangos = np.apply_along_axis(rankdata, 0, matriz)
    rangos_medios = rangos.mean(axis=1)

    _q_tabla = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
                6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q_alpha = _q_tabla.get(k, 3.164)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    filas = []
    for i, j in combinations(range(k), 2):
        diferencia = abs(rangos_medios[i] - rangos_medios[j])
        filas.append({
            'comparacion':    f'{nombres[i]} vs {nombres[j]}',
            'diff_rangos':    round(diferencia, 4),
            'cd_005':         round(cd, 4),
            'significativo':  diferencia > cd,
        })

    return pd.DataFrame(filas)


def kuncheva_index(
    subsets: list[list[int]],
    n_total: int,
    k: int,
) -> float:
    """
    Índice de Consistencia de Kuncheva (KCI): estabilidad de la selección.

    Rango: [-1, 1], donde 1 = selección idéntica en todas las ejecuciones,
    0 = selección aleatoria.
    """
    if len(subsets) < 2 or k == 0 or k >= n_total:
        return float('nan')

    kci_pares = []
    for a, b in combinations(subsets, 2):
        interseccion = len(set(a) & set(b))
        esperado = k ** 2 / n_total
        denominador = k - esperado
        if abs(denominador) < 1e-9:
            kci_pares.append(1.0 if interseccion == k else 0.0)
        else:
            kci_pares.append((interseccion - esperado) / denominador)

    return float(np.mean(kci_pares))


def estrella_significancia(p: float) -> str:
    """Convierte un p-valor en estrellas de significancia estándar."""
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'
