"""
Runner multi-seed estándar para todos los experimentos del TFG.

Garantiza reproducibilidad: cada experimento se ejecuta con 5 semillas
y se reportan media y desviación típica, no resultados de una sola ejecución.
"""

import logging
import time
from typing import Callable, Any

import numpy as np
import pandas as pd

SEMILLAS = [42, 123, 456, 789, 1234]

logger = logging.getLogger(__name__)


def ejecutar_multiseed(
    fn_experimento: Callable,
    semillas: list[int] = SEMILLAS,
    **kwargs,
) -> dict[str, Any]:
    """
    Ejecuta fn_experimento(semilla=s, **kwargs) para cada semilla en la lista.

    La función recibe obligatoriamente semilla=int como argumento.
    Debe devolver un dict o un valor numérico.

    Devuelve un dict con:
      - 'valores'  : lista de resultados por semilla
      - 'semillas' : lista de semillas usadas
      - 'media'    : media por columna (si los valores son dicts/DataFrames)
      - 'std'      : desviación típica por columna
    """
    valores = []
    for semilla in semillas:
        np.random.seed(semilla)
        t0 = time.time()
        resultado = fn_experimento(semilla=semilla, **kwargs)
        duracion = time.time() - t0
        logger.info("Semilla %d completada en %.1fs", semilla, duracion)
        valores.append(resultado)

    return _resumir(valores, semillas)


def _resumir(valores: list, semillas: list[int]) -> dict[str, Any]:
    """Calcula media y std a partir de la lista de resultados."""
    if not valores:
        return {'valores': [], 'semillas': semillas, 'media': None, 'std': None}

    if isinstance(valores[0], pd.DataFrame):
        media = pd.concat(valores).groupby(level=0).mean()
        std = pd.concat(valores).groupby(level=0).std()
    elif isinstance(valores[0], dict):
        claves = valores[0].keys()
        media = {k: float(np.mean([v[k] for v in valores])) for k in claves}
        std   = {k: float(np.std( [v[k] for v in valores])) for k in claves}
    else:
        arr = np.array(valores, dtype=float)
        media = float(np.mean(arr))
        std   = float(np.std(arr))

    return {
        'valores':  valores,
        'semillas': semillas,
        'media':    media,
        'std':      std,
    }


def añadir_columnas_resumen(df: pd.DataFrame, col_valor: str) -> pd.DataFrame:
    """
    Dada una columna con valores numéricos por semilla agrupados,
    añade columnas 'media' y 'std' al DataFrame.

    Útil cuando un experimento devuelve un DataFrame por semilla
    y se concatenan antes de llamar a esta función.
    """
    media = df.groupby(df.index)[col_valor].transform('mean')
    std   = df.groupby(df.index)[col_valor].transform('std')
    df = df.copy()
    df[f'{col_valor}_media'] = media
    df[f'{col_valor}_std']   = std
    return df


def guardar_tabla(
    df: pd.DataFrame,
    nombre: str,
    subcarpeta: str = 'tablas',
    latex: bool = True,
) -> None:
    """
    Guarda un DataFrame en results/{subcarpeta}/{nombre}.csv.

    Si latex=True también exporta results/{subcarpeta}/{nombre}.tex con formato
    publication-ready (booktabs, caption vacío para completar en la memoria).

    Para mantener las tablas ordenadas por experimento, pasa subcarpeta como
    'tablas/01_busqueda_hiperparametros', 'tablas/02_linea_base', etc.
    """
    from pathlib import Path
    raiz = Path(__file__).resolve().parents[3]   # raíz del proyecto (TFG/)
    directorio = raiz / 'results' / subcarpeta
    directorio.mkdir(parents=True, exist_ok=True)

    ruta_csv = directorio / f'{nombre}.csv'
    df.to_csv(ruta_csv, index=False)
    logger.info("Tabla guardada: %s", ruta_csv)

    if latex:
        ruta_tex = directorio / f'{nombre}.tex'
        try:
            tex = df.to_latex(
                index=False,
                float_format='%.4f',
                escape=True,
                column_format='l' + 'r' * (len(df.columns) - 1),
            )
            cabecera = (
                "\\begin{table}[h]\n"
                "  \\centering\n"
                "  \\caption{}\n"
                "  \\label{tab:" + nombre.replace('/', '_') + "}\n"
            )
            pie = "\\end{table}\n"
            with open(ruta_tex, 'w') as f:
                f.write(cabecera + tex + pie)
        except Exception as e:
            logger.warning("No se pudo exportar LaTeX: %s", e)
