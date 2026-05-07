"""
Paleta de colores y estilos para todas las figuras del TFG.

Importar SIEMPRE desde aquí para garantizar consistencia visual entre experimentos.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Colores y estilos por algoritmo
# ---------------------------------------------------------------------------

COLORES = {
    'ANN':     '#3498db',   # azul
    'Proto':   '#e74c3c',   # rojo
    'ReliefF': '#2ecc71',   # verde (baseline)
}

MARCADORES = {
    'ANN':     'o',
    'Proto':   's',
    'ReliefF': '^',
}

ESTILOS_LINEA = {
    'ANN':     '-',
    'Proto':   '--',
    'ReliefF': ':',
}

GROSOR_LINEA = 2.0
TAMANO_MARCADOR = 6
DPI = 300
TAMANO_FIG = (7, 4.5)      # pulgadas, ratio ~16:9

# Paleta para heatmaps y gráficos adicionales
COLOR_FONDO = 'white'
COLOR_CUADRICULA = '#cccccc'
COLOR_EJES = '#333333'

# ---------------------------------------------------------------------------
# Estilo global matplotlib
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.color':         COLOR_CUADRICULA,
    'figure.dpi':         100,
    'savefig.dpi':        DPI,
})


def estilo_academico(ax):
    """Aplica estilo publication-ready a un eje matplotlib."""
    ax.set_facecolor(COLOR_FONDO)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color(COLOR_EJES)
    ax.tick_params(direction='out', length=4, width=0.8, colors=COLOR_EJES)


def nueva_figura(filas=1, columnas=1, tamano=None):
    """Crea una figura con fondo blanco y el tamaño estándar."""
    if tamano is None:
        tamano = (TAMANO_FIG[0] * columnas, TAMANO_FIG[1] * filas)
    fig, axes = plt.subplots(filas, columnas, figsize=tamano)
    fig.patch.set_facecolor(COLOR_FONDO)
    if filas == 1 and columnas == 1:
        estilo_academico(axes)
    else:
        for ax in (axes.flat if hasattr(axes, 'flat') else [axes]):
            estilo_academico(ax)
    return fig, axes


def guardar_figura(fig, nombre, subcarpeta='figuras'):
    """
    Guarda la figura en results/{subcarpeta}/{nombre}.png con DPI=300.

    Para mantener las figuras ordenadas por experimento, pasa subcarpeta como
    'figuras/01_busqueda_hiperparametros', 'figuras/02_linea_base', etc.
    """
    raiz = Path(__file__).resolve().parents[3]   # raíz del proyecto (TFG/)
    directorio = raiz / 'results' / subcarpeta
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f'{nombre}.png'
    fig.savefig(ruta, dpi=DPI, bbox_inches='tight', facecolor=COLOR_FONDO)
    plt.close(fig)
    return ruta
