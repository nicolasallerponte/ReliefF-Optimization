# Optimización Computacional de ReliefF

Implementación y evaluación de dos variantes que reducen el coste cuadrático de **ReliefF**
sin sacrificar la calidad de la selección de características: **HNSW-ReliefF** (vecinos
aproximados) y **Proto-ReliefF** (prototipos de clase).

| Algoritmo | Idea clave | Complejidad temporal |
|---|---|---|
| **ReliefF** (referencia, `skrebate`) | Vecinos exactos por fuerza bruta | O(n²·m) |
| **HNSW-ReliefF** (`ANN`) | Vecinos aproximados con índice HNSW + núcleo Numba JIT | O(n·log n·m) |
| **Proto-ReliefF** (`Proto`) | Compresión a prototipos con MiniBatchKMeans | O(p²·m), p ≪ n |

Ambas variantes preservan el criterio de relevancia de ReliefF (y por tanto su capacidad de
detectar interacciones entre variables), de modo que la ganancia de eficiencia no se paga con
pérdida de calidad. La evaluación cubre eficiencia, escalabilidad al límite, robustez al ruido,
preservación de la calidad y **huella de carbono**.

---

## Algoritmos implementados

### HNSW-ReliefF (`ANN`)

Sustituye la búsqueda exacta de vecinos por un índice HNSW (*Hierarchical Navigable Small
World*, `hnswlib`), construido por clase. El núcleo de puntuación está compilado con Numba JIT
(`@njit`, `fastmath`, `parallel`) para eliminar el coste del intérprete de Python.

- **`n_neighbors` adaptativo** según el ratio variables/muestras `ρ = m/n`:
  ```
  k = floor(15 / (1 + ρ · 5.0)),  acotado a [5, 15]
  ```
- **`iter_ratio`**: fracción de muestras usadas como consulta. En modo `auto`, escala de
  1.0 (n < 2000) a 0.3 (n ≥ 20000), reduciendo el coste sin perder correlación con los pesos
  de cobertura completa (r > 0.99 con ratio = 0.3).

### Proto-ReliefF (`Proto`)

Comprime cada clase en un conjunto acotado de prototipos con `MiniBatchKMeans` y aplica ReliefF
exacto sobre ellos. Los centroides se anclan a instancias reales (*snap-to-grid*) para
preservar la validez en variables discretas.

- **`sigma` adaptativo** (grado de compresión):
  ```
  sigma_eff = clip(0.10 · (1 + log(ρ / 0.05)), 0.10, 0.25)
  p_c       = min(200, floor(n_c · sigma_eff))
  ```
- Ponderación *softmax* de distancia inversa de los prototipos vecinos.
- Refinamiento opcional con LVQ (`use_lvq`, desactivado por defecto).

---

## Estructura del proyecto

```
ReliefF-Optimization/
├── src/relieff_opt/             # Paquete Python
│   ├── ann.py                   # HNSW-ReliefF (HNSW + Numba)
│   ├── proto.py                 # Proto-ReliefF (MiniBatchKMeans + LVQ opcional)
│   └── utils/
│       ├── conjuntos.py         # Generación/carga de los conjuntos de datos
│       ├── experimento.py       # Runner multi-semilla, guardado de tablas (CSV + LaTeX)
│       ├── metricas.py          # Métricas y tests estadísticos
│       └── paleta.py            # Colores y estilos de las figuras
│
├── experimentos/                # 17 scripts independientes (uno por análisis)
│   ├── 01_busqueda_hiperparametros.py   ...   17_robustez_ruido.py
│
├── config/
│   └── hiperparametros.yaml     # Configuración Pareto-óptima centralizada
│
├── scripts/
│   └── run_all_experiments.sh   # Lanza toda la batería en orden
│
├── pyproject.toml               # Dependencias y metadatos del paquete
├── LICENSE                      # MIT
└── README.md
```

Las salidas (figuras y tablas) se generan en una carpeta `results/` al ejecutar; no se
versiona en el repositorio.

---

## Instalación

Requiere **Python ≥ 3.10**. Se recomienda [`uv`](https://github.com/astral-sh/uv) para
gestionar el entorno:

```bash
git clone https://github.com/nicolasallerponte/ReliefF-Optimization.git
cd ReliefF-Optimization
uv sync
```

Alternativamente, con `pip`:

```bash
pip install -e .
```

---

## Uso

### Como librería

```python
from relieff_opt import ANN, Proto

# X: array (n, m), y: array (n,)
selector = ANN(n_features_to_select=10)      # HNSW-ReliefF
selector.fit(X, y)
ranking = selector.top_features_              # índices ordenados por relevancia

proto = Proto(n_features_to_select=10)        # Proto-ReliefF
proto.fit(X, y)
```

### Reproducir los experimentos

```bash
# Batería completa
chmod +x scripts/run_all_experiments.sh
./scripts/run_all_experiments.sh

# Un experimento concreto
uv run python experimentos/02_linea_base.py
```

El experimento 06 (análisis estadístico) y el 16 (similitud de *rankings*) consumen las salidas
del 02 (línea base); el resto son independientes entre sí.

---

## Descripción de los experimentos

| # | Script | Pregunta | Métrica principal |
|---|---|---|---|
| 01 | `01_busqueda_hiperparametros.py` | ¿Configuración óptima de HNSW-ReliefF y Proto-ReliefF? | Frontera de Pareto F1–tiempo |
| 02 | `02_linea_base.py` | ¿Cómo comparan los tres algoritmos en calidad? | F1 macro (5×5 CV) |
| 03 | `03_robustez.py` | ¿Cómo degradan ante distintos tipos de ruido? | Curvas de degradación |
| 04 | `04_escalabilidad.py` | ¿Cómo escala el tiempo con n? | Exponente α (ley de potencia) |
| 05 | `05_trampa_corral100.py` | ¿Distinguen variables relevantes de correlación espuria? | Posición media en el ranking |
| 06 | `06_analisis_estadistico.py` | ¿Son significativas las diferencias? | Wilcoxon, Friedman, Cohen's d |
| 07 | `07_convergencia_iter_ratio.py` | ¿Con qué `iter_ratio` convergen los pesos? | Correlación vs cobertura completa |
| 08 | `08_huella_carbono.py` | ¿Cuánto CO₂ emite cada algoritmo? | Frontera Pareto F1–CO₂ |
| 09 | `09_estabilidad_seleccion.py` | ¿Cómo de estable es la selección? | Índice de Kuncheva |
| 10 | `10_sensibilidad_k.py` | ¿A partir de qué k se estabiliza? | Curva F1 vs k |
| 11 | `11_crossover_hnsw_exacto.py` | ¿Degrada el índice aproximado frente al exacto? | Tiempo y F1 vs n |
| 12 | `12_validacion_adaptativo.py` | ¿Funcionan las fórmulas adaptativas de k y sigma? | Validación de las heurísticas |
| 13 | `13_ablacion_componentes.py` | ¿Qué componente aporta la aceleración? | Aceleración acumulada |
| 14 | `14_numba_warmup.py` | ¿Cuál es el coste de compilación JIT? | Latencia cold vs warm |
| 15 | `15_escala_real.py` | ¿Escala a datos reales masivos? | Tiempo y F1 (CoverType/SUSY/HEPMASS) |
| 16 | `16_similitud_rankings.py` | ¿Reproducen el ranking de ReliefF? | Spearman + solapamiento top-10 |
| 17 | `17_robustez_ruido.py` | ¿Añaden fragilidad ante ruido de etiquetas? | Recuperación en CorrAL-100 |

Los conjuntos de tamaño pequeño y medio se generan/cargan automáticamente; los reales masivos
del experimento 15 (CoverType, SUSY, HEPMASS) se descargan en la primera ejecución.

---

## Reproducibilidad

- **Semillas:** `[42, 123, 456, 789, 1234]` en todos los análisis de calidad.
- **Validación:** *5-fold* estratificada repetida sobre las 5 semillas (25 evaluaciones por celda).
- **Clasificador posterior:** Random Forest (100 árboles, profundidad 10) sobre las 10 variables
  de mayor relevancia.
- **Tiempo:** medido con una única semilla (el tiempo de ajuste es casi determinista).
- **Tests:** Wilcoxon pareado, Friedman, corrección de Bonferroni y tamaño de efecto (Cohen's d).

Las configuraciones por defecto son las Pareto-óptimas de `config/hiperparametros.yaml`.

---

## Dependencias principales

`numpy`, `scikit-learn`, `hnswlib`, `numba`, `pandas`, `matplotlib`, `seaborn`, `scipy`,
`joblib`, `skrebate`, `pyyaml`, `codecarbon`. Instalación completa con `uv sync`.

---

## Cómo contribuir

Las contribuciones son bienvenidas. Consulta [`CONTRIBUTING.md`](CONTRIBUTING.md) para el flujo
de trabajo y [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) para las normas de convivencia.

## Licencia

Distribuido bajo licencia **MIT**. Consulta [`LICENSE`](LICENSE).

---

## Contexto

Este proyecto reduce el coste computacional y energético de ReliefF mediante búsqueda aproximada
de vecinos (HNSW) y compresión por prototipos, con especial atención a la eficiencia y la huella
de carbono en el marco de la **Cátedra Inditex-UDC de IA en Algoritmos Verdes** (Universidade da
Coruña).

Si este trabajo te resulta útil, puedes citarlo:

```bibtex
@software{aller_relieff_optimization,
  author  = {Aller Ponte, Nicolás},
  title   = {Optimización Computacional de ReliefF: Búsqueda Aproximada y Prototipado},
  year    = {2026},
  url     = {https://github.com/nicolasallerponte/ReliefF-Optimization}
}
```
