# Optimización de ReliefF para Selección de Características

**TFG - Nicolás Aller Ponte**  
Grado en Ciencia de Datos, Universidade da Coruña  
En colaboración con la **Cátedra Inditex-UDC de IA en Algoritmos Verdes**

---

## Descripción

Este trabajo propone y evalúa dos optimizaciones del algoritmo ReliefF que reducen su complejidad computacional sin sacrificar calidad en la selección de características:

| Algoritmo | Idea clave | Complejidad |
|---|---|---|
| **ReliefF** (skrebate) | Vecinos exactos por fuerza bruta | O(n²·d) |
| **ANN** | Vecinos aproximados con HNSW + JIT Numba | O(n log n · d) |
| **Proto** | Prototipos de clase con MiniBatchKMeans | O(n · k · d) |

La evaluación cubre rendimiento en selección de características, robustez ante distintos tipos de ruido, escalabilidad, estabilidad y, en línea con el enfoque verde de la Cátedra, **huella de carbono** (experimento 08).

---

## Algoritmos implementados

### ANN - ReliefF con Vecinos Aproximados

Sustituye la búsqueda exacta de vecinos por un índice HNSW (*Hierarchical Navigable Small World*) de la librería `hnswlib`. El núcleo de cálculo de puntuaciones está compilado con Numba JIT para evitar el bucle Python.

**Parámetro adaptativo clave - n_neighbors:**
El tamaño del vecindario se adapta al ratio features/muestras mediante:

```
k = 15 / (1 + ratio × 5.0),  clip([5, 15])
```

Esto proporciona un vecindario amplio cuando las features son pocas respecto a las muestras (datasets limpios) y más local cuando la dimensión relativa es alta (datasets con dilución).

**iter_ratio:** fracción de muestras usadas como queries. En modo `'auto'`, escala de 1.0 (N<2000) a 0.3 (N>20000), reduciendo el coste sin perder correlación con los pesos completos (r>0.95 con ratio=0.3, ver experimento 07).

### Proto - ReliefF con Prototipos

Reemplaza los vecinos individuales por centroides de clase generados con `MiniBatchKMeans`. El número de clusters escala con el ratio features/muestras mediante sigma:

```
sigma_eff = clip(0.10 × (1 + log(ratio / 0.05)), 0.10, 0.25)
n_clusters = int(count × sigma_eff)
```

Opcionalmente, los centros abstractos se refinan con LVQ (*Learning Vector Quantization*) con momentum, aunque los experimentos muestran que `use_lvq=False` es igual o mejor en todos los datasets evaluados.

---

## Estructura del proyecto

```
TFG/
├── src/relieff_opt/            # Paquete Python
│   ├── ann.py                  # ANN (HNSW + Numba)
│   ├── proto.py                # Proto (KMeans + LVQ opcional)
│   └── utils/
│       ├── paleta.py           # Colores y estilos centralizados
│       ├── experimento.py      # Runner multi-seed, guardar_tabla (CSV + LaTeX)
│       ├── conjuntos.py        # 12 datasets con ground truth
│       └── metricas.py         # KCI, Cohen's d, Nemenyi, IC bootstrap
│
├── experimentos/               # Scripts de experimentos (ejecución independiente)
│   ├── 01_busqueda_hiperparametros.py
│   ├── 02_linea_base.py
│   ├── 03_robustez.py
│   ├── 04_escalabilidad.py
│   ├── 05_trampa_corral100.py
│   ├── 06_analisis_estadistico.py
│   ├── 07_convergencia_iter_ratio.py
│   ├── 08_huella_carbono.py     ← Green AI
│   ├── 09_estabilidad_seleccion.py
│   └── 10_sensibilidad_k.py
│
├── config/
│   └── hiperparametros.yaml    # Parámetros centralizados
│
├── results/
│   ├── figuras/                # PNG (DPI=300), subcarpeta por experimento
│   └── tablas/                 # CSV + .tex (LaTeX), subcarpeta por experimento
│
└── scripts/
    └── run_all_experiments.sh  # Ejecuta los 10 experimentos en orden
```

---

## Instalación

```bash
git clone https://github.com/nicolasallerponte/ReliefF-Optimization.git
cd ReliefF-Optimization

# uv gestiona el entorno automáticamente
uv sync
```

---

## Ejecución

### Ejecutar todos los experimentos

```bash
chmod +x scripts/run_all_experiments.sh
./scripts/run_all_experiments.sh
```

### Ejecutar un experimento individual

```bash
uv run python experimentos/02_linea_base.py
```

### Orden recomendado

El experimento 06 (análisis estadístico) depende de que el 02 (línea base) haya generado sus resultados.  
El experimento 08 (huella de carbono) es independiente pero se enriquece con los F1 del 02.  
El resto son completamente independientes entre sí.

---

## Descripción de los experimentos

| # | Script | Pregunta de investigación | Outputs |
|---|---|---|---|
| 01 | `01_busqueda_hiperparametros.py` | ¿Cuáles son los hiperparámetros óptimos de ANN y Proto? | Grid search ANN y Proto en 12 datasets, 5-fold CV |
| 02 | `02_linea_base.py` | ¿Cómo se comparan los tres algoritmos con sus configuraciones óptimas? | F1 medio ± std en 12 datasets |
| 03 | `03_robustez.py` | ¿Cómo degrada el rendimiento ante ruido gaussiano, de etiquetas, de features y valores faltantes? | Curvas de degradación + radar chart |
| 04 | `04_escalabilidad.py` | ¿Cómo escala el tiempo de ejecución con n? ¿ANN rompe O(n²)? | Tiempo vs n, log-log con ajuste potencial |
| 05 | `05_trampa_corral100.py` | ¿Distinguen los algoritmos features relevantes de features correlacionadas o irrelevantes? | Posición media de f0-f5 en CorrAL-100 |
| 06 | `06_analisis_estadistico.py` | ¿Son las diferencias estadísticamente significativas? | Wilcoxon, Friedman, Nemenyi, Cohen's d, IC 95% |
| 07 | `07_convergencia_iter_ratio.py` | ¿Con qué iter_ratio convergen los pesos de ANN? | Correlación de Spearman vs iter_ratio |
| 08 | `08_huella_carbono.py` | ¿Cuánto CO₂ emite cada algoritmo? ¿Existe trade-off con el rendimiento? | Frontera de Pareto F1 vs CO₂, ratio verde |
| 09 | `09_estabilidad_seleccion.py` | ¿Cuán estable es la selección entre distintas ejecuciones? | Índice de Kuncheva (KCI), heatmaps de frecuencia |
| 10 | `10_sensibilidad_k.py` | ¿A partir de qué k se estabiliza el rendimiento? | Curva F1 vs k, estimación del codo |

---

## Datasets

| Dataset | Muestras | Features | Relevantes | Origen |
|---|---|---|---|---|
| Breast Cancer | 569 | 30 | - | sklearn |
| Wine | 178 | 13 | - | sklearn |
| Digits | 1797 | 64 | - | sklearn |
| Corral | 800 | 20 | 0–5 | sintético |
| CorrAL-100 | 1600 | 99 | 0–3 | sintético |
| XOR | 1000 | 10 | 0, 1 | sintético |
| Moons | 1000 | 2 | 0, 1 | sintético |
| Circles | 1000 | 2 | 0, 1 | sintético |
| AltaDim | 500 | 100 | 0–9 | sintético |
| Desbalanceado | 1000 | 20 | 0–14 | sintético |
| Ruidoso | 800 | 25 | 0–9 | sintético |
| Grande | 5000 | 30 | 0–19 | sintético |

---

## Prácticas de reproducibilidad

- **Multi-seed:** todos los experimentos usan `SEMILLAS = [42, 123, 456, 789, 1234]`
- **Métricas:** se reporta media ± std sobre las 5 semillas
- **IC 95%:** calculado por bootstrap (n=2000 remuestras)
- **Tests estadísticos:** Wilcoxon (pares) + Friedman (global) + Bonferroni + Nemenyi
- **Tamaño de efecto:** Cohen's d para cada comparación pareada
- **CV:** 5-fold stratified cross-validation
- **Colores fijos:** ANN=azul (#3498db), Proto=rojo (#e74c3c), ReliefF=verde (#2ecc71)
- **Figuras:** DPI=300, guardadas en `results/figuras/{experimento}/`
- **Tablas:** CSV + .tex (LaTeX) en `results/tablas/{experimento}/`

---

## Dependencias principales

```
numpy, scikit-learn, hnswlib, numba, pandas, matplotlib, seaborn
scipy, joblib, skrebate, pyyaml, codecarbon
```

Instalación completa con `uv sync`.

---

## Contexto académico

Este TFG explora la reducción del coste computacional de ReliefF (O(n²)) mediante estructuras de datos aproximadas (HNSW) y representaciones compactas (prototipos). El ángulo verde -medición de huella de carbono y eficiencia energética- es especialmente relevante en el marco de la **Cátedra Inditex-UDC de IA en Algoritmos Verdes**, orientada al desarrollo de técnicas de IA sostenibles.
