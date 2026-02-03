# 🧪 Experimentos de Robustez para Feature Selection

Este directorio contiene **5 experimentos complementarios** para evaluar la robustez y rendimiento de los algoritmos de feature selection: **ReliefF**, **ANN (Approximate Nearest Neighbors)** y **Proto (Prototype-based)**.

---

## 📋 Resumen de Experimentos

| #     | Experimento                                               | Dataset             | Qué evalúa                                      |
| ----- | --------------------------------------------------------- | ------------------- | ----------------------------------------------- |
| **1** | [Baseline + Ruido Gaussiano + Dilución](#1-test_corralpy) | Corral (sintético)  | Rendimiento base y robustez a ruido en features |
| **2** | [Wine Quality](#2-test_wine_qualitypy)                    | Wine Quality (real) | Validación en datos reales con ruido            |
| **3** | [Label Noise](#3-test_label_noisepy)                      | Corral              | Robustez a etiquetas corruptas                  |
| **4** | [Features Redundantes](#4-test_redundant_featurespy)      | Corral              | Capacidad de distinguir originales de copias    |
| **5** | [Sample Size](#5-test_sample_sizepy)                      | Corral              | Convergencia con pocos datos                    |

**Cobertura total:**

- ✅ Robustez a ruido (en X y en y)
- ✅ Discriminación (relevante vs irrelevante vs redundante)
- ✅ Eficiencia de datos (N grande vs pequeño)
- ✅ Validación en datos reales

---

## 🚀 Ejecución Rápida

```bash
# Ejecutar todos los experimentos
python test_corral.py
python test_wine_quality.py
python test_label_noise.py
python test_redundant_features.py
python test_sample_size.py

# O desde el directorio raíz
uv run tests/test_corral.py
# ... (similar para los demás)
```

**Duración total:** ~15-20 minutos  
**Output:** Gráficos PNG + CSVs + análisis en consola

---

## 📊 Experimentos Detallados

### 1. `test_corral.py`

**Dataset Corral: Baseline + Robustez Fundamental**

#### Qué es Corral

Dataset sintético con regla conocida:

```
y = sign(X₁·X₂ + X₃·X₄ - X₅·X₆)
```

- **800 samples**
- **20 features** (6 informativas, 14 ruido)
- **Ground truth conocido:** Features 0, 1, 2, 3, 4, 5

#### Tests incluidos

**A) BASELINE (dataset limpio)**

- Evalúa rendimiento base de cada algoritmo
- Mide F1 de selección: ¿Encuentra las 6 correctas?
- Mide F1 de clasificación: ¿Puede predecir bien después de seleccionar?

**B) RUIDO GAUSSIANO**
Inyecta ruido en la **regla de decisión**:

```python
decision = (X₁·X₂ + X₃·X₄ - X₅·X₆) + N(0, σ)
σ = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
```

**Qué mide:**

- σ pequeño (0.1) → ~5% label flips → Test de robustez suave
- σ grande (1.0) → ~40% label flips → Test extremo

**Resultado esperado:**

```
σ=0.1: F1 > 0.90 (todos aguantan)
σ=0.5: F1 > 0.70 (empieza a degradar)
σ=1.0: F1 > 0.50 (muy difícil)
```

**C) DILUCIÓN DIMENSIONAL**
Añade features **completamente aleatorias**:

```python
n_random = [10, 20, 50, 100, 200, 500, 1000]
X_augmented = [X_original, random_features]
```

**Qué mide:**

- ¿Pueden filtrar ruido puro?
- ¿Se confunden con muchas features irrelevantes?

**Resultado esperado:**

```
+10:   F1 > 0.90 (poco impacto)
+100:  F1 > 0.75 (aguanta bien)
+1000: F1 > 0.50 (muy diluido)
```

#### Output

- `corral_simple.png` - 2 gráficos (ruido + dilución)
- `noise_results.csv` - Tabla de ruido
- `dilution_results.csv` - Tabla de dilución

---

### 2. `test_wine_quality.py`

**Dataset Real: Wine Quality**

#### Qué es Wine Quality

Dataset de química de vinos (UCI ML Repository):

- **1599 muestras** de vino tinto
- **11 features químicas:**
  - Acidez (fixed, volatile, citric)
  - Azúcar residual
  - Cloruros, sulfatos
  - Densidad, pH, alcohol
- **Clasificación binaria:** Calidad ≥ 6 (bueno) vs < 6 (malo)

#### Por qué este dataset

- ✅ **No es típico** (no MNIST/Iris/Covertype)
- ✅ **Features interpretables** (química)
- ✅ **Problema real** con ruido natural
- ✅ **Tamaño razonable** (~1600 samples)

#### Tests incluidos

**A) BASELINE**

- Evalúa en datos limpios
- Muestra **Top-5 features con nombres** (ej: 'alcohol', 'sulphates')
- F1 y Accuracy de clasificación

**B) RUIDO EN FEATURES**
Simula **errores de medición** de sensores:

```python
X_noisy = X + N(0, noise_ratio × std(X))
noise_ratio = [0.05, 0.10, 0.20, 0.30, 0.50]
```

**Qué mide:**

- 5% → Sensor con 5% de error
- 50% → Sensor muy impreciso

**Resultado esperado:**

```
5%:  Degradación < 5%
30%: Degradación 10-20%
50%: Degradación 20-35%
```

**C) DILUCIÓN**
Igual que Corral pero con dataset real:

```python
n_random = [5, 10, 20, 50, 100]
```

#### Output

- `wine_quality_test.png` - 2 gráficos
- `wine_noise_results.csv`
- `wine_dilution_results.csv`

#### Interpretación de resultados

**Sin ground truth de features**, se evalúa mediante:

1. **F1 de clasificación** (no de selección)
2. **Coherencia con conocimiento experto** (¿tiene sentido que alcohol sea importante?)
3. **Robustez** (menor degradación = mejores features)

---

### 3. `test_label_noise.py`

**Etiquetas Corruptas: Errores de Anotación**

#### Qué evalúa

¿Pueden los algoritmos encontrar features importantes cuando las **etiquetas están incorrectas**?

#### Metodología

Corrompe aleatoriamente las etiquetas de **entrenamiento**:

```python
flip_ratio = [0%, 5%, 10%, 15%, 20%, 30%]
y_train[random_indices] = 1 - y_train[random_indices]  # 0→1, 1→0
```

**Importante:** El test set **queda limpio** para evaluar correctamente.

#### Por qué importa

- **Muy común en real:** Errores humanos de etiquetado
- **Crítico en:** Medical data, crowd-sourcing, sensores defectuosos
- **Diferente a ruido gaussiano:** Aquel corrompe X, este corrompe y

#### Resultado esperado

```
Label noise 0%:  F1 = 1.00 (baseline)
Label noise 5%:  F1 = 0.95 (↓5%)
Label noise 20%: F1 = 0.80 (↓20%)
Label noise 30%: F1 = 0.65-0.72 (↓25-35%)
```

**¿Quién aguanta mejor?**
El algoritmo con mejor regularización implícita.

#### Output

- `test_label_noise.png` - Curva de degradación
- Análisis de degradación relativa

---

### 4. `test_redundant_features.py`

**Features Redundantes: Copias Correlacionadas**

#### Qué evalúa

¿Pueden distinguir features **originales** de **copias casi idénticas**?

#### Metodología

Crea copias correlacionadas de features importantes:

```python
correlation = 0.85  # Muy alta
for each informative feature:
    redundant = original × 0.85 + noise × 0.15
```

Añade: `[0, 5, 10, 20, 30, 50]` copias

#### Por qué importa

- **Realismo:** Múltiples sensores miden lo mismo (temperatura en 3 lugares)
- **Multi-view data:** Diferentes vistas de la misma información
- **Test de discriminación:** ¿Eligen una o se confunden con todas?

#### Resultado esperado

```
+0:  F1 = 1.00 (baseline)
+10: F1 = 0.90 (↓10%) - Debería aguantar
+50: F1 = 0.70-0.80 (↓20-30%) - Se confunde con muchas copias
```

**¿Quién es mejor?**
El que tenga mejor capacidad de identificar información única.

#### Output

- `test_redundant_features.png` - Curva de degradación
- Análisis de degradación relativa

---

### 5. `test_sample_size.py`

**Sample Size: Convergencia con Pocos Datos**

#### Qué evalúa

¿Funcionan con **datasets pequeños** o necesitan muchos datos?

#### Metodología

Entrena con subconjuntos de diferentes tamaños:

```python
sample_sizes = [50, 100, 200, 300, 500, 800]
# Test set fijo de 200 samples
```

#### Por qué importa

- **Crítico en medicina:** Solo 100 pacientes disponibles
- **Materiales raros:** Experimentos caros
- **Early stopping:** ¿Cuándo dejar de recolectar datos?

#### Resultado esperado

```
N=50:  F1 = 0.55 (muy poco para aprender)
N=200: F1 = 0.75 (empiezan a funcionar)
N=500: F1 = 0.85+ (convergen)
```

**Métricas de convergencia:**

- **Samples para F1 > 0.85:** Menor es mejor
- **Tasa de crecimiento:** Más rápida es mejor

#### Output

- `test_sample_size.png` - Curva de convergencia
- Análisis de convergencia (samples necesarios)

---

## 📊 Métricas Utilizadas

### F1-Score

**Fórmula:**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Tipos de F1:**

**1) F1 de Selección** (solo en Corral):

```python
Top-6 seleccionadas: [0, 1, 2, 3, 4, 14]
Ground truth: [0, 1, 2, 3, 4, 5]
→ TP = 5, FP = 1, FN = 1
→ Precision = 5/6, Recall = 5/6
→ F1 = 0.83
```

**2) F1 de Clasificación** (en todos):

```python
# Después de seleccionar features
y_pred = classifier.predict(X_test_selected)
F1 = f1_score(y_test, y_pred)
```

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Cuándo usar cada una:**

- **Datasets balanceados:** Accuracy es suficiente
- **Datasets desbalanceados:** F1 es más robusto
- **En los tests:** Reportamos ambas

### Perfect Top-K

Métrica binaria: ¿Las top-K features seleccionadas coinciden **exactamente** con el ground truth?

```python
Perfect = (set(top_k) == set(ground_truth))
```

Solo aplicable en Corral (dataset sintético con ground truth conocido).

---

## 🎯 Interpretación de Resultados

### Degradación Relativa

```python
degradation = ((F1_clean - F1_noisy) / F1_clean) × 100
```

**Clasificación:**

- 🏆 **< 10%:** Excelente robustez
- ✓ **10-25%:** Buena robustez
- ⚠️ **> 25%:** Robustez pobre

### Comparación de Algoritmos

**ReliefF (Baseline):**

- Simple, rápido
- Buen rendimiento base
- Referencia estándar

**ANN v2:**

- Búsqueda eficiente de vecinos
- Mejor escalabilidad (N grande)
- Esperado: Ligeramente mejor que ReliefF

**Proto v2:**

- Clustering + prototipos
- Más rápido en algunos casos
- Esperado: Comparable o ligeramente mejor

---

## 📁 Estructura de Archivos

```
tests/
├── test_corral.py              # Experimento 1
├── test_wine_quality.py        # Experimento 2
├── test_label_noise.py         # Experimento 3
├── test_redundant_features.py  # Experimento 4
├── test_sample_size.py         # Experimento 5
└── README.md                   # Este archivo
```

**Output generado:**

```
tests/
├── corral_simple.png
├── noise_results.csv
├── dilution_results.csv
├── wine_quality_test.png
├── wine_noise_results.csv
├── wine_dilution_results.csv
├── test_label_noise.png
├── test_redundant_features.png
└── test_sample_size.png
```

---

## 🔧 Dependencias

```python
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
matplotlib>=3.4
```

**Instalación:**

```bash
pip install numpy pandas scikit-learn matplotlib
# O si usas uv
uv pip install numpy pandas scikit-learn matplotlib
```

---

## 💡 Tips para Análisis

### Baseline (Corral + Wine)

1. **Perfect Top-6 = ✓:** Algoritmo encuentra exactamente las correctas
2. **F1 > 0.90:** Excelente selección
3. **F1 = 0.75-0.90:** Buena selección
4. **F1 < 0.75:** Revisar parámetros

### Ruido (Gaussiano + Features + Labels)

1. **Degradación < 15%:** Muy robusto
2. **Degradación 15-25%:** Aceptable
3. **Degradación > 30%:** Poco robusto

### Dilución + Redundantes

1. **Aguanta +100 features aleatorias:** Buen filtrado
2. **Aguanta +50 copias correlacionadas:** Buena discriminación
3. **Colapsa con +20:** Necesita mejora

### Sample Size

1. **Converge con N < 300:** Eficiente con pocos datos
2. **Converge con N = 300-500:** Estándar
3. **Necesita N > 500:** Ineficiente

---

## 📖 Para el TFG

### Sección de Resultados Experimentales

**Estructura sugerida:**

1. **Descripción de Datasets**
   - Corral: Dataset sintético con ground truth
   - Wine Quality: Dataset real de química

2. **Metodología**
   - 5 experimentos complementarios
   - 3 algoritmos comparados
   - Métricas: F1, Accuracy, Perfect Top-K

3. **Resultados por Experimento**
   - Baseline: Rendimiento en condiciones ideales
   - Robustez: Cómo aguantan diferentes tipos de corrupción
   - Eficiencia: Comportamiento con pocos datos

4. **Análisis Comparativo**
   - Tabla resumen de degradaciones
   - Gráficos de robustez
   - Conclusiones sobre ventajas/desventajas

### Ejemplo de Texto

> "Se evaluaron tres algoritmos de feature selection (ReliefF, ANN v2, Proto v2) mediante cinco experimentos complementarios que cubren robustez a ruido, discriminación de features redundantes y eficiencia de datos. Los resultados muestran que ANN v2 presenta una degradación del 24% ante 30% de label noise, comparado con 32% de ReliefF, indicando mejor robustez en escenarios con etiquetado imperfecto. En el dataset real Wine Quality, todos los algoritmos identificaron correctamente alcohol y acidez volátil como features principales, validando la coherencia con el conocimiento del dominio."

---

## 🐛 Troubleshooting

### Error: "Module not found"

```bash
# Asegúrate de estar en el directorio raíz
cd /path/to/relieff-tfg
uv run tests/test_corral.py
```

### Los gráficos no se generan

```python
# Verifica que matplotlib esté instalado
pip install matplotlib
```

### Resultados inconsistentes

- Usa `random_state=42` en todos los scripts (ya incluido)
- Los resultados deberían ser reproducibles

### Tiempo de ejecución muy largo

- Reduce `n_samples` en los scripts (línea ~60)
- Reduce `noise_levels` o `n_random_list` (menos puntos)

---

## 📚 Referencias

**Datasets:**

- Corral: Synthetic dataset para feature selection (John et al., 1994)
- Wine Quality: P. Cortez et al., Decision Support Systems, 2009

**Algoritmos:**

- ReliefF: Kononenko, 1994
- ANN: Approximate Nearest Neighbors para escalabilidad
- Proto: Prototype-based selection con clustering

---

## ✅ Checklist Final

Antes de presentar resultados en el TFG, verifica:

- [ ] Los 5 experimentos ejecutan sin errores
- [ ] Se generan todos los gráficos PNG
- [ ] Se generan todos los CSVs
- [ ] Los resultados son consistentes entre ejecuciones
- [ ] Baseline tiene F1 > 0.80 (si no, revisar implementación)
- [ ] Los gráficos son legibles (labels, leyendas, títulos)
- [ ] Tienes backup de todos los resultados
- [ ] Has documentado parámetros utilizados

---

## 📧 Contacto

**Autor:** Nicolás Aller  
**Fecha:** Enero 2026  
**TFG:** Algoritmos de Feature Selection basados en ReliefF
