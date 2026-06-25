#!/bin/bash
# ============================================================================
# Script maestro - ejecuta todos los experimentos del TFG en orden
# ============================================================================
# Uso:  chmod +x scripts/run_all_experiments.sh
#       ./scripts/run_all_experiments.sh
#
# Para ejecutar solo uno:
#       uv run python experimentos/02_linea_base.py
# ============================================================================

set -e

INICIO=$(date +%s)

echo "============================================================"
echo "TFG: Optimización de ReliefF - Cátedra Inditex-UDC"
echo "Autor: Nicolás Aller Ponte"
echo "============================================================"
echo ""
echo "Duración estimada: ~200 minutos"
echo "Experimentos: 01 al 17 en secuencia"
echo ""
read -p "Pulsa ENTER para comenzar o Ctrl+C para cancelar... "
echo ""

ejecutar() {
    local num="$1"
    local archivo="$2"
    local descripcion="$3"
    echo "------------------------------------------------------------"
    echo "[${num}] ${descripcion}"
    echo "------------------------------------------------------------"
    if uv run python "${archivo}"; then
        echo "  OK"
    else
        echo "  FALLO (continuando con el siguiente...)"
    fi
    echo ""
}

ejecutar "01/17" "experimentos/01_busqueda_hiperparametros.py" \
    "Búsqueda de hiperparámetros - ANN y Proto"

ejecutar "02/17" "experimentos/02_linea_base.py" \
    "Línea base - ReliefF vs ANN vs Proto en 12 datasets"

ejecutar "03/17" "experimentos/03_robustez.py" \
    "Robustez ante ruido - CorrAL-100 (4 tipos)"

ejecutar "04/17" "experimentos/04_escalabilidad.py" \
    "Escalabilidad temporal - O(n) vs O(n²) + benchmark CoverType real"

ejecutar "05/17" "experimentos/05_trampa_corral100.py" \
    "Feature trampa/correlacionada - CorrAL-100"

ejecutar "06/17" "experimentos/06_analisis_estadistico.py" \
    "Análisis estadístico - Wilcoxon + Friedman + Nemenyi + Cohen's d"

ejecutar "07/17" "experimentos/07_convergencia_iter_ratio.py" \
    "Convergencia del iter_ratio - ANN (force_exact, aísla subsampling)"

ejecutar "08/17" "experimentos/08_huella_carbono.py" \
    "Huella de carbono - Green AI (Cátedra Inditex-UDC)"

ejecutar "09/17" "experimentos/09_estabilidad_seleccion.py" \
    "Estabilidad de la selección - Índice de Kuncheva"

ejecutar "10/17" "experimentos/10_sensibilidad_k.py" \
    "Sensibilidad al número de features seleccionadas (k)"

ejecutar "11/17" "experimentos/11_crossover_hnsw_exacto.py" \
    "Crossover HNSW vs búsqueda exacta - validación de la aproximación"

ejecutar "12/17" "experimentos/12_validacion_adaptativo.py" \
    "Validación de fórmulas adaptativas - k (ANN) y sigma (Proto)"

ejecutar "13/17" "experimentos/13_ablacion_componentes.py" \
    "Ablación de componentes ANN - contribución de Numba / HNSW / subsampling"

ejecutar "14/17" "experimentos/14_numba_warmup.py" \
    "Numba warm-up - latencia de compilación JIT (cold vs warm start)"

ejecutar "15/17" "experimentos/15_escala_real.py" \
    "Escalabilidad en datasets reales masivos - CoverType / SUSY / HEPMASS"

ejecutar "16/17" "experimentos/16_similitud_rankings.py" \
    "Similitud de rankings - Spearman + solapamiento top-10 vs ReliefF"

ejecutar "17/17" "experimentos/17_robustez_ruido.py" \
    "Robustez ante ruido de etiquetas - recuperación en CorrAL-100"

FIN=$(date +%s)
MINUTOS=$(( (FIN - INICIO) / 60 ))
SEGUNDOS=$(( (FIN - INICIO) % 60 ))

echo "============================================================"
echo "COMPLETADO en ${MINUTOS}m ${SEGUNDOS}s"
echo ""
echo "Resultados:"
echo "  results/figuras/  - ordenadas por experimento"
echo "  results/tablas/   - CSV + LaTeX por experimento"
echo "============================================================"
