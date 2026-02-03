import numpy as np
import pandas as pd
import sys
import warnings
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
warnings.filterwarnings('ignore')

try:
    # Ajusta la ruta para encontrar tus librerías 'src'
    current = Path(__file__).parent.resolve()
    sys.path.insert(0, str(current / 'src')) 
    
    from relieff_improved.ann.ann_v2_optimized import ANN
    from relieff_improved.proto.proto_v2_optimized import Proto
    from skrebate import ReliefF
    METHODS_AVAILABLE = True
    print("✓ Librerías importadas correctamente.")
except ImportError as e:
    print(f"⚠ No se encuentran las librerías ({e}).")
    print("  (Asegúrate de ejecutar esto donde tengas acceso a 'relieff_improved')")
    sys.exit(1)

# =============================================================================
# 1. GENERACIÓN EXACTA (32 MUESTRAS)
# =============================================================================
def generate_corral_32_deterministic():
    """
    Genera EXACTAMENTE las 32 muestras del dataset CorrAL original.
    Garantiza estructuralmente las estadísticas sin azar.
    """
    import itertools
    
    # 1. Generar base de 4 bits (16 combinaciones)
    # f1, f2, f3, f4
    base_16 = list(itertools.product([0, 1], repeat=4))
    
    # Duplicamos para tener 32 muestras (Bloque A y Bloque B)
    X_base = np.array(base_16 * 2) 
    
    df = pd.DataFrame(X_base, columns=['f1', 'f2', 'f3', 'f4'])
    
    # 2. Calcular Target Y (Regla fija)
    # Y = (f1 ^ f2) v (f3 ^ f4)
    y = ((df['f1'] == 1) & (df['f2'] == 1)) | ((df['f3'] == 1) & (df['f4'] == 1))
    y = y.astype(int).values
    
    # 3. Generar f5 (Irrelevante)
    # Para que sea irrelevante en 32 muestras, debe estar balanceada y no correlacionada.
    # Simplemente alternamos 0 y 1 para máxima entropía no relacionada con Y
    f5 = np.array([0, 1] * 16) 
    
    # 4. Generar f6 (Correlacionada 75%)
    # En 32 muestras, 75% = 24 aciertos (coinciden con Y) y 8 fallos (invertidos).
    # Vamos a forzar esto distribuyendo los errores.
    f6 = y.copy()
    
    # Invertimos 8 posiciones específicas para romper la perfección
    # Elegimos índices espaciados para no crear un patrón obvio
    indices_to_flip = [0, 4, 8, 12, 16, 20, 24, 28] 
    f6[indices_to_flip] = 1 - f6[indices_to_flip]
    
    # Concatenar todo
    X = np.column_stack((df['f1'], df['f2'], df['f3'], df['f4'], f5, f6))
    
    feature_names = ['f1 (Rel)', 'f2 (Rel)', 'f3 (Rel)', 'f4 (Rel)', 'f5 (Irr)', 'f6 (Corr)']
    
    return X, y, feature_names

# =============================================================================
# 2. EVALUACIÓN Y REPORTE
# =============================================================================
def analyze_rankings(weights, feature_names):
    # Crear pares (nombre, peso)
    feat_weights = list(zip(feature_names, weights))
    
    # Ordenar por peso descendente (Mayor es mejor)
    feat_weights.sort(key=lambda x: x[1], reverse=True)
    
    # Crear mapa de posiciones
    ranks = {}
    for i, (name, w) in enumerate(feat_weights):
        ranks[name] = i + 1 # 1-based index (1º, 2º...)
        
    return ranks, feat_weights

def run_experiment():
    print("\n" + "="*60)
    print(" EXPERIMENTO CORRAL ORIGINAL (32 MUESTRAS)")
    print(" Objetivo: Ver si los algoritmos caen en la trampa de f6")
    print("="*60)
    
    # Generar datos
    X, y, feat_names = generate_corral_32_deterministic()
    
    methods = {
        'ReliefF': ReliefF(n_features_to_select=6, n_neighbors=5), # k=5 es típico para N=32
        'ANN': ANN(),
        'Proto': Proto()
    }
    
    for name, model in methods.items():
        print(f"\n>> Evaluando: {name}")
        
        # Entrenar
        model.fit(X, y)
        
        # Analizar
        ranks, ordered_weights = analyze_rankings(model.feature_importances_, feat_names)
        
        # --- IMPRIMIR TABLA DE RANKING ---
        print(f"  {'Pos':<4} | {'Feature':<10} | {'Peso':<10} | {'Estado'}")
        print("  " + "-"*40)
        
        f6_rank = 99
        relevant_ranks = []
        
        for i, (fname, w) in enumerate(ordered_weights):
            pos = i + 1
            status = ""
            
            if "(Rel)" in fname:
                status = "✅ Relevante"
                relevant_ranks.append(pos)
            elif "(Corr)" in fname:
                status = "⚠️ TRAMPA"
                f6_rank = pos
            elif "(Irr)" in fname:
                status = "🗑️ Basura"
            
            print(f"  {pos:<4} | {fname:<10} | {w:.4f}     | {status}")
            
        # --- DIAGNÓSTICO FINAL ---
        print("  " + "-"*40)
        if f6_rank < min(relevant_ranks):
            print(f"  ❌ FALLO: Prefirió la trampa (f6 es #{f6_rank}) antes que las relevantes.")
        elif f6_rank > max(relevant_ranks):
            print(f"  🏆 ÉXITO PERFECTO: f6 está detrás de todas las relevantes.")
        else:
            print(f"  ⚠️ MIXTO: f6 se mezcló entre las relevantes.")

if __name__ == "__main__":
    run_experiment()