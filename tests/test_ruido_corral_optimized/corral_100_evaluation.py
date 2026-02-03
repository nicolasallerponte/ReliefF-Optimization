# corral_100_evaluation.py
# Evaluación de ANN, Proto y ReliefF en CorrAL-100 con clase AND/OR

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys

# Setup
def setup():
    current = Path(__file__).parent.resolve()
    root = current
    while root != root.parent:
        if (root / 'src').exists():
            break
        root = root.parent
    sys.path.insert(0, str(root / 'src'))

setup()

from relieff_improved.ann.ann_v2_optimized import ANN
from relieff_improved.proto.proto_v2_optimized import Proto

try:
    from skrebate import ReliefF as ReliefF_Original
    relieff_available = True
except ImportError:
    relieff_available = False


def generate_corrAL100(n_samples=5000, p_noise=93, seed=42, f6_corr=0.75):
    """
    Genera dataset CorrAL-100 según John et al. (1994):
      - f1..f4 Bernoulli(0.5) - RELEVANTES
      - y = (f1 & f2) | (f3 & f4)  [AND/OR, NO XOR]
      - f5 ~ Bernoulli(0.5) - IRRELEVANTE
      - f6 con P(f6==y) = f6_corr - CORRELACIONADA
      - p_noise características aleatorias Bernoulli(0.5)
    
    Devuelve X (n x 100), y (n,), dict con indices relevantes
    """
    rng = np.random.RandomState(seed)
    n = n_samples
    
    # f1..f4 - RELEVANTES
    f = rng.binomial(1, 0.5, size=(n, 4)).astype(np.int32)
    f1, f2, f3, f4 = f[:,0], f[:,1], f[:,2], f[:,3]
    
    # Clase = (f1 AND f2) OR (f3 AND f4)
    y = ((f1 & f2) | (f3 & f4)).astype(np.int32)
    
    # f5 - IRRELEVANTE
    f5 = rng.binomial(1, 0.5, size=n).astype(np.int32)
    
    # f6 - CORRELACIONADA con y al f6_corr%
    # Método: con prob f6_corr -> f6=y, con prob (1-f6_corr) -> f6 random
    random_flip = rng.binomial(1, 1 - f6_corr, size=n)
    rand_bits = rng.binomial(1, 0.5, size=n)
    f6 = np.where(random_flip==0, y, rand_bits).astype(np.int32)
    
    # Ruido: 93 features aleatorias
    noise = rng.binomial(1, 0.5, size=(n, p_noise)).astype(np.int32)
    
    # Concatenar: [f1, f2, f3, f4, f5, f6, noise] -> 100 columnas
    X = np.hstack([f, f5.reshape(-1,1), f6.reshape(-1,1), noise]).astype(np.float32)
    
    # Ground truth causal indices (0-based)
    gt_causal = np.array([0, 1, 2, 3], dtype=int)
    idx_f5 = 4
    idx_f6 = 5
    
    return X, y, {'causal_idx': gt_causal, 'f5_idx': idx_f5, 'f6_idx': idx_f6}


print("="*80)
print("EVALUACIÓN EN CorrAL-100 (John et al. 1994)")
print("Clase = (f1 AND f2) OR (f3 AND f4)")
print("="*80)
print()

# Generar dataset
X, y, info = generate_corrAL100(n_samples=5000, seed=42)

print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} features")
print()
print("Estructura:")
print(" • f0, f1, f2, f3: RELEVANTES (participan en AND/OR)")
print(" • f4: IRRELEVANTE")
print(" • f5: CORRELACIONADA (75% con clase)")
print(f" • f6-f99: {X.shape[1]-6} features de ruido")
print()

# Verificar distribución de clase
print(f"Distribución de clase: {np.bincount(y)}")
print(f"Proporción de 1s: {np.mean(y):.2%}")
print()

results = {}

# 1. ANN
print("Evaluando métodos...")
try:
    ann = ANN(n_neighbors=20, n_features_to_select=10,
              discrete_threshold=10, random_state=42)
    ann.fit(X, y)
    ranking_ann = ann.rank()
    importances_ann = ann.feature_importances_
    
    positions_ann = {}
    for i in range(6):
        position = np.where(ranking_ann == i)[0][0] + 1
        positions_ann[f'f{i}'] = position
    
    results['ANN'] = {
        'positions': positions_ann,
        'importances': {f'f{i}': importances_ann[i] for i in range(6)}
    }
    print("  ✓ ANN evaluado")
except Exception as e:
    print(f"  ✗ ANN error: {e}")

# 2. Proto
try:
    proto = Proto(n_features_to_select=10, sigma=0.15,
                  discrete_threshold=10, k_protos=10, use_lvq=False, n_jobs=1)
    proto.fit(X, y)
    ranking_proto = proto.rank()
    importances_proto = proto.feature_importances_
    
    positions_proto = {}
    for i in range(6):
        position = np.where(ranking_proto == i)[0][0] + 1
        positions_proto[f'f{i}'] = position
    
    results['Proto'] = {
        'positions': positions_proto,
        'importances': {f'f{i}': importances_proto[i] for i in range(6)}
    }
    print("  ✓ Proto evaluado")
except Exception as e:
    print(f"  ✗ Proto error: {e}")

# 3. ReliefF
if relieff_available:
    try:
        relieff = ReliefF_Original(n_features_to_select=10, n_neighbors=20)
        relieff.fit(X, y)
        importances_relieff = relieff.feature_importances_
        ranking_relieff = np.argsort(-importances_relieff)
        
        positions_relieff = {}
        for i in range(6):
            position = np.where(ranking_relieff == i)[0][0] + 1
            positions_relieff[f'f{i}'] = position
        
        results['ReliefF'] = {
            'positions': positions_relieff,
            'importances': {f'f{i}': importances_relieff[i] for i in range(6)}
        }
        print("  ✓ ReliefF evaluado")
    except Exception as e:
        print(f"  ✗ ReliefF error: {e}")

print()
print("="*80)
print("RANKING DE FEATURES f0-f5 (posición de 1 a 100)")
print("="*80)
print()

if results:
    # Tabla de posiciones
    print("┌─────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Feature     │ ANN          │ Proto        │ ReliefF      │")
    print("├─────────────┼──────────────┼──────────────┼──────────────┤")
    
    feature_types = {
        'f0': 'RELEVANTE',
        'f1': 'RELEVANTE',
        'f2': 'RELEVANTE',
        'f3': 'RELEVANTE',
        'f4': 'IRRELEVANTE',
        'f5': 'CORRELACIONADA'
    }
    
    for i in range(6):
        feature_name = f'f{i}'
        feature_type = feature_types[feature_name]
        row = f"│ {feature_name} ({feature_type[:3]}) │"
        
        for method in ['ANN', 'Proto', 'ReliefF']:
            if method in results:
                pos = results[method]['positions'][feature_name]
                if pos <= 10:
                    marker = "✓"
                elif pos <= 30:
                    marker = "~"
                else:
                    marker = "✗"
                row += f" {marker} Pos {pos:>3} │"
            else:
                row += "      N/A      │"
        
        print(row)
    
    print("└─────────────┴──────────────┴──────────────┴──────────────┘")
    print()
    print("Leyenda:")
    print("  REL = Relevante | IRR = Irrelevante | COR = Correlacionada")
    print("  ✓ = Top 10 | ~ = Top 30 | ✗ = Fuera del top 30")
    print()
    
    # Tabla de importancias
    print("="*80)
    print("IMPORTANCIAS (valores)")
    print("="*80)
    print()
    
    print("┌─────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Feature     │ ANN          │ Proto        │ ReliefF      │")
    print("├─────────────┼──────────────┼──────────────┼──────────────┤")
    
    for i in range(6):
        feature_name = f'f{i}'
        feature_type = feature_types[feature_name]
        row = f"│ {feature_name} ({feature_type[:3]}) │"
        
        for method in ['ANN', 'Proto', 'ReliefF']:
            if method in results:
                imp = results[method]['importances'][feature_name]
                row += f" {imp:>10.6f}   │"
            else:
                row += "      N/A      │"
        
        print(row)
    
    print("└─────────────┴──────────────┴──────────────┴──────────────┘")
    print()
    
    # Análisis comparativo
    print("="*80)
    print("ANÁLISIS COMPARATIVO")
    print("="*80)
    print()
    
    for method in ['ANN', 'Proto', 'ReliefF']:
        if method not in results:
            continue
        
        print(f"{method}:")
        positions = results[method]['positions']
        
        relevantes_top10 = sum(1 for f in ['f0', 'f1', 'f2', 'f3'] if positions[f] <= 10)
        print(f"  • Features relevantes en Top 10: {relevantes_top10}/4")
        
        pos_corr = positions['f5']
        print(f"  • Posición de f5 (correlacionada): {pos_corr}")
        
        best_relevant_pos = min(positions[f] for f in ['f0', 'f1', 'f2', 'f3'])
        best_relevant = [f for f in ['f0', 'f1', 'f2', 'f3'] if positions[f] == best_relevant_pos][0]
        print(f"  • Mejor relevante: {best_relevant} en posición {best_relevant_pos}")
        
        # Mostrar todas las posiciones relevantes
        print(f"  • Posiciones relevantes: ", end='')
        for f in ['f0', 'f1', 'f2', 'f3']:
            print(f"{f}:{positions[f]}", end=' ')
        print()
        print()

else:
    print("No se pudo evaluar ningún método.")
    print()
