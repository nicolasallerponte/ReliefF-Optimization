import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuración de imports
try:
    current = Path(__file__).parent.resolve()
    sys.path.insert(0, str(current / 'src')) 
    from relieff_improved.ann.ann_v2_optimized import ANN
    from relieff_improved.proto.proto_v2_optimized import Proto
    from skrebate import ReliefF
except ImportError:
    sys.exit("Error importando librerías.")

# =============================================================================
# GENERADOR (Escalable en muestras)
# =============================================================================
def generate_corral_dynamic(n_samples, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Relevantes
    X_rel = np.random.randint(0, 2, (n_samples, 4))
    df = pd.DataFrame(X_rel, columns=['f1', 'f2', 'f3', 'f4'])
    
    # 2. Target
    y = ((df['f1'] == 1) & (df['f2'] == 1)) | ((df['f3'] == 1) & (df['f4'] == 1))
    y = y.astype(int).values
    
    # 3. Irrelevante (f5)
    f5 = np.random.randint(0, 2, n_samples)
    
    # 4. Correlacionada (f6) - 75% match
    mask = np.random.rand(n_samples) < 0.75
    f6 = np.where(mask, y, 1 - y)
    
    X = np.column_stack((X_rel, f5, f6))
    return X, y

# =============================================================================
# EXPERIMENTO DE CURVA DE APRENDIZAJE
# =============================================================================
def learning_curve_experiment():
    # Probamos tamaños desde el original (32) hasta algo razonable (1000)
    sample_sizes = [32, 64, 128, 256, 512, 1024]
    n_iterations = 10
    
    results = []
    
    print("Iniciando Curva de Aprendizaje...")
    print(f"{'Muestras':<10} | {'Método':<10} | {'Rank f6':<10} | {'Acierto?'}")
    
    for n in sample_sizes:
        for i in range(n_iterations):
            X, y = generate_corral_dynamic(n, seed=i*10)
            
            # Ajustamos k-vecinos dinámicamente según el tamaño de muestra
            # ReliefF falla si k > n_samples. Usamos k=10 o n-1
            k_neighbors = min(10, n - 1)
            
            methods = {
                'ReliefF': ReliefF(n_features_to_select=6, n_neighbors=k_neighbors, n_jobs=-1),
                'ANN': ANN(),
                'Proto': Proto()
            }
            
            for name, model in methods.items():
                model.fit(X, y)
                
                # Obtener ranking
                w = model.feature_importances_
                ranks = np.argsort(w)[::-1] # Índices de mayor a menor peso
                
                # Buscamos en qué posición (0 a 5) quedó f6 (índice 5 en el array original)
                # OJO: f1..f4 son indices 0..3, f5 es 4, f6 es 5.
                
                # Convertimos ranks a un mapa: {indice_feature: posicion_ranking}
                rank_map = {feat_idx: rank_pos for rank_pos, feat_idx in enumerate(ranks)}
                
                rank_f6 = rank_map[5] + 1 # +1 para que sea posición humana (1º, 2º...)
                
                # ¿Éxito? Consideramos éxito si f6 NO es la #1.
                # Mejor aún: Éxito estricto es si f6 > 4 (está en cola).
                success = 1 if rank_f6 > 4 else 0
                
                results.append({
                    'Samples': n,
                    'Method': name,
                    'Rank_f6': rank_f6,
                    'Success': success
                })
        
        # Log rápido al terminar un tamaño
        print(f"Completado N={n}")

    return pd.DataFrame(results)

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
def plot_learning_curve(df):
    plt.figure(figsize=(10, 6))
    
    # Graficar el Ranking Promedio de f6 a medida que aumentan las muestras
    # Queremos que la línea SUBA (que el ranking de f6 sea peor, ej. 5º o 6º)
    sns.lineplot(
        data=df, 
        x='Samples', 
        y='Rank_f6', 
        hue='Method', 
        marker='o',
        linewidth=2
    )
    
    plt.title("¿Cuándo descubren la trampa? (Ranking de f6 vs Tamaño de Muestra)")
    plt.ylabel("Posición de f6 (Mayor es mejor, 6º es ideal)")
    plt.xlabel("Número de Muestras (Escala Log)")
    plt.xscale('log')
    plt.xticks([32, 64, 128, 256, 512, 1024], [32, 64, 128, 256, 512, 1024])
    
    # Zonas de referencia
    plt.axhspan(0.5, 1.5, color='red', alpha=0.1, label='Zona de Trampa (Top 1)')
    plt.axhspan(4.5, 6.5, color='green', alpha=0.1, label='Zona de Éxito (Cola)')
    
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('corral_learning_curve.png')
    print("Gráfico guardado: corral_learning_curve.png")

if __name__ == "__main__":
    df = learning_curve_experiment()
    plot_learning_curve(df)