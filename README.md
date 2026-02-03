# ReliefF Mejorado - TFG

## Descripción

Este proyecto implementa mejoras al algoritmo ReliefF para selección de características, explorando dos enfoques principales:

- **ANN-based**: Utilizando Approximate Nearest Neighbors (HNSW)
- **Prototype-based**: Utilizando clustering y prototipos (LVQ)

## Instalación

### Requisitos previos

- Python 3.10+
- uv (gestor de paquetes)

### Configuración del entorno

```bash
# Clonar el repositorio
git clone https://github.com/tuusuario/relieff-tfg.git
cd relieff-tfg

# Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear entorno e instalar dependencias
uv venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```
