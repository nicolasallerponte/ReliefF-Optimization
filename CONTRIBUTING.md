# Guía de contribución

¡Gracias por tu interés en mejorar este proyecto! Estas pautas buscan que contribuir sea sencillo
y predecible.

## Cómo empezar

1. Haz un *fork* del repositorio y clónalo.
2. Crea el entorno de desarrollo:
   ```bash
   uv sync --extra dev      # o: pip install -e ".[dev]"
   ```
3. Crea una rama descriptiva para tu cambio:
   ```bash
   git checkout -b fix/descripcion-corta
   ```

## Flujo de trabajo

- Mantén cada *pull request* centrado en un único cambio.
- Escribe mensajes de *commit* claros y en imperativo (p. ej. «Añade test para Proto en datos
  desbalanceados»).
- Si el cambio afecta al comportamiento, actualiza el `README.md` y, si procede, la
  configuración en `config/hiperparametros.yaml`.
- Asegúrate de que el código pasa las comprobaciones antes de abrir el PR.

## Estilo de código

- Python ≥ 3.10, siguiendo [PEP 8](https://peps.python.org/pep-0008/).
- Nombres, comentarios y *docstrings* en español, en coherencia con el resto del proyecto.
- Cada módulo o experimento nuevo debe documentar su propósito en un *docstring* de cabecera.

## Tests

```bash
uv run pytest
```

Acompaña los cambios de lógica con pruebas que cubran el caso nuevo. Para los experimentos,
verifica que se ejecutan de principio a fin sobre un conjunto pequeño.

## Añadir un experimento

Los experimentos viven en `experimentos/` con el patrón `NN_nombre.py`, son **independientes** y
guardan sus salidas en `results/`. Reutiliza las utilidades de `src/relieff_opt/utils/`
(runner multi-semilla, métricas, paleta de colores) para mantener la coherencia.

## Reporte de errores

Abre un *issue* indicando:

- versión de Python y del sistema operativo,
- pasos para reproducir el problema,
- comportamiento esperado frente al observado y la traza de error si la hay.

## Licencia

Al contribuir, aceptas que tu aportación se distribuya bajo la licencia **MIT** del proyecto.
