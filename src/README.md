# Código Fuente (Core Logic)

Contiene el motor algorítmico del proyecto. Se prohíbe el uso de scripts "sueltos"; todo debe estar modularizado.

## Módulos Principales
- `physics.py`: Implementación de las leyes físicas y generación de datos.
- `models.py`: Definición de clases para el regresor `PySR` y la arquitectura `MLP`.
- `utils.py`: Configuración de estilo de plots y métricas de error ($MSE$, $R^2$, Pareto Score).

## Reglas de Oro
1. Seguir `snake_case` para funciones y variables.
2. Comentar los parámetros de entrada y salida (docstrings).