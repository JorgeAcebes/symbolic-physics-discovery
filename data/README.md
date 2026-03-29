# Directorio de Datos (Datasets)

Este directorio contiene los conjuntos de datos sintéticos generados para el entrenamiento y test de los modelos.

## Estructura de archivos
- Los archivos deben usar el formato `.csv`.
- Nomenclatura: `[ley]_[muestras]_[ruido].csv` (ej: `kepler_500_n01.csv`).

## Estándar de Contenido
Cada dataset debe incluir columnas normalizadas y una columna de target con ruido inyectado $\epsilon \sim \mathcal{N}(0, \sigma^2)$.