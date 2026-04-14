# Symbolic Physics Discovery — Web Interface

Interfaz web para lanzar experimentos sin tocar el código fuente.

## Estructura

```
web/
├── app.py          ← Servidor Flask (API + HTML)
├── runner.py       ← Adaptador entre la web y src/
├── templates/
│   └── index.html  ← UI principal
├── static/
│   ├── style.css
│   └── app.js
├── uploads/        ← Archivos subidos temporalmente  [gitignored]
├── outputs/        ← Resultados de cada ejecución    [gitignored]
└── README.md
```

## Arrancar el servidor

```bash
cd symbolic-physics-discovery/web
python app.py
```

Abre http://localhost:5050 en el navegador.

## Flujo de uso

1. **Datos** — elige un CSV de la carpeta `data/` o sube un CSV/JSON/TXT externo.
   - Si el archivo externo no tiene cabeceras reconocibles, el configurador de columnas
     te pedirá que nombres cada columna y marques cuál es el target.

2. **Modelos** — haz clic en los modelos que quieras ejecutar (puedes elegir varios).

3. **Opciones avanzadas** *(opcional)* — despliega para cambiar hiperparámetros.
   Estos cambios **solo afectan a la ejecución actual**, no modifican ningún archivo.

4. **Lanzar** — el servidor ejecuta los modelos en segundo plano.
   El log aparece en tiempo real en el terminal de la interfaz.

5. **Resultados** — al finalizar aparecen:
   - Tarjetas con MSE / MAE y la ecuación descubierta por cada modelo.
   - Galería de gráficas de residuos y curvas de convergencia.
   - Botones de descarga para cada fichero.

## Aislamiento de outputs

Todos los archivos generados van a `web/outputs/<job_id>/`. Nunca se escribe
en `results/`, `data/` ni ninguna otra carpeta del proyecto.

Cada ejecución tiene su propio subdirectorio con UUID, por lo que los resultados
de distintas corridas no se machan entre sí.
