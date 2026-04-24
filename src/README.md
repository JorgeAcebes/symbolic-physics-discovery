# Arquitectura y Pipeline de Modelos (`src/`)

Este directorio contiene el pipeline de experimentación para el descubrimiento de leyes físicas a partir de datos estocásticos.

## 1. Estructura del Directorio

* **`data/`**: Ingesta y preprocesamiento.
    * `loader.py`: Contiene `PhysicalDataset`. Es la única fuente de verdad. Gestiona las particiones de datos y la transformación estadística estandarizada de $\mathbf{X}$ e $\mathbf{y}$. Extrae dinámicamente la base vectorial $\mathcal{B} = \{x_1, \dots, x_D\}$.
* **`models/`**: Lógica de regresión y *Wrappers*.
    * `base.py`: Define el contrato matemático abstracto `PhysicalModel`.
    * `mlp.py`, `pysindy_sr.py`, `qlattice_sr.py`, `gplearn_sr`, `polynomial.py`, ...: Implementaciones aisladas. Encapsulan las asimetrías de las distintas librerías subyacentes (PyTorch, PySINDy, Feyn, GPLearn, ...) para exponer una interfaz uniforme.
* **`utils/`**: Evaluación y persistencia.
    * `metrics.py`: Calcula el Error Cuadrático Medio ($MSE$) y Absoluto ($MAE$) forzando la inversión analítica de los datos al espacio físico original.
    * `io.py`: Estandariza la exportación de la forma funcional descubierta y las curvas de convergencia a disco.
* **`main.py`**: Orquestador. Itera sobre los conjuntos de datos, instancia el espacio dimensional, alimenta a los modelos y registra los resultados.

---

## 2. Guía de Integración: Cómo añadir un nuevo modelo

Para introducir un nuevo algoritmo de regresión y preservar el rigor comparativo, debes encapsular su comportamiento matemáticamente mediante un *Wrapper*. El modelo debe ser capaz de mapear $\hat{f}: \mathbb{R}^D \to \mathbb{R}$.

Sigue estos 4 pasos estrictos:

### Paso 1: Crear el Wrapper

Crea un archivo `src/models/tu_modelo.py`. Importa la clase abstracta y define tu clase heredando de ella.

```python
import numpy as np
from models.base import PhysicalModel
# Importa tu librería aquí

class MiNuevoModeloWrapper(PhysicalModel):
    def __init__(self, hyperparametro=10, feature_names=None):
        super().__init__()
        self.feature_names = feature_names
        self.model = # Instancia de tu algoritmo
```

### Paso 2: Implementar el ajuste (fit)
Debes sobreescribir el método fit. Es obligatorio que, antes del `return self`, extraigas la ecuación descubierta por el modelo (en formato string) y la asignes a `self.equation`. Si el modelo tiene historial de pérdida (p. ej. pérdida estocástica por épocas), guárdalo en `self.history["train_loss"]`.


```python

def fit(self, X_train, y_train):
        # Ajuste del modelo sobre la matriz X y el vector y
        self.model.fit(X_train, y_train)
        
        # 1. Recuperar la ecuación funcional
        self.equation = self.model.get_equation_string() # Reemplaza por el método real
        
        # 2. (Opcional) Guardar historial si optimiza iterativamente
        # self.history["train_loss"] = self.model.loss_curve_
        
        return self
```


### Paso 3: Implementar la predicción (predict)
Sobreescribe el método `predict`. Es crítico que el retorno sea un array de NumPy con dimensiones `[N, 1]` (vector columna) para que el broadcasting en el espacio de evaluación físico no falle al calcular el residuo $\mathbf{y} - \mathbf{\hat{y}}$.

```python
def predict(self, X):
    y_pred = self.model.predict(X)
    
    # Asegurar dimensionalidad
    return np.array(y_pred).reshape(-1, 1)
```

### Paso 4: Inyectar el modelo en el Orquestador
Abre `src/main.py.` Importa tu nuevo wrapper y añádelo al diccionario `models` dentro de la función `run_all_experiments()`.

```python

from models.tu_modelo import MiNuevoModeloWrapper

# ... [dentro de run_all_experiments] ...
        models = {
            "MLP": MLPWrapper(input_dim=X_train.shape[1], epochs=50),
            "PySINDy": PySINDyWrapper(feature_names=dataset.feature_names),
            # Inyección del nuevo modelo pasando los nombres de las variables
            "MiModelo": MiNuevoModeloWrapper(feature_names=dataset.feature_names) 
        }

```
