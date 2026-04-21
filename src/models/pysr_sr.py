# Importación de PySR, modelo base y métrica MSE
from pysr import PySRRegressor
import numpy as np
from models.base import PhysicalModel
from sklearn.metrics import mean_squared_error

class PySRWrapper(PhysicalModel):
    def __init__(self, feature_names=None, niterations=50, **kwargs): # Constructor. Definimos el número de iteraciones del algoritmo genético
        super().__init__()
        self.feature_names = feature_names
        self.total_iterations = niterations
        
        self.model = PySRRegressor(
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["inv", "square"],
            constraints={'^': (-1, 1)},
            nested_constraints={'^': {'^': 0}},
            # Definimos la base del espacio funcional, con ciertas restricciones en la operación exponencial y anidación de potencias
            elementwise_loss="loss(prediction, target) = (prediction - target)^2", # Imponemos square error como LOSS (topológicamente idéntico a MSE)
            populations=10, # Número de "poblaciones" (islas genéticas aisladas) que intercambian las ecuaciones
            maxsize=15, # Máxima longitud de nodos del árbol (nodos de la ecuación)
            model_selection="best", # Criterio de selección (explicado más abajo)
            parsimony=0.0, #Factor multiplicativo para castigar complejidad
            random_state=42, # Reproducibilidad
            deterministic=True, # Reproducibilidad
            parallelism='serial', # Reproducibilidad
            warm_start=True, # Tells fit to continue from where the last call to fit finished
            verbosity=0, # No queremos que esté printeando muchas cosas en la ventana de comandos
            **kwargs
        )

        # Model selection: criterion when selecting a final expression from
        # the list of best expression at each complexity.
        # Can be `'accuracy'`, `'best'`, or `'score'`. Default is `'best'`.
        # `'accuracy'` selects the candidate model with the lowest loss (highest accuracy).
        # `'score'` selects the candidate model with the highest score.
        # Score is defined as the negated derivative of the log-loss with
        # respect to complexity - if an expression has a much better
        # loss at a slightly higher complexity, it is preferred.
        # `'best'` selects the candidate model with the highest score
        # among expressions with a loss better than at least 1.5x the
        # most accurate model.

    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Discretización del tiempo evolutivo
        delta_n = 5
        epochs = self.total_iterations // delta_n

        # A diferencia de lo que hacíamos en GPLearn, no cojo el resultado del modelo en cada iteración,
        # pues comunicarse con Julia es lento. Nos comunicamos cada 5 iteraciones (a lo que denominamos epoch)

        # Inicialización del registro del observable macroscópico
        self.history = {"train_loss": [], "val_loss": []}

        # Fijamos rígidamente el paso de integración evolutiva
        self.model.set_params(niterations=delta_n)

        for epoch in range(epochs):
            
            # Fit interno en Julia y colapsa hacia la frontera de Pareto (balanceo entre precisión y simplicidad)   
            self.model.fit(X_train, y_train, variable_names=self.feature_names) 
            
            self.history["train_loss"].append(mean_squared_error(y_train, self.predict(X_train)))
            
            if X_val is not None and y_val is not None:
                self.history["val_loss"].append(mean_squared_error(y_val, self.predict(X_val)))

        self.equation = str(self.model.sympy()) # Extracción rigurosa del mapeo funcional en forma analítica con Sympy

            
        return self
    
    def predict(self, X):   
        y_pred = self.model.predict(X) # Predecimos el valor de y dado X empleando el mejor modelo de PySR
        return np.array(y_pred).reshape(-1, 1)
    
    def get_weights(self):
    # PySR expone el DataFrame de ecuaciones en self.model.equations_
        eqs = self.model.equations_
        return {
            "best_equation": str(self.equation),
            "all_equations": eqs[["equation", "loss", "complexity"]].to_dict(orient="records")
        }
