# Importación de PySR, modelo base y métrica MSE
from pysr import PySRRegressor
import numpy as np
from models.base import PhysicalModel
from sklearn.metrics import mean_squared_error

class PySRWrapper(PhysicalModel):
    def __init__(self, feature_names=None, niterations=50): # Constructor. Definimos el número de iteraciones del algoritmo genético
        super().__init__()
        self.feature_names = feature_names
        
        self.model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["inv", "square"],
            constraints={'^': (-1, 1)},
            nested_constraints={'^': {'^': 0}},
            # Definimos la base del espacio funcional, con ciertas restricciones en la operación exponencial y anidación de potencias
            elementwise_loss="loss(prediction, target) = (prediction - target)^2", # Imponemos norma L2 para evaluar el fitness 
            populations=10, # Número de "poblaciones" (islas genéticas aisladas) que intercambian las ecuaciones
            maxsize=15, # Máxima longitud de nodos del árbol (nodos de la ecuación)
            model_selection="best", # Criterio de selección (explicado más abajo)
            random_state=42, # Reproducibilidad
            verbosity=0 # No queremos que esté printeando muchas cosas en la ventana de comandos
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
        self.model.fit(X_train, y_train, variable_names=self.feature_names)
        # Fit interno en Julia y colapsa hacia la frontera de Pareto (balanceo entre precisión y simplicidad)   

        # Extracción rigurosa del mapeo funcional en forma analítica con Sympy
        self.equation = str(self.model.sympy()) 


        # TODO: ESTO NO FUNCIONA. Revisar si PySR tiene internamente para obtener los loss en cada momento o no. Si no lo tiene: lo omitimos

        # # Evaluación determinista del modelo convergido
        # train_loss = mean_squared_error(y_train, self.predict(X_train))
        # self.history["train_loss"] = [train_loss]
        
        # # Proyección sobre el subespacio de validación
        # if X_val is not None and y_val is not None:
        #     val_loss = mean_squared_error(y_val, self.predict(X_val))
        #     self.history["val_loss"] = [val_loss]
            
        return self

    def predict(self, X):   
        y_pred = self.model.predict(X) # Predecimos el valor de y dado X empleando el mejor modelo de PySR
        return np.array(y_pred).reshape(-1, 1)