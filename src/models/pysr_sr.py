import numpy as np
from models.base import PhysicalModel
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error

class PySRWrapper(PhysicalModel):
    def __init__(self, feature_names=None, niterations=50):
        super().__init__()
        self.feature_names = feature_names
        
        self.model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["inv", "square"],
            constraints={'^': (-1, 1)},
            nested_constraints={'^': {'^': 0}},
            elementwise_loss="loss(prediction, target) = (prediction - target)^2",
            populations=10,
            maxsize=15,
            model_selection="best",
            verbosity=1
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train, variable_names=self.feature_names)
        
        # Extracción rigurosa del mapeo funcional en forma analítica
        self.equation = str(self.model.sympy())
        
        # Evaluación determinista del modelo convergido
        train_loss = mean_squared_error(y_train, self.predict(X_train))
        self.history["train_loss"] = [train_loss]
        
        # Proyección sobre el subespacio de validación
        if X_val is not None and y_val is not None:
            val_loss = mean_squared_error(y_val, self.predict(X_val))
            self.history["val_loss"] = [val_loss]
            
        return self

    def predict(self, X):   
        y_pred = self.model.predict(X)
        return np.array(y_pred).reshape(-1, 1)