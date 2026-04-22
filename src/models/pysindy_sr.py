# Importaciones para PySINDY + el modelo base para la clase
import numpy as np
from pysindy.optimizers import STLSQ # Importamos Sequential Threshold Least Squares
from pysindy.feature_library import PolynomialLibrary, CustomLibrary, GeneralizedLibrary
from models.base import PhysicalModel

# Este modelo no cuenta con épocas por lo que no procede graficar el loss para el train y el validation
class PySINDyWrapper(PhysicalModel):
    def __init__(self, feature_names=None, degree=4, **kwargs):
        super().__init__()
        self.feature_names = feature_names
        self.degree = degree
        
        lib_poly = PolynomialLibrary(degree=self.degree, include_interaction=True) # Creamos la base del espacio de funciones (idéntico a polynomial + 1)
        lib_custom = CustomLibrary(
            library_functions=[lambda x: 1.0 / (x + 1e-8), lambda x: 1.0 / (x**2 + 1e-8)],
            function_names=[lambda x: f"(1/{x})", lambda x: f"(1/{x}^2)"]
        ) # Ampliamos la base del espacio de funciones añadiendo funciones racionales (1/x, 1/x^2) asegurando que no se den divergencias
        
        # Base funcional unificada
        self.library = GeneralizedLibrary([lib_poly, lib_custom])

        threshold_val = kwargs.pop('threshold', 2.4934e-02)
        self.optimizer = STLSQ(threshold=threshold_val, **kwargs) # Optimizador con L2 (Ridge) pero con un threshold en los pesos
    def fit(self, X_train, y_train):
        # Proyección sobre el espacio de características
        self.library.fit(X_train) # Ve cómo es X para saber cómo tendrán que ser las dimensiones de Theta. No actualiza nada 
        Theta = self.library.transform(X_train) # Transformamos el vector X en la base del espacio de funciones (por eso necesitaba saber dimensiones)
        
        # Optimización dispersa L_0
        self.optimizer.fit(Theta, y_train) # El optimizador resuelve el problema y = Thteta* Xi, donde Xi es la matriz de coeficientes dispersos
        
        # Recuperación de la ecuación física
        coefs = self.optimizer.coef_[0] # Extraemos el vector Xi (subíndice cero porque y podría ser un vector en vez de un escalar, y Xi entonces sería una matriz)
        feat_names = self.library.get_feature_names(self.feature_names) # Recuperamos las etiquetas simbólicas que le corresponden a cada uno
        
        # Obtenemos la ecuación eliminando los coeficientes que no sean ~0
        terms = [f"{c:.3e}*{n}" for c, n in zip(coefs, feat_names) if abs(c) > 1e-8]
        self.equation = " + ".join(terms) if terms else "0" 
        return self

    def predict(self, X):
        Theta = self.library.transform(X) # Mapeamos al espacio de funciones
        y_pred = self.optimizer.predict(Theta) # Predecimos (ya tiene internamente la matriz de pesos por el fit)
        return y_pred.reshape(-1, 1) # Reshape para después poder calcular MSE
    
    def get_weights(self):
        return {
            "feature_names": self.library.get_feature_names(self.feature_names), # Recuperamos las etiquetas simbólicas que le corresponden a cada uno
            "coefficients": self.optimizer.coef_[0] # Extraemos el vector Xi (subíndice cero porque y podría ser un vector en vez de un escalar, y Xi entonces sería una matriz)
        }

