import numpy as np
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary, CustomLibrary, GeneralizedLibrary
from models.base import PhysicalModel

class PySINDyWrapper(PhysicalModel):
    def __init__(self, feature_names=None, degree=2):
        super().__init__()
        self.feature_names = feature_names
        self.degree = degree
        
        lib_poly = PolynomialLibrary(degree=self.degree, include_interaction=True)
        lib_custom = CustomLibrary(
            library_functions=[lambda x: 1.0 / (x + 1e-8), lambda x: 1.0 / (x**2 + 1e-8)],
            function_names=[lambda x: f"(1/{x})", lambda x: f"(1/{x}^2)"]
        )
        
        # Base funcional unificada
        self.library = GeneralizedLibrary([lib_poly, lib_custom])
        self.optimizer = STLSQ(threshold=0.01)

    def fit(self, X_train, y_train):
        # Proyección sobre el espacio de características
        self.library.fit(X_train)
        Theta = self.library.transform(X_train)
        
        # Optimización dispersa L_0
        self.optimizer.fit(Theta, y_train)
        
        # Recuperación de la ecuación física
        coefs = self.optimizer.coef_[0]
        feat_names = self.library.get_feature_names(self.feature_names)
        
        terms = [f"{c:.3e}*{n}" for c, n in zip(coefs, feat_names) if abs(c) > 0]
        self.equation = " + ".join(terms) if terms else "0"
        return self

    def predict(self, X):
        Theta = self.library.transform(X)
        y_pred = self.optimizer.predict(Theta)
        return y_pred.reshape(-1, 1)