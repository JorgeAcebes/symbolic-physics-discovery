import pysindy as ps
import numpy as np
from models.base import PhysicalModel

class PySINDyWrapper(PhysicalModel):
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        
        lib_poly = ps.PolynomialLibrary(degree=2, include_interaction=True)
        lib_custom = ps.CustomLibrary(
            library_functions=[lambda x: 1.0 / (x + 1e-8), lambda x: 1.0 / (x**2 + 1e-8)],
            function_names=[lambda x: f"(1/{x})", lambda x: f"(1/{x}^2)"]
        )
        
        try:
            combined_lib = ps.TensoredLibrary([lib_poly, lib_custom])
        except AttributeError:
            combined_lib = ps.GeneralizedLibrary([lib_poly, lib_custom])
            
        optimizer = ps.STLSQ(threshold=0.01)
        self.model = ps.SINDy(feature_library=combined_lib, optimizer=optimizer)

    def fit(self, X_train, y_train):
        # A diferencia del dataloader en PyTorch, SINDy recibe la matriz completa
        self.model.fit(X_train, t=1.0, x_dot=y_train, feature_names=self.feature_names)
        
        eqs = self.model.equations()
        best_expr = eqs[0] if eqs else "0"
        print(f"Ecuación SINDy descubierta: y = {best_expr}")
        return self

    def predict(self, X):
        # Asegura la dimensionalidad del vector columna [N, 1]
        y_pred = self.model.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        return y_pred