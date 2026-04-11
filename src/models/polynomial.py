import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from models.base import PhysicalModel

class PolynomialWrapper(PhysicalModel):
    def __init__(self, degree=3, feature_names=None, scaler_y=None):
        super().__init__()
        self.degree = degree
        self.feature_names = feature_names if feature_names else ["x"]
        self.scaler_y = scaler_y
        
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler_X", StandardScaler()),
            ("reg", LinearRegression())
        ])

    def fit(self, X_train_unscaled, y_train_scaled):
        # Ajustamos sobre X en unidades físicas y y escalado para estabilidad
        self.model.fit(X_train_unscaled, y_train_scaled.ravel())
        self._extract_equation()
        return self

    def predict(self, X_unscaled):
        y_pred_scaled = self.model.predict(X_unscaled)
        return y_pred_scaled.reshape(-1, 1)

    def _extract_equation(self):
        poly = self.model.named_steps["poly"]
        scaler_X = self.model.named_steps["scaler_X"]
        reg = self.model.named_steps["reg"]

        feature_names = poly.get_feature_names_out(self.feature_names)

        # Parámetros estadísticos
        sigma_y = self.scaler_y.scale_[0] if self.scaler_y else 1.0
        mu_y = self.scaler_y.mean_[0] if self.scaler_y else 0.0
        sigma_P = scaler_X.scale_
        mu_P = scaler_X.mean_
        beta = reg.coef_
        beta_0 = reg.intercept_

        # Proyección al espacio físico
        c = (sigma_y * beta) / sigma_P
        c_0 = sigma_y * beta_0 + mu_y - np.sum(c * mu_P)

        terms = [f"{coef:.3e}*{name}" for coef, name in zip(c, feature_names) if abs(coef) > 1e-10]
        equation = " + ".join(terms) + f" {'+' if c_0 >= 0 else '-'} {abs(c_0):.3e}"
        self.equation = equation