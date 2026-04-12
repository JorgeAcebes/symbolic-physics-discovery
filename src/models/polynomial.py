import numpy as np
import sympy as sp
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from models.base import PhysicalModel

class PolynomialWrapper(PhysicalModel):
    def __init__(self, degree=3, feature_names=None, scaler_X=None, scaler_y=None):
        super().__init__()
        self.degree = degree
        self.feature_names = feature_names if feature_names else ["x"]
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler_P", StandardScaler()), # Condicionamiento de las variables cruzadas
            ("reg", LinearRegression())
        ])

    def fit(self, X_train_scaled, y_train_scaled):
        # 1. Ajuste estrictamente numérico en el espacio reducido
        self.model.fit(X_train_scaled, y_train_scaled.ravel())
        self._extract_equation()
        return self

    def predict(self, X_scaled):
        y_pred_scaled = self.model.predict(X_scaled)
        return y_pred_scaled.reshape(-1, 1)

    def _extract_equation(self):
        poly = self.model.named_steps["poly"]
        scaler_P = self.model.named_steps["scaler_P"]
        reg = self.model.named_steps["reg"]

        # 1. Base simbólica física (x, y, z...)
        physical_vars = sp.symbols(self.feature_names)

        # 2. Inyección de la topología estandarizada
        if self.scaler_X is not None:
            mu_X, sigma_X = self.scaler_X.mean_, self.scaler_X.scale_
            scaled_vars = [(var - mu) / sigma for var, mu, sigma in zip(physical_vars, mu_X, sigma_X)]
        else:
            scaled_vars = physical_vars

        # 3. Construcción del espacio polinómico P(x')
        P_terms = []
        for powers in poly.powers_:
            term = 1
            for i, p in enumerate(powers):
                if p > 0:
                    term *= scaled_vars[i]**p
            P_terms.append(term)

        # 4. Escalamiento cruzado e inferencia lineal y'= beta_0 + \sum beta_i P'_i
        beta_0 = reg.intercept_
        betas = reg.coef_
        mu_P = scaler_P.mean_
        sigma_P = scaler_P.scale_
        
        y_scaled_expr = beta_0
        for i in range(len(P_terms)):
            if abs(betas[i]) > 1e-12: # Filtro inicial de derivadas nulas
                P_tilde = (P_terms[i] - mu_P[i]) / sigma_P[i]
                y_scaled_expr += betas[i] * P_tilde

        # 5. Transformación afín al espacio físico objetivo (y)
        if self.scaler_y is not None:
            sigma_y, mu_y = self.scaler_y.scale_[0], self.scaler_y.mean_[0]
            y_physical_expr = sigma_y * y_scaled_expr + mu_y
        else:
            y_physical_expr = y_scaled_expr

        # 6. Expansión binomial de Newton y colapso de constantes numéricas (ruido de coma flotante)
        expanded_expr = sp.expand(y_physical_expr)
        clean_expr = sp.Integer(0)
        
        for term in sp.Add.make_args(expanded_expr):
            coeff, core = term.as_coeff_Mul()
            if abs(coeff) > 1e-8: # Descarte de ruido analítico post-expansión
                clean_expr += term

        self.equation = str(clean_expr.evalf(4))