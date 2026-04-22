
# Imports para realizar una regresión polinomial multivariable + SymPy para salida de los datos
import numpy as np
import sympy as sp
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from models.base import PhysicalModel


class PolynomialWrapper(PhysicalModel):
    def __init__(self, degree=5, feature_names=None, scaler_X=None, scaler_y=None, **kwargs):
        super().__init__()
        self.degree = degree # Máximo grado de polinomio 
        self.feature_names = feature_names if feature_names else ["x"]
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)), # Genera la base del espacio de funciones
            # Ejemplo: degree=2, input: [x1, x2]
            # Genera: [x1, x2, x1^2, x2^2, x1*x2]
            # Include_bias = False --> No añade columna de 1s (eso lo hará Linear regressor)
            ("scaler_P", StandardScaler()), # Normalizaremos las características de la matriz de coeficientes (mejora estabilidad numérica)
            ("reg", LinearRegression()) # Ajustaremos a un modelo lineal sobre las características (transformadas)
        ])

    def fit(self, X_train_scaled, y_train_scaled):
        # Ajuste numérico en el espacio reducido
        self.model.fit(X_train_scaled, y_train_scaled.ravel())
        self._extract_equation() # Extraemos la ecuación en el espacio físico
        return self

    def predict(self, X_scaled):
        y_pred_scaled = self.model.predict(X_scaled) # Obtiene la predicción en el espacio z-score
        return y_pred_scaled.reshape(-1, 1)

    def _extract_equation(self):
        '''
        Obtención de la expresión en el espacio físico
        '''
        poly = self.model.named_steps["poly"] # Extrameos el polinomio
        scaler_P = self.model.named_steps["scaler_P"] # Extraemos el escalador
        reg = self.model.named_steps["reg"] # Extraemos el regresador 

        # Base simbólica física (x, y, z...)
        physical_vars = sp.symbols(self.feature_names)

        # Inyección de la topología estandarizada
        if self.scaler_X is not None:
            mu_X, sigma_X = self.scaler_X.mean_, self.scaler_X.scale_
            scaled_vars = [(var - mu) / sigma for var, mu, sigma in zip(physical_vars, mu_X, sigma_X)] # Transformamos a z-score las variables físicas
        else:
            scaled_vars = physical_vars

        # Construcción del polinomio con las variables transformadas
        P_terms = [] # Lista para guardar cada término del polinomio
        for powers in poly.powers_: # Para cáda término del polinomio (al ser multivariable, powers es una de listas,
            # de tal manera que tenemos, por ejemplo: [ [1, 0], [0, 1], [2, 1] ] para el caso x1 + x2 + x1^2 * x2
            term = 1
            for i, p in enumerate(powers): # i = Índice de la lista [a,b], p = elemento de la lista 
                if p > 0:
                    term *= scaled_vars[i]**p # Elevamos la correspondiente variable escalada a la potencia correspondiente
            P_terms.append(term)

        # Tenemos las variables transformadas PERO todavía nos falta por transformar los coeficientes

        # Escalamiento cruzado e inferencia lineal y'= beta_0 + \sum beta_i P'_i
        beta_0 = reg.intercept_ # Intercepción con eje Y
        betas = reg.coef_  # Coeficientes de la regresión
        mu_P = scaler_P.mean_ # Mu del escalado
        sigma_P = scaler_P.scale_ # Sigma del escalado
        
        y_scaled_expr = beta_0
        for i in range(len(P_terms)):
            if abs(betas[i]) > 1e-12: # Filtro inicial de derivadas nulas
                P_tilde = (P_terms[i] - mu_P[i]) / sigma_P[i]
                y_scaled_expr += betas[i] * P_tilde

        # Ya tenemos la expresión en el espacio z-score

        # Transformación afín al espacio físico objetivo (y): revertimos la transformación
        if self.scaler_y is not None:
            sigma_y, mu_y = self.scaler_y.scale_[0], self.scaler_y.mean_[0]
            y_physical_expr = sigma_y * y_scaled_expr + mu_y # Puedo operar de manera sencilla sobre la expresión de manera global
        else:
            y_physical_expr = y_scaled_expr

        # Expansión binomial de Newton y colapso de constantes numéricas (ruido de coma flotante)
        expanded_expr = sp.expand(y_physical_expr) # Expande la expresión con SymPy
        clean_expr = sp.Integer(0) # Iniciamos con una expresión limpia 
        
        for term in sp.Add.make_args(expanded_expr): # Separamos los términos. Iteramos para cada uno de ellos
            coeff, core = term.as_coeff_Mul() # Separamos entre coefficientes y el "cuerpo" (la variable)
            if abs(coeff) > 1e-8: # Descarte de ruido analítico post-expansión
                clean_expr += term  # Añadimos el término a la expresión siempre que no tenga coeficiente ~nulo

        self.equation = str(clean_expr.evalf(4)) # Convertimos a string con 4 cifras significativas

    def get_weights(self):
        return {
            "feature_names_out": list(self.model[0].get_feature_names_out(self.feature_names)),
            "coefficients": self.model[-1].coef_,   # sklearn LinearRegression / Ridge
            "intercept": float(self.model[-1].intercept_)
        }