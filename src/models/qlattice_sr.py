# Imports necesario: Feyn (QLattice) y la clase base de los modelos
import numpy as np
import pandas as pd
import feyn
from sklearn.metrics import mean_squared_error
from models.base import PhysicalModel

class QLatticeWrapper(PhysicalModel):
    def __init__(self, feature_names=None, target_name="y", epochs=10, max_complexity=7, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.max_complexity = max_complexity # Número máximo de nodos computacionales del grafo. 
        # No necesariamente debemos tener el mismo número de coplexity en cada uno, mientras que pueda servir para varios está guay (yo creo)
        self.target_name = target_name
        self.feature_names = feature_names
        self.ql = feyn.QLattice(random_seed=42) # Instancia el retículo probabilístico con semilla de reproducibilidad
        self.best_model = None 

    def _to_dataframe(self, X, y=None):
        if self.feature_names is None:
            self.feature_names = [f"x{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=self.feature_names) # Debido a cómo funciona Feyn internamente, necesitamos pasarle los nombres de las features en formato dataframes
        if y is not None: df[self.target_name] = y.ravel() # Igual para el nombre del target
        return df

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        df_train = self._to_dataframe(X_train, y_train)  # Inyectamos el train y validation rigurosamente con dataframes
        if X_val is not None and y_val is not None:
            df_val = self._to_dataframe(X_val, y_val) # Inyección rigurosa del target
            
        models = [] 
        for epoch in range(self.epochs): 
            models += self.ql.sample_models(
                df_train, self.target_name, 'regression', max_complexity=self.max_complexity
            ) # En cada época se actualiza el retículo de probabildiad. El retículo colapsa probabilisticamente 
            # en un subconjunto de modelos (grafos analíticos candidatos). Filtra el espacio buscando leyes de regresión que no superen max complexity

            # Base de funciones y el número de elementos que comen:
            # exp:1
            # gaussian:1
            # inverse:1
            # linear:1
            # log:1
            # sqrt:1
            # squared:1
            # tanh:1
            # add:2
            # gaussian:2
            # multiply:2

            # Se puede restringir el número de funciones pero no he visto cómo ampliarlo

            
            models = feyn.fit_models( 
                models, df_train, threads=4, loss_function='squared_error', criterion='bic'
            ) # Descenso de gradiente local sobre los grafos sampleados para ajustar las constantes físicas.
            # Empleamos el criterio BIC (Bayesian Information Criterion): mezcla entre residuo L2 y parsimonia de la ecuación
            models = feyn.prune_models(models) # Eliminamos subgrafos redundantes, ramas muertas o modelos idénticos.
            self.ql.update(models) # Retropropagación de los mejores grafos al QLattice. Aqueyos grafos exitosos tiene mayor probabilidad de ser muestreados en la próxima iteración.
            
            best_epoch_model = models[0] # Extraemos el mejor grafo en esta iteración
            
            # Colapso topológico a 1D para la métrica
            train_mse = mean_squared_error(y_train.ravel(), best_epoch_model.predict(df_train)) # Evaluación del MSE para el train
            self.history["train_loss"].append(train_mse)
            
            if X_val is not None and y_val is not None:
                val_mse = mean_squared_error(y_val.ravel(), best_epoch_model.predict(df_val)) # Evaluación del MSE para el validation
                self.history["val_loss"].append(val_mse)
            
        self.best_model = models[0] # Guardamos el mejor grafo tras todas las iteraciones
        self.equation = str(self.best_model.sympify(signif=4)) # Convertimos el modelo en una ecuación de sympy y limitamos a 4 cifras significativas
        return self

    def predict(self, X):
        df_test = self._to_dataframe(X) 
        return np.array(self.best_model.predict(df_test)).reshape(-1, 1) # Realizamos la evaluación del test con el mejor modelo
    
    def get_weights(self):
        # feyn expone los parámetros del modelo como diccionario
        return {
            "equation":  str(self.best_model.sympify(signif=4)),
        }