# Importaciones para GPLearn + el modelo base para la clase
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error
from models.base import PhysicalModel


# TODO: If you want to play with the hyperparameters, the main ones to modify are:
# generations, pop size, p_crossover, p_subtree, p_hoist, p_point

class GPLearnWrapper(PhysicalModel):
    # Definimos las condiciones de contorno del algoritmo genético, por Default: 2000 ecuaciones "compitiendo" durante 30 ciclos
    def __init__(self, feature_names=None, generations=30, population_size=2000): 
        super().__init__()
        self.model = SymbolicRegressor(
            population_size=population_size, generations=generations,
            warm_start=True, function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'inv'), # Base del espacio de funciones
            metric='mse', # Función de coste: mean square error
            p_crossover=0.7, # Probabilidad de mezclar dos fórmulas
            p_subtree_mutation=0.1, # Probabilidad de cambiar una parte de una fórmula por otro
            p_hoist_mutation=0.2, # Probabilidad de simplificar una parte de la fórmula
            p_point_mutation=0.1, # Probabilidad de cambiar nodos individuales (e.g. cambiar sin por cos)
            n_jobs=-1, # Usar todos los núcleos del procesador
            random_state=42, # Para reproducibilidad
            feature_names=feature_names # Nombre de las features
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train.ravel()) # Ravel: similar a flatten pero más eficiente. 
        # Realizamos el fit que tiene implementado el modelo de GPLearn
        self.equation = str(self.model._program) # Obtenemos la función ganadora y la proyectamos en una cadena de texto

        # TODO: Estaría guay que a posteriori el resultado estuviese simplificado y legible para sympy...


        # Recuperación de la dinámica de convergencia.
        if hasattr(self.model, 'run_details_') and 'best_fitness' in self.model.run_details_:
            self.history["train_loss"] = self.model.run_details_['best_fitness']
                        
        return self

    def predict(self, X):
        # Llama al modelo entrenado y devuelve las predicciones para cada muestra de X
        return self.model.predict(X).reshape(-1, 1) 