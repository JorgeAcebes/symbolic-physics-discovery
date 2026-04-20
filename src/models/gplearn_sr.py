# Importaciones para GPLearn + el modelo base para la clase
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error
from models.base import PhysicalModel
import sympy as sp


# If you want to play with the hyperparameters, the main ones to modify are:
# generations, pop size, p_crossover, p_subtree, p_hoist, p_point

class GPLearnWrapper(PhysicalModel):
    # Definimos las condiciones de contorno del algoritmo genético, por Default: 2000 ecuaciones "compitiendo" durante 30 ciclos
    def __init__(self, feature_names=None, generations=30, population_size=2000): 
        super().__init__()
        self.model = SymbolicRegressor(
            population_size=population_size, generations=generations,
            warm_start=True, function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'inv'), # Base del espacio de funciones
            metric='mse', # Función de coste: mean square error
            p_crossover=0.65, # Probabilidad de mezclar dos fórmulas
            p_subtree_mutation=0.1, # Probabilidad de cambiar una parte de una fórmula por otro
            p_hoist_mutation=0.15, # Probabilidad de simplificar una parte de la fórmula
            p_point_mutation=0.05, # Probabilidad de cambiar nodos individuales (e.g. cambiar sin por cos)
            # La suma de las probabilidades debe ser inferior a 1
            n_jobs=-1, # Usar todos los núcleos del procesador
            random_state=42, # Para reproducibilidad
            parsimony_coefficient=0.1, # Para reducir complejidad
            feature_names=feature_names # Nombre de las features
        )



    def fit(self, X_train, y_train, X_val=None, y_val=None):
            total_generations = self.model.generations
            
            for gen in range(1, total_generations + 1): 
                self.model.set_params(generations=gen) # Esto me permite modificar parámetros de un modelo sin instanciarlo de nuevo;
                # en particular, estoy modificando el número de generaciones del modelo. 

                self.model.fit(X_train, y_train.ravel())  # Ravel: similar a flatten pero más eficiente. Aquí ya es cuando fiteamos el modelo.
                
                # Calculamos nostros de manera explícita el MAE del train y del validation.

                self.history["train_loss"].append(mean_squared_error(y_train, self.predict(X_train)))
                if X_val is not None and y_val is not None:
                    self.history["val_loss"].append(mean_squared_error(y_val, self.predict(X_val)))

            raw_equation = str(self.model._program)

            # Diccionario de transformaciones para SymPy:
            locals_dict = {'add': lambda x, y: x + y, 'sub': lambda x, y: x - y,
                        'mul': lambda x, y: x * y, 'div': lambda x, y: x / y,
                        'inv': lambda x: 1/x, 'sin': sp.sin, 'cos': sp.cos}
            try: 
                self.equation = sp.simplify(sp.sympify(raw_equation, locals=locals_dict)) # Convertimos de GPLearn a Sympy y SIMPLIFICAMOS
            except:
                self.equation = raw_equation  # Si no conseguimos convertir a sympy, que nos dé la ecuación que obtiene GPLearn

            return self


    def predict(self, X):
        # Llama al modelo entrenado y devuelve las predicciones para cada muestra de X
        return self.model.predict(X).reshape(-1, 1) 

    def get_weights(self):
        return {
            "best_program": str(self.model._program),
            "equation": str(self.equation)
        }