from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error
from models.base import PhysicalModel

class GPLearnWrapper(PhysicalModel):
    def __init__(self, generations=30, population_size=2000):
        super().__init__()
        self.model = SymbolicRegressor(
            population_size=population_size, generations=generations,
            warm_start=True, function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos'),
            metric='mse', p_crossover=0.7, p_subtree_mutation=0.1,
            p_hoist_mutation=0.05, p_point_mutation=0.1, n_jobs=-1, random_state=42
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train.ravel())
        self.equation = str(self.model._program)
        
        # Recuperación de la dinámica de convergencia en espacio estandarizado
        if hasattr(self.model, 'run_details_') and 'best_fitness' in self.model.run_details_:
            self.history["train_loss"] = self.model.run_details_['best_fitness']
            
        if X_val is not None and y_val is not None:
            # Dado que GPLearn no itera el val_loss intrínsecamente, evaluamos
            # el modelo convergido como baseline.
            final_val_loss = mean_squared_error(y_val.ravel(), self.model.predict(X_val))
            self.history["val_loss"] = [final_val_loss] * len(self.history["train_loss"])
            
        return self

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)