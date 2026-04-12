from gplearn.genetic import SymbolicRegressor
from models.base import PhysicalModel

class GPLearnWrapper(PhysicalModel):
    def __init__(self, generations=30, population_size=2000):
        super().__init__()
        self.model = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            warm_start=True,
            function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos'),
            metric='mse',
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            n_jobs=-1,
            random_state=42
        )

    def fit(self, X_train, y_train):
        # GPLearn exige vectores unidimensionales [N,] para y
        self.model.fit(X_train, y_train.ravel())
        self.equation = str(self.model._program)
        
        if hasattr(self.model, 'run_details_') and 'best_fitness' in self.model.run_details_:
            self.history["train_loss"] = self.model.run_details_['best_fitness']
            
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        # Retornamos vector columna [N, 1] por consistencia matricial
        return y_pred.reshape(-1, 1)