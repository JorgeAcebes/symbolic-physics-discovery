import numpy as np
import pandas as pd
import feyn
from models.base import PhysicalModel


class QLatticeWrapper(PhysicalModel):
    def __init__(self, feature_names=None, target_name="y", epochs=10, max_complexity=7):
        self.epochs = epochs
        self.max_complexity = max_complexity
        self.target_name = target_name
        self.feature_names = feature_names
        
        self.ql = feyn.QLattice(random_seed=42)
        self.best_model = None

    def _to_dataframe(self, X, y=None):
        """Conversor interno para aislar la dependencia de pandas del pipeline general."""
        if self.feature_names is None:
            self.feature_names = [f"x{i}" for i in range(X.shape[1])]
            
        df = pd.DataFrame(X, columns=self.feature_names)
        if y is not None:
            df[self.target_name] = y.ravel()
        return df

    def fit(self, X_train, y_train):
        df_train = self._to_dataframe(X_train, y_train)
        
        models = []
        for epoch in range(self.epochs):
            models += self.ql.sample_models(
                df_train, 
                self.target_name, 
                'regression', 
                max_complexity=self.max_complexity
            )
            models = feyn.fit_models(
                models, 
                df_train, 
                threads=4, 
                loss_function='squared_error', 
                criterion='bic'
            )
            models = feyn.prune_models(models)
            self.ql.update(models)
            
        self.best_model = models[0]
        self.equation = str(self.best_model.sympify(signif=4))
        print(f"[QLattice] Ecuación: {self.best_model.sympify(signif=4)}")
        return self

    def predict(self, X):
        df_test = self._to_dataframe(X)
        y_pred = self.best_model.predict(df_test)
        
        # Feyn puede devolver pd.Series, forzamos NumPy array bidimensional
        return np.array(y_pred).reshape(-1, 1)