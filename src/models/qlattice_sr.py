import numpy as np
import pandas as pd
import feyn
from sklearn.metrics import mean_squared_error
from models.base import PhysicalModel

class QLatticeWrapper(PhysicalModel):
    def __init__(self, feature_names=None, target_name="y", epochs=10, max_complexity=7):
        super().__init__()
        self.epochs = epochs
        self.max_complexity = max_complexity
        self.target_name = target_name
        self.feature_names = feature_names
        self.ql = feyn.QLattice(random_seed=42)
        self.best_model = None

    def _to_dataframe(self, X, y=None):
        if self.feature_names is None:
            self.feature_names = [f"x{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=self.feature_names)
        if y is not None: df[self.target_name] = y.ravel()
        return df

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        df_train = self._to_dataframe(X_train, y_train)
        if X_val is not None and y_val is not None:
            df_val = self._to_dataframe(X_val)
            
        models = []
        for epoch in range(self.epochs):
            models += self.ql.sample_models(
                df_train, self.target_name, 'regression', max_complexity=self.max_complexity
            )
            models = feyn.fit_models(
                models, df_train, threads=4, loss_function='squared_error', criterion='bic'
            )
            models = feyn.prune_models(models)
            self.ql.update(models)
            
            # Evaluación del mejor sub-grafo de la época actual
            best_epoch_model = models[0]
            
            train_mse = mean_squared_error(y_train, best_epoch_model.predict(df_train))
            self.history["train_loss"].append(train_mse)
            
            if X_val is not None and y_val is not None:
                val_mse = mean_squared_error(y_val, best_epoch_model.predict(df_val))
                self.history["val_loss"].append(val_mse)
            
        self.best_model = models[0]
        self.equation = str(self.best_model.sympify(signif=4))
        return self

    def predict(self, X):
        df_test = self._to_dataframe(X)
        return np.array(self.best_model.predict(df_test)).reshape(-1, 1)