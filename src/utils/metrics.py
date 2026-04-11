import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Asegura el reescalado de los datos para calcular mse y mae

def evaluate_physical_space(model, X, y_true, scaler_y=None):
    y_pred = model.predict(X)
    
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1))
        
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    return mse, mae