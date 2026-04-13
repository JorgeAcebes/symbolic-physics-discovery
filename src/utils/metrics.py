import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_physical_space(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae