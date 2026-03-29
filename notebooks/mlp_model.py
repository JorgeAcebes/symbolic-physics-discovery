"""
Modelo MLP (red neuronal fully connected) para ajustar leyes físicas
usando los datasets generados en /data (formato CSV)

@author: jfcte
"""

# =========================
# IMPORTS
# =========================

import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# =========================
# RUTAS (IMPORTANTE)
# =========================

# Carpeta donde está este script (redes_neuronales/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde están los datos: symbolic-physics-discovery/data/
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))


# =========================
# CARGA DE DATOS
# =========================

def load_dataset(filename):
    """
    Carga un dataset CSV desde la carpeta data
    """

    full_path = os.path.join(DATA_DIR, filename)

    print(f"Cargando: {full_path}")

    # Usamos pandas para leer CSV con headers
    data = pd.read_csv(full_path)

    return data


# =========================
# MODELO MLP
# =========================

def run_mlp(X, y, model_name):
    """
    Entrena una red neuronal fully connected (MLP)
    y grafica la pérdida vs épocas
    """

    import matplotlib.pyplot as plt

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # Escalado
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Modelo MLP
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        warm_start=True,   # IMPORTANTE para entrenar por épocas
        max_iter=1,        # 1 iteración por llamada
        random_state=1
    )

    # Entrenamiento manual por épocas
    EPOCHS = 100
    losses = []

    for epoch in range(EPOCHS):
        model.fit(X_train, y_train)

        # Guardamos la pérdida (loss de entrenamiento)
        losses.append(model.loss_)

    # Predicción
    y_pred = model.predict(X_test)

    # Volver a escala original
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Métricas
    mse = mean_squared_error(y_test_original, y_pred)
    score = model.score(X_test, y_test)

    print("\n==============================")
    print(f"RESULTADO MLP: {model_name}")
    print("==============================")
    print(f"MSE: {mse:.6e}")
    print(f"R² score: {score:.6f}")

    # =========================
    # GRÁFICA DE PÉRDIDA
    # =========================

    plt.figure()
    plt.plot(range(EPOCHS), losses)
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.title(f"Pérdida vs épocas ({model_name})")
    plt.grid()

    # Guardar gráfico
    plt.savefig(f"{model_name}_loss.png")
    # plt.show()
    plt.close()

    return model

# =========================
# EXPERIMENTOS
# =========================

def run_all():

    # -------- Oscilador armónico --------
    data = load_dataset("oscillator_no_noise.csv")
    X = data[["x"]].values
    y = data["F"].values

    run_mlp(X, y, "Oscilador armónico")


    # -------- Ley de Kepler --------
    data = load_dataset("kepler_no_noise.csv")
    X = data[["r"]].values
    y = data["T"].values

    run_mlp(X, y, "Ley de Kepler")


    # -------- Ley de Coulomb --------
    data = load_dataset("coulomb_no_noise.csv")
    X = data[["q1", "q2", "r"]].values
    y = data["F"].values

    run_mlp(X, y, "Ley de Coulomb")


    # -------- Gas ideal --------
    data = load_dataset("ideal_gas_no_noise.csv")
    X = data[["n", "T", "V"]].values
    y = data["P"].values

    run_mlp(X, y, "Gas ideal")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    run_all()