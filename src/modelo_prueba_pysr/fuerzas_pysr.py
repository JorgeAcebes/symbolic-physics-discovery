# este programa hay que correrlo desde symbolic-physics-discovery con por ejemplo, python modelo_pysr/

import numpy as np
import os
from pysr import PySRRegressor

# =========================
# RUTAS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset(filename):
    """
    Carga dataset desde:
    symbolic-physics-discovery/data_generator/datasets/
    """
    full_path = os.path.join(
        BASE_DIR,
        "..",
        "data_generator",
        "datasets",
        filename
    )
    full_path = os.path.abspath(full_path)
    print(f"Cargando: {full_path}")
    return np.loadtxt(full_path, comments="#")


# =========================
# MODELO PySR
# =========================
def run_pysr(X, y, variable_names, model_name):
    """
    Ejecuta regresión simbólica con PySR
    """
    model = PySRRegressor(
        niterations=4,           # Número de iteraciones
        binary_operators=["+", "-", "*", "/", "^"],
        constraints={"^": (-1, 1)},  # exponentes simples, para poder obtener raíces
        unary_operators=[],
        populations=10,            # tamaño de población
        maxsize=15,                # controla complejidad
        model_selection="best",
        verbosity=1
    )

    model.fit(X, y)

    print("\n==============================")
    print(f"RESULTADO: {model_name}")
    print("==============================")
    print(model)

    return model


# =========================
# EJECUCIÓN DE EXPERIMENTOS
# =========================
def run_all(): 
    #sin ruido las clava en 5 iteraciones excepto Coulomb (Kepler muy ligeramente peor)
    #con ruido no he probado actualmente
    # Oscilador armónico
    data = load_dataset("oscillator_no_noise.txt")
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    run_pysr(X, y, ["x"], "Oscilador armónico")

    # Ley de Kepler
    data = load_dataset("kepler_no_noise.txt")
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    run_pysr(X, y, ["r"], "Ley de Kepler")

    # Ley de Coulomb
    data = load_dataset("coulomb_no_noise.txt")
    X = data[:, :3]
    y = data[:, 3]
    run_pysr(X, y, ["q1", "q2", "r"], "Ley de Coulomb")

    # Gas ideal
    data = load_dataset("ideal_gas_no_noise.txt")
    X = data[:, :3]
    y = data[:, 3]
    run_pysr(X, y, ["n", "T", "V"], "Gas ideal")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_all()
