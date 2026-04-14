import numpy as np
import os
import pandas as pd
from pysr import PySRRegressor
import sympy
from datetime import datetime




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ruta absoluta POSIX (Julia es sensible a esto)
RESULTS_BASE = os.path.abspath(os.path.join(BASE_DIR, "../../results/testing_pysr")).replace("\\", "/")

def load_dataset(filename):
    full_path = os.path.abspath(os.path.join(BASE_DIR, "../../data", filename)).replace("\\", "/")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Dataset no hallado: {full_path}")
    return np.loadtxt(full_path, delimiter=',', skiprows=1)


def run_pysr(X, y, variable_names, model_name):
    output_dir = f"{RESULTS_BASE}/{model_name.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Formateo de fecha para el archivo
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_final = f"{output_dir}/hall_of_fame_{date_str}.csv"

    model = PySRRegressor(
        niterations=50,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["inv", "square"],
        constraints={'^': (-1, 1)},
        nested_constraints={'^': {'^': 0}},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        temp_equation_file=csv_final, 
        populations=10,            # tamaño de población
        maxsize=15,                # controla complejidad
        model_selection="best",
        verbosity=1
    )

    model.fit(X, y, variable_names=variable_names)

    # ==========================================
    # FILTRADO Y GUARDADO DEL HALL OF FAME
    # ==========================================
    if hasattr(model, "equations_"):
        df = model.equations_.copy()
        
        # 1. El mejor modelo (mayor score) aparece el primero
        df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
        
        # 2. Purgar formato lambda
        if "lambda_format" in df.columns:
            df = df.drop(columns=["lambda_format"])
            
        # 3. Mantener symbolic simplification (sympy_format) SOLO para el mejor modelo (índice 0)
        if "sympy_format" in df.columns:
            df.loc[1:, "sympy_format"] = "" 
            
        cols = df.columns.tolist()
        if "score" in cols and "equation" in cols:
            cols.remove("score")
            index_eq = cols.index("equation")
            cols.insert(index_eq, "score") # Inserta score justo antes de equation
            df = df[cols]

        df.to_csv(csv_final, index=False)
        print(f"\n[SISTEMA]: Archivo persistido manualmente en: {csv_final}")

    # SIMPLIFICACIÓN SIMBÓLICA POR TERMINAL
    try:
        best_sympy = model.sympy()
        simplified = sympy.simplify(sympy.nsimplify(best_sympy, tolerance=1e-4))
        print(f"\nLey descubierta ({model_name}):")
        print(f"  f({', '.join(variable_names)}) = {simplified}")
    except Exception as e:
        print(f"Error simplificando: {e}")

    return model


def run_all(noise): 
    if noise not in ['no', 'low', 'high']:
        raise ValueError(f"Argumento '{noise}' no válido. Usar: 'no', 'low', 'high'.")
    experiments = [
        {
            "name": "Oscilador Armónico",
            "file": f"oscillator_{noise}_noise.csv",
            "vars": ["x"],
            "target_idx": 1
        },
        {
            "name": "Tercera Ley de Kepler",
            "file": f"kepler_{noise}_noise.csv",
            "vars": ["r"],
            "target_idx": 1
        },
        {
            "name": "Ley de Coulomb",
            "file": f"coulomb_{noise}_noise.csv",
            "vars": ["q1", "q2", "r"],
            "target_idx": 3
        },
        {
            "name": "Gas Ideal",
            "file": f"ideal_gas_{noise}_noise.csv",
            "vars": ["n", "T", "V"],
            "target_idx": 3
        }
    ]

    for exp in experiments:
        try:
            data = load_dataset(exp["file"])
            
            # Selección dinámica de columnas
            num_vars = len(exp["vars"])
            X = data[:, :num_vars].reshape(-1, num_vars)
            y = data[:, exp["target_idx"]]
            
            run_pysr(X, y, exp["vars"], exp["name"])
            
        except FileNotFoundError:
            print(f"Error: El archivo {exp['file']} no existe.")
        except IndexError:
            print(f"Error: Dimensiones de datos incompatibles en {exp['name']}.")
        except Exception as e:
            print(f"Error inesperado en {exp['name']}: {e}")


# EJECUCIÓN
if __name__ == "__main__":
    run_all('low')