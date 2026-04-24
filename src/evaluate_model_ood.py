import numpy as np
import os
import sys
import json
import glob
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import re
import sympy as sp
from sympy import lambdify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

# --- RUTAS DE DIRECTORIO ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_OOD_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data/data_ood"))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../results"))
RESULTS_OOD_DIR = os.path.abspath(os.path.join(BASE_DIR, "../results/results_ood"))

os.makedirs(DATA_OOD_DIR, exist_ok=True)
os.makedirs(RESULTS_OOD_DIR, exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
N_SAMPLES = 5000

# Base vectorial estricta del espacio de entrada
FEATURE_MAP = {
    "coulomb": ["q1", "q2", "r"],
    "oscillator": ["x"],
    "kepler": ["r"],
    "ideal_gas": ["n", "T", "V"],
    "projectile_range": ["v0", "theta"],
    "time_dilation": ["t", "v"],
    "radioactive_decay": ["lambda_", "t"],
    "newton_cooling": ["k", "t"],
    "boltzmann_entropy": ["omega"]
}

# ==========================================
# 1. GENERACIÓN DE DATOS OOD (Física)
# ==========================================

def coulomb_law(q1, q2, r): return (q1 * q2) / (r**2)
def harmonic_oscillator(x): return -x
def kepler_third_law(r): return np.sqrt(r**3)
def ideal_gas_law(n, T, V): return (n * T) / V
def projectile_range(v0, theta): return v0**2 * np.sin(2 * theta)
def time_dilation(t, v): return t / np.sqrt(1 - v**2)
def radioactive_decay(lam, t): return np.exp(-lam * t)
def newton_cooling(k, t): return 1.0 + (2.0 - 1.0) * np.exp(-k * t)
def boltzmann_entropy(omega): return np.log(omega)

def generate_ood_data(law_name):
    if law_name == "coulomb":
        q1, q2 = np.random.uniform(3.0, 6.0, (2, N_SAMPLES))
        r = np.random.uniform(4.0, 8.0, N_SAMPLES)
        return np.column_stack((q1, q2, r)), coulomb_law(q1, q2, r)
    elif law_name == "oscillator":
        x = np.random.uniform(3.0, 6.0, N_SAMPLES)
        return x.reshape(-1, 1), harmonic_oscillator(x)
    elif law_name == "kepler":
        r = np.random.uniform(4.0, 10.0, N_SAMPLES)
        return r.reshape(-1, 1), kepler_third_law(r)
    elif law_name == "ideal_gas":
        n, T, V = np.random.uniform(3.0, 6.0, (3, N_SAMPLES))
        return np.column_stack((n, T, V)), ideal_gas_law(n, T, V)
    elif law_name == "projectile_range":
        v0 = np.random.uniform(4.0, 10.0, N_SAMPLES)
        theta = np.random.uniform(0.1, np.pi/2 - 0.1, N_SAMPLES)
        return np.column_stack((v0, theta)), projectile_range(v0, theta)
    elif law_name == "time_dilation":
        t = np.random.uniform(3.0, 6.0, N_SAMPLES)
        v = np.random.uniform(0.96, 0.999, N_SAMPLES)
        return np.column_stack((t, v)), time_dilation(t, v)
    elif law_name == "radioactive_decay":
        lam, t = np.random.uniform(3.0, 6.0, (2, N_SAMPLES))
        return np.column_stack((lam, t)), radioactive_decay(lam, t)
    elif law_name == "newton_cooling":
        k, t = np.random.uniform(3.0, 6.0, (2, N_SAMPLES))
        return np.column_stack((k, t)), newton_cooling(k, t)
    elif law_name == "boltzmann_entropy":
        omega = np.random.uniform(20.0, 100.0, N_SAMPLES)
        return omega.reshape(-1, 1), boltzmann_entropy(omega)
    raise ValueError(f"Ley {law_name} no reconocida.")

# ==========================================
# 2. CARGA DE MODELOS Y PREDICCIÓN
# ==========================================
def load_and_predict(weights_path, X_ood, law):
    with open(weights_path, 'r') as f:
        model_data = json.load(f)

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    syms = [sp.Symbol(name) for name in FEATURE_MAP[law]]
    n_samples = X_ood.shape[0]

    def clean_str(eq):
        """Pre-procesamiento base. Aplica coerciones idiomáticas directas."""
        eq = str(eq).replace('^', '**')
        # Blindaje contra fragmentación de lambdas en PySR/GPLearn
        eq = re.sub(r'\blambd\b', 'lambda_', eq)
        eq = re.sub(r'\blambda\b', 'lambda_', eq)
        
        # Base vectorial genérica a base física estricta
        for i, name in enumerate(FEATURE_MAP[law]):
            eq = re.sub(fr'\bx{i}\b', name, eq)
            
        eq = re.sub(r'(?<=[a-zA-Z0-9_])\s+(?=[a-zA-Z0-9_])', '*', eq)
        return eq

    def safe_parse_and_evaluate(eq_str):
        eq_clean = clean_str(eq_str)
        
        # 1. Inyección de contexto atómico para blindar bases complejas (ej. q1, lambda_)
        parse_dict = {name: sp.Symbol(name) for name in FEATURE_MAP[law]}
        parse_dict.update({
            "square": lambda x: x**2, "inv": lambda x: 1.0 / x,
            "div": lambda x, y: x / y, "mul": lambda x, y: x * y,
            "add": lambda x, y: x + y, "sub": lambda x, y: x - y,
            "neg": lambda x: -x
        })

        try:
            expr = parse_expr(eq_clean, local_dict=parse_dict, transformations=transformations)
        except Exception as e:
            raise RuntimeError(f"Error de parsing algebraico: {e}")

        # 2. Auditoría espectral de fugas
        unresolved = expr.free_symbols - set(syms)
        if unresolved:
            raise RuntimeError(f"Símbolos no resueltos tras parsing algebraico: {unresolved}")

        # 3. Compilación topológica a Numpy
        local_numpy_math = {
            "square": lambda x: x**2, "inv": lambda x: 1.0 / x,
            "div": lambda x, y: x / y, "mul": lambda x, y: x * y,
            "add": lambda x, y: x + y, "sub": lambda x, y: x - y,
            "neg": lambda x: -x, "abs": np.abs, "max": np.maximum,
            "min": np.minimum, "sin": np.sin, "cos": np.cos,
            "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt
        }
        
        f = lambdify(syms, expr, modules=[local_numpy_math, "numpy"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = f(*X_ood.T)

        if np.isscalar(y_pred) or (hasattr(y_pred, 'shape') and y_pred.shape == ()):
            y_pred = np.full(n_samples, float(y_pred))

        try:
            y_pred = np.array(y_pred, dtype=float).flatten()
        except TypeError:
            raise RuntimeError("Fallo crítico: El evaluador retiene objetos SymPy residuales.")

        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Divergencia Matemática (NaN/Inf detectado).")

        return y_pred

    # --- Evaluación condicional de arquitecturas ---
    if "MLP" in weights_path:
        from src.models.mlp import MLPWrapper
        if "Standard" in weights_path: model_type = 'standard'
        elif "Sparse" in weights_path: model_type = 'sparse'
        elif "Dropout" in weights_path: model_type = 'dropout'
        else: raise ValueError("Topología MLP no reconocida.")

        wrapper = MLPWrapper(input_dim=X_ood.shape[1], model_type=model_type)
        state_dict = {k: torch.tensor(v) for k, v in model_data.items()}
        wrapper.model.load_state_dict(state_dict)

        y_pred = wrapper.predict(X_ood).flatten()
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Divergencia Matemática (NaN/Inf detectado).")
        return y_pred

    elif any(m in weights_path for m in ["PySR", "GPLearn", "QLattice"]):
        eq_str = model_data["best_equation"] if "PySR" in weights_path else model_data["equation"] 
        return safe_parse_and_evaluate(eq_str)

    elif "PySINDy" in weights_path:
        coefs, feats = model_data["coefficients"], model_data["feature_names"]
        terms = [f"({c})*({clean_str(n)})" for c, n in zip(coefs, feats) if abs(c) > 1e-12]
        eq_str = " + ".join(terms) if terms else "0"
        return safe_parse_and_evaluate(eq_str)

    elif "Polynomial" in weights_path:
        coefs = model_data["coefficients"]
        feats_out = model_data["feature_names_out"]
        terms = [f"({c})*({clean_str(n)})" for c, n in zip(coefs, feats_out) if abs(c) > 1e-12]
        eq_str = f'{model_data["intercept"]} + ' + " + ".join(terms)
        return safe_parse_and_evaluate(eq_str)

    else:
        raise ValueError(f"Arquitectura no soportada.")

# ==========================================
# 3. EVALUACIÓN DE OOD
# ==========================================
if __name__ == "__main__":
    laws = [
        "coulomb", "oscillator", "kepler", "ideal_gas",
        "projectile_range", "time_dilation", "radioactive_decay",
        "newton_cooling", "boltzmann_entropy"
    ]

    results = {}

    for law in laws:
        print(f"--- Evaluando {law} OOD ---")
        X_ood, y_true = generate_ood_data(law)

        data_path = os.path.join(DATA_OOD_DIR, f"{law}_ood_data.npz")
        np.savez(data_path, X=X_ood, y=y_true)

        # Usamos list(set()) para purgar solapamiento de duplicados en la jerarquía del glob
        weight_files = sorted(list(set(glob.glob(os.path.join(RESULTS_DIR, "*", f"{law}_*_weights.json")))))
        law_results = {}

        for w_file in weight_files:
            filename = os.path.basename(w_file)
            base_str = filename.replace(f"{law}_", "").replace("_weights.json", "")
            
            if "_noise_" in base_str:
                noise_str, model_str = base_str.split("_noise_", 1)
                noise_level = f"{noise_str}_noise"
                model_name = model_str
            else:
                noise_level = "unknown"
                model_name = base_str

            tag = f"{model_name}_{noise_level}"

            try:
                y_pred = load_and_predict(w_file, X_ood, law)
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)

                law_results[tag] = {"MAE": mae, "MSE": mse}
                print(f"  [{tag}] MAE: {mae:.4e} | MSE: {mse:.4e}")

            except Exception as e:
                law_results[tag] = {"MAE": None, "MSE": None, "error": str(e)}
                print(f"  [{tag}] ERROR: {e}")

        results[law] = law_results

    with open(os.path.join(RESULTS_OOD_DIR, "ood_metrics_summary.json"), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nEvaluación finalizada. Resultados en: {RESULTS_OOD_DIR}")