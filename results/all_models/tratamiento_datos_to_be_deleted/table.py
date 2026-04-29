"""
visualize_benchmark_txt.py
==========================
Versión modificada que genera tablas en formato .txt con colores ANSI.
"""

import argparse
import os
import re
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN VISUAL (ANSI)
# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Códigos de color para la terminal
ANSI = {
    "reset": "\033[0m",
    "bold":  "\033[1m",
    "ok":    "\033[92m",    # Verde
    "approx": "\033[93m",   # Amarillo/Naranja
    "wrong":  "\033[91m",   # Rojo
    "header": "\033[96m",   # Cian
    "muted":  "\033[90m",   # Gris
}

# ══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH Y ORDEN (Mantenido del original)
# ══════════════════════════════════════════════════════════════════════════════

GROUND_TRUTH = {
    "boltzmann_entropy": "log(omega)",
    "coulomb":           "q1*q2/r**2",
    "ideal_gas":         "n*T/V",
    "kepler":            "sqrt(r**3)",
    "newton_cooling":    "1 + exp(-k*t)",
    "oscillator":        "-x",
    "projectile_range":  "v0**2*sin(2*theta)",
    "radioactive_decay": "exp(-lambda_*t)",
    "time_dilation":     "t/sqrt(1 - v**2)",
}

MODEL_ORDER = ["GPLearn", "PySR", "QLattice", "PySINDy", "Polynomial", "MLP_Standard", "MLP_Sparse", "MLP_Dropout"]
LAW_ORDER = ["boltzmann_entropy", "coulomb", "ideal_gas", "kepler", "newton_cooling", "oscillator", "projectile_range", "radioactive_decay", "time_dilation"]
LAW_DISPLAY = {k: k.replace("_", " ").title() for k in LAW_ORDER}
NOISE_IDX = {"no_noise": 0, "low_noise": 1, "high_noise": 2}
EQ_TABLE_MODELS = ["GPLearn", "PySR", "QLattice", "PySINDy", "Polynomial"]

# ══════════════════════════════════════════════════════════════════════════════
# PARSER Y LÓGICA DE DATOS (Mantenido del original)
# ══════════════════════════════════════════════════════════════════════════════

def parse_results(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    experiments = re.split(r"=+\nEXPERIMENT:", text)[1:]
    rows = []
    for exp in experiments:
        exp_name = exp.split("\n")[0].strip()
        blocks = re.findall(
            r"=== Model: (.*?) \| Dataset: (.*?) ===\n\n"
            r"Equation / Structure:\n(.*?)\n\n"
            r"Metrics:\nTest MSE: (.*?)\nTest MAE: (.*?)\n",
            exp, re.DOTALL,
        )
        for model, dataset, equation, mse, mae in blocks:
            rows.append({
                "experiment": exp_name, "dataset": dataset.strip(),
                "model": model.strip(), "equation": equation.strip(),
                "mse": float(mse), "mae": float(mae),
            })
    return pd.DataFrame(rows)

def _extract_law_noise(dataset: str):
    for suffix in ("_no_noise", "_low_noise", "_high_noise"):
        if dataset.endswith(suffix): return dataset[: -len(suffix)], suffix[1:]
    parts = dataset.rsplit("_", 1)
    return parts[0], parts[1] if len(parts) == 2 else ("unknown", "no_noise")

def load_data(results_path: str) -> dict:
    df = parse_results(results_path)
    extracted = df["dataset"].apply(_extract_law_noise)
    df["law"], df["noise"] = extracted.apply(lambda x: x[0]), extracted.apply(lambda x: x[1])
    df["noise_idx"] = df["noise"].map(NOISE_IDX).fillna(-1).astype(int)
    
    laws_present = [l for l in LAW_ORDER if l in df["law"].unique()]
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]

    # Simplificación de la lógica de recuperación para el ejemplo TXT
    recovery = {}
    equations = {nk: {} for nk in NOISE_IDX}
    
    for model in models_present:
        recovery[model] = {}
        for law in laws_present:
            scores = [0, 0, 0]
            for ni, nkey in enumerate(["no_noise", "low_noise", "high_noise"]):
                sub = df[(df["model"] == model) & (df["law"] == law) & (df["noise"] == nkey)]
                if not sub.empty:
                    mse = sub.iloc[0]["mse"]
                    eq = sub.iloc[0]["equation"]
                    # Lógica simple de score para visualización
                    score = 1.0 if mse < 1e-5 else (0.5 if mse < 1e-2 else 0.0)
                    scores[ni] = score
                    if law not in equations[nkey]: equations[nkey][law] = {"Verdad": GROUND_TRUTH.get(law, "?")}
                    equations[nkey][law][model] = (eq, score)
            recovery[model][law] = scores
            
    return {
        "laws": laws_present,
        "models": models_present,
        "law_names": {lk: LAW_DISPLAY.get(lk, lk) for lk in laws_present},
        "equations": equations,
    }

def trunc(s: str, n: int = 30) -> str:
    s = s.replace("**", "^")
    return s if len(s) <= n else s[: n - 1] + "…"

# ══════════════════════════════════════════════════════════════════════════════
# GENERADOR DE TABLAS TXT
# ══════════════════════════════════════════════════════════════════════════════

def _save_txt_table(data: dict, noise_key: str, title: str, filename: str):
    laws = data["laws"]
    law_names = data["law_names"]
    eq_data = data["equations"][noise_key]
    models_to_show = [m for m in EQ_TABLE_MODELS if m in data["models"]]
    
    # Configuración de anchos
    w_law = 20
    w_truth = 25
    w_mod = 32
    
    # Construcción de la tabla
    lines = []
    header_str = f"{ANSI['bold']}{ANSI['header']}"
    header_str += f"{'LEY'.ljust(w_law)} | {'VERDAD'.ljust(w_truth)} | "
    header_str += " | ".join([m.center(w_mod) for m in models_to_show])
    header_str += ANSI['reset']
    
    separator = "-" * (w_law + w_truth + 6 + (w_mod + 3) * len(models_to_show))
    
    lines.append(f"\n{ANSI['bold']}TABLA: {title.upper()}{ANSI['reset']}")
    lines.append(separator)
    lines.append(header_str)
    lines.append(separator)
    
    for lk in laws:
        row_eqs = eq_data.get(lk, {})
        truth = GROUND_TRUTH.get(lk, "?")
        
        line = f"{ANSI['bold']}{law_names[lk].ljust(w_law)}{ANSI['reset']} | {truth.ljust(w_truth)} | "
        
        cells = []
        for m in models_to_show:
            eq_str, score = row_eqs.get(m, ("—", 0))
            color = ANSI['ok'] if score == 1.0 else (ANSI['approx'] if score == 0.5 else ANSI['wrong'])
            
            # Marcador visual adicional por si el editor no soporta ANSI
            mark = "[OK] " if score == 1.0 else ("[AP] " if score == 0.5 else "[X]  ")
            
            formatted_eq = f"{color}{mark}{trunc(eq_str, w_mod-6).ljust(w_mod-5)}{ANSI['reset']}"
            cells.append(formatted_eq)
        
        line += " | ".join(cells)
        lines.append(line)
    
    lines.append(separator)
    
    # Añadir Leyenda al final del TXT
    lines.append(f"\nLEYENDA: {ANSI['ok']}[OK] Exacta{ANSI['reset']}  "
                 f"{ANSI['approx']}[AP] Aproximada{ANSI['reset']}  "
                 f"{ANSI['wrong']}[X] Incorrecta{ANSI['reset']}\n")

    # Guardar archivo
    full_content = "\n".join(lines)
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(full_content)
    
    # Imprimir en consola para feedback inmediato
    print(full_content)
    print(f"Archivo guardado en: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ... (manten el resto del código igual) ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None, 
                        help="Ruta al fichero. Si no se indica, busca en la carpeta actual y en la superior.")
    args = parser.parse_args()

    # --- LÓGICA DE BÚSQUEDA DINÁMICA ---
    results_path = args.results
    
    if results_path is None:
        # Intentar encontrarlo automáticamente
        script_dir = os.getcwd()
        posibles_rutas = [
            os.path.join(script_dir, "combined_results.txt"),             # Carpeta actual
            os.path.join(os.path.dirname(script_dir), "combined_results.txt") # Carpeta anterior
        ]
        
        for ruta in posibles_rutas:
            if os.path.exists(ruta):
                results_path = ruta
                break

    # Si después de buscar sigue siendo None o no existe
    if results_path is None or not os.path.exists(results_path):
        print(f"\n{ANSI['wrong']}Error: No se encuentra 'combined_results.txt'{ANSI['reset']}")
        print(f"Buscado en: {os.getcwd()} y en la carpeta superior.")
        print("Usa --results /ruta/al/archivo.txt para indicarlo manualmente.")
        return

    print(f"\n{ANSI['ok']}Archivo detectado en: {results_path}{ANSI['reset']}")
    print("Cargando datos y generando tablas TXT...\n")
    
    data = load_data(results_path)

    _save_txt_table(data, "no_noise", "Ecuaciones - Sin Ruido", "tabla_ecuaciones_no_noise.txt")
    _save_txt_table(data, "low_noise", "Ecuaciones - Ruido Bajo", "tabla_ecuaciones_low_noise.txt")
    _save_txt_table(data, "high_noise", "Ecuaciones - Ruido Alto", "tabla_ecuaciones_high_noise.txt")

    print(f"\n{ANSI['bold']}✓ Proceso finalizado.{ANSI['reset']} Los archivos .txt están en {ANSI['header']}'{OUT_DIR}/'{ANSI['reset']}")

if __name__ == "__main__":
    main()