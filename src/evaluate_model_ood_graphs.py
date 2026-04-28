import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import json
import glob
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import re
import random
import sympy as sp
from sympy import lambdify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # Si usas GPU con Torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- RUTAS DE DIRECTORIO ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_OOD_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data/data_ood"))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../results"))
RESULTS_OOD_DIR = os.path.abspath(os.path.join(BASE_DIR, "../results/results_ood"))

os.makedirs(DATA_OOD_DIR, exist_ok=True)
os.makedirs(RESULTS_OOD_DIR, exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import set_plot_style
set_plot_style(for_paper=True)

# --- ESTRUCTURAS DE DATOS ---
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

MODEL_ORDER = {
    "Polynomial": 0,
    "MLP_Standard": 1,
    "MLP_Sparse": 2,
    "MLP_Dropout": 3,
    "PySR": 4,
    "QLattice": 5,
    "GPLearn": 6,
    "PySINDy": 7
}

NOISE_ORDER = {
    "no": 0,
    "low": 1,
    "high": 2
}

# --- PARÁMETROS DE VISUALIZACIÓN ---
COLOR_EXACT = 'green'     # Verde
COLOR_APPROX = 'orange'   # Naranja
COLOR_INCORRECT = 'red' # Rojo

def plot_recovery_grid_json():
    # Cargar los datos JSON
    json_path = os.path.join(RESULTS_OOD_DIR, "ood_metrics_summary.json")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {json_path}")
        return

    # Ordenar las listas según los diccionarios
    laws = list(FEATURE_MAP.keys())
    models = sorted(MODEL_ORDER.keys(), key=lambda x: MODEL_ORDER[x])

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')

    # Bucle para dibujar las celdas y los puntos
    for j, law in enumerate(laws):
        for i, model in enumerate(models):
            # Dibujar el fondo gris de la celda
            rect = plt.Rectangle((j - 0.45, -i - 0.4), 0.9, 0.8, color='#f5f5f5', zorder=1)
            ax.add_patch(rect)

            for noise, n_idx in NOISE_ORDER.items():
                key = f"{model}_{noise}_noise"
                
                # Extraer MSE con manejo de errores
                try:
                    mse_dict = data.get(law, {}).get(key, {})
                    mse = mse_dict.get("MSE", None)
                except AttributeError:
                    mse = None

                # Evaluar umbrales
                if mse is None or mse == "null" or np.isnan(mse) or np.isinf(mse):
                    c = COLOR_INCORRECT
                elif float(mse) <= 1e-3:
                    c = COLOR_EXACT
                elif float(mse) <= 1e-2:
                    c = COLOR_APPROX
                else:
                    c = COLOR_INCORRECT

                # Coordenadas: desplazar horizontalmente según el nivel de ruido (-0.25, 0, 0.25)
                dx = (n_idx - 1) * 0.25
                ax.scatter(j + dx, -i, color=c, s=180, zorder=2, edgecolors='none')

    # Configuración de los ejes
    ax.set_xticks(range(len(laws)))
    # Formatear nombres de las leyes (eliminar guiones bajos y capitalizar)
    ax.set_xticklabels([l.replace('_', ' ').capitalize() for l in laws], 
                       rotation=35, ha='left', va='bottom', fontsize=11, color='#555555')
    ax.xaxis.tick_top()

    ax.set_yticks([-i for i in range(len(models))])
    ax.set_yticklabels(models, fontsize=11, color='#333333')

    # Limpiar bordes y marcas de los ejes
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    # --- LEYENDAS ---
    # Leyenda descriptiva superior (niveles de ruido)
    ax.text(0.5, 1.15, r'$\cdot$ 3 puntos = sin ruido $\cdot$ ruido bajo $\cdot$ ruido alto $\cdot$',
            transform=ax.transAxes, ha='center', va='center', color='#777777', fontsize=12, style='italic')

    # Leyenda de categorías (colores)
    legend_elements = [
        mpatches.Circle((0, 0), radius=1, color=COLOR_EXACT, label=r'Exacta ($\text{MSE} \le 10^{-4}$)'),
        mpatches.Circle((0, 0), radius=1, color=COLOR_APPROX, label=r'Aproximada ($\text{MSE} \le 10^{-2}$)'),
        mpatches.Circle((0, 0), radius=1, color=COLOR_INCORRECT, label=r'Incorrecta ($\text{MSE} > 10^{-2}$)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, -0.15), 
              frameon=False, ncol=1, fontsize=11, labelcolor='#555555', handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_recovery_grid.pdf"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_recovery_grid_json()



# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import os
# import sys
# import re
# import random
# import torch

# def set_seed(seed=42):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(42)

# # --- RUTAS DE DIRECTORIO ---
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# DATA_OOD_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data/data_ood"))
# RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../results"))
# RESULTS_OOD_DIR = os.path.abspath(os.path.join(BASE_DIR, "../results/results_ood"))
# TXT_RESULTS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../results/all_models/combined_results.txt"))

# os.makedirs(DATA_OOD_DIR, exist_ok=True)
# os.makedirs(RESULTS_OOD_DIR, exist_ok=True)

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.utils import set_plot_style
# set_plot_style(for_paper=True)

# # --- ESTRUCTURAS DE DATOS ---
# FEATURE_MAP = {
#     "coulomb": ["q1", "q2", "r"],
#     "oscillator": ["x"],
#     "kepler": ["r"],
#     "ideal_gas": ["n", "T", "V"],
#     "projectile_range": ["v0", "theta"],
#     "time_dilation": ["t", "v"],
#     "radioactive_decay": ["lambda_", "t"],
#     "newton_cooling": ["k", "t"],
#     "boltzmann_entropy": ["omega"]
# }

# MODEL_ORDER = {
#     "Polynomial": 0,
#     "MLP_Standard": 1,
#     "MLP_Sparse": 2,
#     "MLP_Dropout": 3,
#     "PySR": 4,
#     "QLattice": 5,
#     "GPLearn": 6,
#     "PySINDy": 7
# }

# NOISE_ORDER = {
#     "no": 0,
#     "low": 1,
#     "high": 2
# }

# # --- PARÁMETROS DE VISUALIZACIÓN ---
# COLOR_EXACT = '#5b9e31'     # Verde: MSE <= 10^{-4}
# COLOR_APPROX = '#c07b22'    # Naranja: MSE <= 10^{-2}
# COLOR_INCORRECT = '#d63031' # Rojo: MSE > 10^{-2} o Divergencia

# def parse_txt_results(filepath):
#     """
#     Parsea el archivo txt de resultados in-domain y reconstruye 
#     un diccionario compatible con la lógica de ploteo.
#     """
#     data = {}
#     if not os.path.exists(filepath):
#         print(f"Advertencia: No se encontró el archivo {filepath}")
#         return data

#     with open(filepath, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     current_model = None
#     current_dataset = None

#     for line in lines:
#         line = line.strip()
#         if line.startswith("=== Model:"):
#             # Extraer modelo y dataset (Ej: "=== Model: GPLearn | Dataset: coulomb_high_noise ===")
#             parts = line.replace("===", "").split("|")
#             current_model = parts[0].replace("Model:", "").strip()
#             current_dataset = parts[1].replace("Dataset:", "").strip()
        
#         elif line.startswith("Test MSE:"):
#             if current_model and current_dataset:
#                 mse_str = line.split(":")[1].strip()
#                 try:
#                     mse_val = float(mse_str)
#                 except ValueError:
#                     mse_val = None

#                 # Mapear el nivel de ruido y la ley
#                 for noise in ["no_noise", "low_noise", "high_noise"]:
#                     if current_dataset.endswith(noise):
#                         law = current_dataset.replace(f"_{noise}", "")
#                         key = f"{current_model}_{noise}"
                        
#                         if law not in data:
#                             data[law] = {}
#                         data[law][key] = {"MSE": mse_val}
#                         break

#     return data

# def plot_recovery_grid():
#     # Extraer datos procesando el TXT
#     data = parse_txt_results(TXT_RESULTS_PATH)
    
#     if not data:
#         print("No hay datos para graficar. Verifique la ruta del archivo txt.")
#         return

#     laws = list(FEATURE_MAP.keys())
#     models = sorted(MODEL_ORDER.keys(), key=lambda x: MODEL_ORDER[x])

#     fig, ax = plt.subplots(figsize=(12, 10))
#     ax.set_aspect('equal')

#     for j, law in enumerate(laws):
#         for i, model in enumerate(models):
#             # Fondo de la celda
#             rect = plt.Rectangle((j - 0.45, -i - 0.4), 0.9, 0.8, color='#f5f5f5', zorder=1)
#             ax.add_patch(rect)

#             for noise, n_idx in NOISE_ORDER.items():
#                 key = f"{model}_{noise}_noise"
                
#                 # Obtener MSE iterando el diccionario dinámico
#                 try:
#                     mse = data.get(law, {}).get(key, {}).get("MSE", None)
#                 except AttributeError:
#                     mse = None

#                 # Lógica estricta de evaluación
#                 if mse is None or np.isnan(mse) or np.isinf(mse):
#                     c = COLOR_INCORRECT
#                 elif mse <= 1e-4:
#                     c = COLOR_EXACT
#                 elif mse <= 1e-2:
#                     c = COLOR_APPROX
#                 else:
#                     c = COLOR_INCORRECT

#                 # Posicionamiento dependiente de nivel de ruido
#                 dx = (n_idx - 1) * 0.25
#                 ax.scatter(j + dx, -i, color=c, s=180, zorder=2, edgecolors='none')

#     # Configuración tipográfica y de ejes
#     ax.set_xticks(range(len(laws)))
#     ax.set_xticklabels([l.replace('_', ' ').capitalize() for l in laws], 
#                        rotation=35, ha='left', va='bottom', fontsize=11, color='#555555')
#     ax.xaxis.tick_top()

#     ax.set_yticks([-i for i in range(len(models))])
#     ax.set_yticklabels(models, fontsize=11, color='#333333')

#     # Limpieza de bounding box
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#     ax.tick_params(axis='both', which='both', length=0)

#     # Leyendas sin título
#     ax.text(0.5, 1.15, '$\cdot$ 3 puntos = sin ruido $\cdot$ ruido bajo $\cdot$ ruido alto $\cdot$',
#             transform=ax.transAxes, ha='center', va='center', color='#777777', fontsize=12, style='italic')

#     legend_elements = [
#         mpatches.Circle((0, 0), radius=1, color=COLOR_EXACT, label=r'Exacta ($\text{MSE} \le 10^{-4}$)'),
#         mpatches.Circle((0, 0), radius=1, color=COLOR_APPROX, label=r'Aproximada ($\text{MSE} \le 10^{-2}$)'),
#         mpatches.Circle((0, 0), radius=1, color=COLOR_INCORRECT, label=r'Incorrecta ($\text{MSE} > 10^{-2}$)')
#     ]
#     ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, -0.15), 
#               frameon=False, ncol=1, fontsize=11, labelcolor='#555555', handletextpad=0.5)

#     plt.tight_layout()
#     plt.savefig(os.path.join(RESULTS_DIR, "model_recovery_grid_indomain.pdf"), dpi=300, bbox_inches='tight')
#     plt.show()

# if __name__ == "__main__":
#     plot_recovery_grid()