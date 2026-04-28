# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import json
import torch
import random
from matplotlib.lines import Line2D

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
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
# --- CONFIGURACIÓN LATEX Y TAMAÑOS ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 16, 
    "ytick.labelsize": 16
})

# --- ESTRUCTURAS DE DATOS ---
LAWS_INFO = [
    ("coulomb", "Ley de Coulomb"),
    ("oscillator", "Oscilador armónico"),
    ("kepler", "Tercera ley de Kepler"),
    ("ideal_gas", "Ley de los gases ideales"),
    ("time_dilation", "Dilatación temporal"),
    ("projectile_range", "Rango proyectil"),
    ("radioactive_decay", "Decaimiento radiactivo"),
    ("newton_cooling", "Enfriamiento de Newton"),
    ("boltzmann_entropy", "Entropía de Boltzmann")
]

MODELS_INFO = [
    ("MLP_Standard", "MLP Estándar"),
    ("MLP_Sparse", "MLP Disperso"),
    ("MLP_Dropout", "MLP Dropout"),
    ("Polynomial", "Polinómico"),
    ("PySINDy", "PySINDy"),
    ("GPLearn", "GPLearn"),
    ("PySR", "PySR"),
    ("QLattice", "QLattice")
]

NOISE_ORDER = {
    "no": 0,
    "low": 1,
    "high": 2
}

# --- PARÁMETROS DE VISUALIZACIÓN ---
COLOR_EXACT = '#2ca02c'
COLOR_APPROX = '#ff7f0e'
COLOR_INCORRECT = '#d62728'

def plot_recovery_grid_json():
    json_path = os.path.join(RESULTS_OOD_DIR, "ood_metrics_summary.json")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {json_path}")
        return

    laws_keys = [l[0] for l in LAWS_INFO]
    laws_names = [l[1] for l in LAWS_INFO]
    models_keys = [m[0] for m in MODELS_INFO]
    models_names = [m[1] for m in MODELS_INFO]

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_aspect('equal')
    
    # Ajustar margen inferior para acomodar las leyendas
    fig.subplots_adjust(bottom=0.25)

    # --- RENDERIZADO DE MATRIZ ---
    for i, model_key in enumerate(models_keys):
        for j, law_key in enumerate(laws_keys):
            rect = plt.Rectangle((j - 0.45, -i - 0.4), 0.9, 0.8, color='#f5f5f5', zorder=1)
            ax.add_patch(rect)

            for noise, n_idx in NOISE_ORDER.items():
                key = f"{model_key}_{noise}_noise"
                
                try:
                    mse = data.get(law_key, {}).get(key, {}).get("MSE", None)
                except AttributeError:
                    mse = None

                if mse is None or mse == "null" or np.isnan(mse) or np.isinf(mse):
                    c = COLOR_INCORRECT
                elif float(mse) <= 1e-4:
                    c = COLOR_EXACT
                elif float(mse) <= 1e-2:
                    c = COLOR_APPROX
                else:
                    c = COLOR_INCORRECT

                dx = (n_idx - 1) * 0.3
                ax.scatter(j + dx, -i, color=c, s=200, zorder=2, edgecolors='none')

    ax.set_xticks(range(len(laws_keys)))
    ax.set_xticklabels(laws_names, rotation=35, ha='left', va='bottom', color='#222222')
    ax.xaxis.tick_top()

    ax.set_yticks([-i for i in range(len(models_keys))])
    ax.set_yticklabels(models_names, color='#222222')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    # --- LEYENDA ESTRUCTURAL INFERIOR (RUIDO) ---
# --- LEYENDA ESTRUCTURAL (RUIDO) ---
    # Mapeo topológico: la lectura izquierda-derecha refleja la posición en la celda.
    noise_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666', markersize=12, label=r'Sin ruido'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666', markersize=12, label=r'Ruido bajo'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666', markersize=12, label=r'Ruido alto')
    ]
    
    leg_noise = ax.legend(handles=noise_elements, loc='lower left', bbox_to_anchor=(0, -0.18),
                          frameon=False, ncol=3, columnspacing=2.0, labelcolor='#222222', 
                          handletextpad=0.4, title=r'\textbf{Disposición espacial en celda (izq. a der.):}')
    
    leg_noise._legend_box.align = "left"
    leg_noise.get_title().set_fontsize(13)
    ax.add_artist(leg_noise) # Fija esta leyenda para que la siguiente no la sobrescriba

    # --- LEYENDA DE PRECISIÓN (MSE) ---
    mse_elements = [
        mpatches.Circle((0, 0), radius=1, color=COLOR_EXACT, label=r'Exacta ($\text{MSE} \le 10^{-4}$)'),
        mpatches.Circle((0, 0), radius=1, color=COLOR_APPROX, label=r'Aproximada ($\text{MSE} \le 10^{-2}$)'),
        mpatches.Circle((0, 0), radius=1, color=COLOR_INCORRECT, label=r'Incorrecta ($\text{MSE} > 10^{-2}$)')
    ]
    
    ax.legend(handles=mse_elements, loc='lower right', bbox_to_anchor=(1, -0.22), 
              frameon=False, ncol=1, labelcolor='#222222', handletextpad=0.5)

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