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
try:
    from utils.utils import set_plot_style
    set_plot_style(for_paper=True)
except ImportError:
    pass

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
    ("MLP_Standard", "MLP Estándar  "),
    ("MLP_Sparse", "MLP Disperso  "),
    ("MLP_Dropout", "MLP Dropout  "),
    ("Polynomial", "Polinómico  "),
    ("PySINDy", "PySINDy  "),
    ("GPLearn", "GPLearn  "),
    ("PySR", "PySR  "),
    ("QLattice", "QLattice  ")
]

NOISE_ORDER = {
    "no": 0,
    "low": 1,
    "high": 2
}

# --- PARÁMETROS DE VISUALIZACIÓN ---
COLOR_EXACT = '#2ca02c'
COLOR_BUENO = 'limegreen'
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

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(bottom=0.20)

    # --- RENDERIZADO DE MATRIZ ---
    w_rect = 0.75  # Ancho del rectángulo
    h_rect = 0.50  # Alto del rectángulo (apaisado)

    # i recorre las leyes (filas/eje y), j recorre los modelos (columnas/eje x)
    for i, law_key in enumerate(laws_keys):
        for j, model_key in enumerate(models_keys):
            
            # Centrado analítico: x_0 = j - w/2, y_0 = -i - h/2
            rect = plt.Rectangle((j - w_rect/2, -i - h_rect/2), w_rect, h_rect, color='#f5f5f5', zorder=1)
            ax.add_patch(rect)

            for noise, n_idx in NOISE_ORDER.items():
                key = f"{model_key}_{noise}_noise"
                
                try:
                    mse = data.get(law_key, {}).get(key, {}).get("MSE", None)
                except AttributeError:
                    mse = None

                if mse is None or mse == "null" or np.isnan(mse) or np.isinf(mse):
                    c = COLOR_INCORRECT
                elif float(mse) < 1e-20:
                    c = COLOR_EXACT
                elif float(mse) < 1e-4:
                    c = COLOR_BUENO
                elif float(mse) < 1e-2:
                    c = COLOR_APPROX
                else:
                    c = COLOR_INCORRECT

                # Variación horizontal para los distintos niveles de ruido
                dx = (n_idx - 1) * 0.25
                ax.scatter(j + dx, -i, color=c, s=120, zorder=2, edgecolors='none')

    # Eje X configurado para los Modelos
    ax.set_xticks(range(len(models_keys)))
    ax.set_xticklabels(models_names, rotation=35, ha='left', va='bottom', color='#222222')
    ax.xaxis.tick_top()

    # Eje Y configurado para las Leyes Físicas
    ax.set_yticks([-i for i in range(len(laws_keys))])
    ax.set_yticklabels(laws_names, color='#222222')
    ax.tick_params(axis='y', pad=5)

    # Ajuste exacto de los límites considerando las nuevas dimensiones matriciales
    ax.set_xlim(-w_rect/2, len(models_keys) - 1 + w_rect/2)
    ax.set_ylim(-len(laws_keys) + 1 - h_rect/2, h_rect/2)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    # --- LEYENDA ESTRUCTURAL (RUIDO) ---
    noise_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666', markersize=12, label=r'Sin ruido'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666', markersize=12, label=r'Ruido bajo'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666', markersize=12, label=r'Ruido alto')
    ]
    
    leg_noise = ax.legend(handles=noise_elements, loc='upper left', bbox_to_anchor=(-0.20, -0.15),
                          frameon=False, ncol=3, columnspacing=0.5, labelcolor='#222222', 
                          handletextpad=0.4, title=r'\textbf{Disposición espacial (izq. a der.):}',
                          fontsize=14, title_fontsize=15)
    
    leg_noise._legend_box.align = "left"
    ax.add_artist(leg_noise) 

    # --- LEYENDA DE PRECISIÓN (MSE) ---
    mse_elements = [
        mpatches.Circle((0, 0), radius=1, color=COLOR_EXACT,  label=r'Exacta ($\text{MSE} < 10^{-20}$)'),
        mpatches.Circle((0, 0), radius=1, color=COLOR_BUENO,  label=r'Precisa ($10^{-4} \leq \text{MSE} < 10^{-4}$)'),
        mpatches.Circle((0, 0), radius=1, color=COLOR_APPROX, label=r'Aproximada ($10^{-4} \leq \text{MSE} < 10^{-2}$)'),
        mpatches.Circle((0, 0), radius=1, color=COLOR_INCORRECT, label=r'Incorrecta ($\text{MSE} \geq 10^{-2}$)' )
    ]
    
    ax.legend(handles=mse_elements, loc='upper right', bbox_to_anchor=(1.0, -0.10), 
              frameon=False, ncol=1, labelcolor='#222222', handletextpad=0.5, fontsize=14)

    plt.savefig(os.path.join(RESULTS_OOD_DIR, "model_recovery_grid.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    plot_recovery_grid_json()