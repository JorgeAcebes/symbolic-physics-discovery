# importo las librerías
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Para poder llamar a utils.utils
from utils.utils import set_plot_style

set_plot_style(for_paper=True)

# Configuración estricta de LaTeX para cumplimiento académico
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "mathtext.fontset": "cm"
})

plots = True

np.random.seed(1) # Aseguramos reproducibilidad

N_SAMPLES = 100000  # nº puntos para cada ley

NOISE_LEVELS = {
    "no_noise": 0.0,
    "low_noise": 0.01,
    "high_noise": 0.1
}

# --- GESTIÓN DE RUTAS ---
# Ubicación del script: symbolic-physics-discovery/src/data/generate_data.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta destino: symbolic-physics-discovery/results/
# Subimos dos niveles desde src/data para llegar a la raíz y entrar en results
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../results"))
DATA_OUT_DIR = os.path.join(RESULTS_DIR, "datasets")
PLOTS_OUT_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(DATA_OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_OUT_DIR, exist_ok=True)

# ==========================================
# Funciones de las leyes físicas (Adimensionales)
# ==========================================
def coulomb_law(q1, q2, r):
    return (q1 * q2) / (r**2)

def harmonic_oscillator(x):
    return -x

def kepler_third_law(r):
    return np.sqrt(r**3)

def ideal_gas_law(n, T, V):
    return (n * T) / V

def projectile_range(v0, theta):
    return v0**2 * np.sin(2 * theta)

def time_dilation(t, v):
    return t / np.sqrt(1 - v**2)

def radioactive_decay(lam, t):
    return np.exp(-lam * t)

def newton_cooling(k, t):
    T_AMB, T0 = 1.0, 2.0
    return T_AMB + (T0 - T_AMB) * np.exp(-k * t)

def boltzmann_entropy(omega):
    return np.log(omega)

# --- Funciones de Ruido ---
def add_noise(y, noise_level):
    '''Función que añade ruido gaussiano proporcional a la desviación estándar de y'''
    noise = noise_level * np.std(y) * np.random.randn(*y.shape)
    return y + noise 

#np.random.randn: Genera números aleatorios con distribución normal.
#*y.shape asegura que el array de ruido e y tengan la misma forma 
#np.std obtiene la desviación estándar de y, así se escala el ruido con la magnitud de los datos
#noise_level sirve para modular cuánto ruido se quiere añadir

# ==========================================
# Generación de datos (Escalas O(1))
# ==========================================
def generate_coulomb():
    q1 = np.random.uniform(0.1, 2.0, N_SAMPLES)
    q2 = np.random.uniform(0.1, 2.0, N_SAMPLES)
    r  = np.random.uniform(0.5, 3.0, N_SAMPLES)
    y  = coulomb_law(q1, q2, r)
    return np.column_stack((q1, q2, r, y))

def generate_oscillator():
    x = np.random.uniform(-2.0, 2.0, N_SAMPLES)
    y = harmonic_oscillator(x)
    return np.column_stack((x, y))

def generate_kepler():
    r = np.random.uniform(0.1, 3.0, N_SAMPLES)
    T = kepler_third_law(r)
    return np.column_stack((r, T))

def generate_ideal_gas():
    n = np.random.uniform(0.5, 2.0, N_SAMPLES)
    T = np.random.uniform(0.5, 2.0, N_SAMPLES)
    V = np.random.uniform(0.5, 2.0, N_SAMPLES)
    P = ideal_gas_law(n, T, V)
    return np.column_stack((n, T, V, P))

def generate_projectile_range():
    v0    = np.random.uniform(0.5, 3.0, N_SAMPLES)
    theta = np.random.uniform(0.1, np.pi / 2 - 0.1, N_SAMPLES)
    R     = projectile_range(v0, theta)
    return np.column_stack((v0, theta, R))

def generate_time_dilation():
    t = np.random.uniform(0.5, 2.0, N_SAMPLES)
    v = np.random.uniform(0.0, 0.95, N_SAMPLES) 
    t_prime = time_dilation(t, v)
    return np.column_stack((t, v, t_prime))

def generate_radioactive_decay():
    lam = np.random.uniform(0.1, 2.0, N_SAMPLES)
    t   = np.random.uniform(0.0, 3.0, N_SAMPLES)
    N   = radioactive_decay(lam, t)
    return np.column_stack((lam, t, N))

def generate_newton_cooling():
    k = np.random.uniform(0.1, 2.0, N_SAMPLES)
    t = np.random.uniform(0.0, 3.0, N_SAMPLES)
    T = newton_cooling(k, t)
    return np.column_stack((k, t, T))

def generate_boltzmann_entropy():
    omega = np.random.uniform(1.0, 10.0, N_SAMPLES)
    S     = boltzmann_entropy(omega)
    return np.column_stack((omega, S))

# Guarda los datos en un .csv con el título que se desee, los encabezados deseados
# y para los datasets en matrices 2D como se han generado con los generate_*
def save_dataset(filename, headers, data):
    full_path = os.path.join(DATA_OUT_DIR, filename)
    np.savetxt(
        full_path,
        data,
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.6e"
    )

def process_law(generate_func, headers, law_name):
    base_data = generate_func()
    for noise_name, noise_level in NOISE_LEVELS.items(): 
        data = base_data.copy()
        data[:, -1] = add_noise(data[:, -1], noise_level)
        save_dataset(f"{law_name}_{noise_name}.csv", headers, data) 
    return base_data

# ==========================================
# Módulo de Visualización Profesional
# ==========================================
LATEX_LABELS = {
    "q1": "q_1", "q2": "q_2", "r": "r", "F": "F",
    "x": "x", "T": "T", "n": "n", "V": "V", "P": "P",
    "v0": "v_0", "theta": r"\theta", "R": "R",
    "t": "t", "v": r"\beta", "t_prime": "t'",
    "lambda": r"\lambda", "N": "N", "k": "k",
    "omega": r"\Omega", "S": "S"
}

def plot_law(base_data, headers, title, x_idx, y_idx, law_name_file):
    plt.figure()
    colors = {"no_noise": "red", "low_noise": "green", "high_noise": "blue"}
    plot_labels = {"high_noise": "Ruido Alto", "low_noise": "Ruido Bajo", "no_noise": "Sin Ruido"}

    for noise_name in ["high_noise", "low_noise", "no_noise"]: 
        data = base_data.copy() 
        data[:, -1] = add_noise(data[:, -1], NOISE_LEVELS[noise_name])
        sorted_idx = np.argsort(data[:, x_idx])
        plt.plot(data[sorted_idx, x_idx], data[sorted_idx, y_idx], 
                 color=colors[noise_name], alpha=0.7, label=plot_labels[noise_name])

    plt.xlabel(rf"${LATEX_LABELS.get(headers[x_idx], headers[x_idx])}$") 
    plt.ylabel(rf"${LATEX_LABELS.get(headers[y_idx], headers[y_idx])}$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUT_DIR, f"{law_name_file}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    # Procesamiento y guardado
    laws = [
        (generate_coulomb, ["q1", "q2", "r", "F"], "coulomb", "Fuerza de Coulomb", 2, 3),
        (generate_oscillator, ["x", "F"], "oscillator", "Oscilador Armónico", 0, 1),
        (generate_kepler, ["r", "T"], "kepler", "Tercera Ley de Kepler", 0, 1),
        (generate_ideal_gas, ["n", "T", "V", "P"], "ideal_gas", "Ley de los Gases Ideales", 2, 3),
        (generate_projectile_range, ["v0", "theta", "R"], "projectile_range", "Alcance de Proyectil", 1, 2),
        (generate_time_dilation, ["t", "v", "t_prime"], "time_dilation", "Dilatación Temporal", 1, 2),
        (generate_radioactive_decay, ["lambda", "t", "N"], "radioactive_decay", "Desintegración Radiactiva", 1, 2),
        (generate_newton_cooling, ["k", "t", "T"], "newton_cooling", "Enfriamiento de Newton", 1, 2),
        (generate_boltzmann_entropy, ["omega", "S"], "boltzmann_entropy", "Entropía de Boltzmann", 0, 1)
    ]

    for gen_func, head, file_n, title, xi, yi in laws:
        data = process_law(gen_func, head, file_n)
        if plots:
            plot_law(data, head, title, xi, yi, file_n)

    print(f"Éxito. Resultados en: {RESULTS_DIR}")