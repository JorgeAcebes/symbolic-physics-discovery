# -*- coding: utf-8 -*- 
"""
Created on Thu Mar 26 14:11:53 2026

@author: jfcte
"""
#importo las librerías
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(1) # inicializamos el random

N_SAMPLES = 1000 # puntos para cada ley

NOISE_LEVELS = {
    "no_noise": 0.0,
    "low_noise": 0.01,
    "high_noise": 0.1
} # diferentes niveles de ruido que luego aplicaremos para distinguir los casos limpios de los ruidosos

# Funciones de las leyes físicas
def coulomb_law(q1, q2, r):
    k = 8.99e9
    return k * (q1 * q2) / (r**2)
def harmonic_oscillator(x):
    k = 1.0
    return -k * x
def kepler_third_law(r):
    return np.sqrt(r**3)
def ideal_gas_law(n, T, V):
    R = 8.314
    return (n * R * T) / V

# Función que añade ruido gaussiano proporcional a la desviación estándar de y
def add_noise(y, noise_level):
    #np.random.rand Genera números aleatorios con distribución normal.
    #*y.shape asegura que el array de ruido e y tengan la misma forma 
    #np.std obtiene la desviación estándar de y, así se escala el ruido con la magnitud de los datos
    #noise_level sirve para modular cuánto ruido se quiere añadir
    noise = noise_level * np.std(y) * np.random.randn(*y.shape)
    return y + noise #devuelve y + el ruido

# Generación de datos
# column_stack aplica arrays 1D para formar columnas de una matriz de 2D
def generate_coulomb():
    #elijo valores aleatorios de q1, q2, r y obtengo las fuerzas de coulomb
    q1 = np.random.uniform(1e-6, 1e-3, N_SAMPLES)
    q2 = np.random.uniform(1e-6, 1e-3, N_SAMPLES)
    r = np.random.uniform(0.01, 1.0, N_SAMPLES)#r = np.linspace(0.01, 1.0, N_SAMPLES)#
    y = coulomb_law(q1, q2, r)
    return np.column_stack((q1, q2, r, y))

def generate_oscillator():
    #elijo valores aleatorios de x y obtengo las fuerzas de un muelle
    x = np.random.uniform(-10, 10, N_SAMPLES)
    y = harmonic_oscillator(x)
    return np.column_stack((x, y))

def generate_kepler():
    #obtengo valores aleatorios del perihelio y obtengo los periodos de traslación
    r = np.random.uniform(0.1, 10, N_SAMPLES)
    T = kepler_third_law(r)
    return np.column_stack((r, T))

def generate_ideal_gas():
    #obtengo valores aleatorios de n,T,V y obtengo la presión correspondiente
    n = np.random.uniform(0.1, 10, N_SAMPLES)
    T = np.random.uniform(100, 500, N_SAMPLES)
    V = np.random.uniform(0.1, 10, N_SAMPLES) #V = np.linspace(0.1, 10, N_SAMPLES)#
    P = ideal_gas_law(n, T, V)
    return np.column_stack((n, T, V, P))

# Guarda los datos en un .txt con el título que se desee, los encabezados deseado
# y para los datasets en matrices 2D como se han generado con los generate_*
def save_dataset(filename, headers, data):
    """Guarda un dataset 2D en un archivo con encabezados"""
    with open(filename, "w") as f:
        f.write("# " + " ".join(headers) + "\n") # escribe los encabezados de las columnas (ej: q1 q2 r F)
        np.savetxt(f, data, fmt="%.6e") # guarda la matriz 2D en el archivo
    return 0

# Función que unifica las funciones anteriores y crea 3 archivos .txt (1 por tipo de ruido) por ley
def process_law(generate_func, headers, law_name):
    base_data = generate_func() # Llama a la función generate_func() para generar el dataset base sin ruido    
    os.makedirs("datasets", exist_ok=True) # Crea la carpeta 'datasets' si no existe, para guardar los archivos
    
    #llamaremos noise_name a las claves (strings) y noise_level a los valores (floats)
    for noise_name, noise_level in NOISE_LEVELS.items(): # Itera sobre los distintos niveles de ruido definidos en NOISE_LEVELS
        data = base_data.copy() # Copia el dataset base
        data[:, -1] = add_noise(data[:, -1], noise_level) # Añade ruido solo a la columna dependiente (última columna del dataset)
        filename = f"datasets/{law_name}_{noise_name}.txt" # Nombra el archivo creado
        save_dataset(filename, headers, data) # guarda el dataset con la función definida anteriormente
    return 0

# Plots -- hay que tener en cuenta que ni el gas ideal ni la ley de Coulomb son dependientes de una única variable
# por eso las relaciones que se muestran en el plot no son exactas (el resto de valores varían)
os.makedirs("plots", exist_ok=True)

def plot_law(generate_func, headers, law_name, x_idx, y_idx):
    """Genera plot de línea con los 3 niveles de ruido"""
    base_data = generate_func()
    plt.figure()

    # Orden de plot: no_noise adelante, luego low_noise, luego high_noise al fondo
    colors = {"no_noise": "red", "low_noise": "green", "high_noise": "blue"}
    alpha_values = {"no_noise": 1.0, "low_noise": 0.7, "high_noise": 0.4}

    for noise_name in ["no_noise", "low_noise", "high_noise"]:
        data = base_data.copy()
        data[:, -1] = add_noise(data[:, -1], NOISE_LEVELS[noise_name])
        # ordenar por x para que las líneas se vean correctas
        sorted_idx = np.argsort(data[:, x_idx])
        plt.plot(data[sorted_idx, x_idx], data[sorted_idx, y_idx],
                 color=colors[noise_name], alpha=alpha_values[noise_name],
                 label=noise_name)

    plt.xlabel(headers[x_idx])
    plt.ylabel(headers[y_idx])
    plt.title(law_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{law_name}.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    
    # Coulomb
    process_law(generate_coulomb, ["q1", "q2", "r", "F"], "coulomb")
    plot_law(generate_coulomb, ["q1", "q2", "r", "F"], "Coulomb (F vs r)", x_idx=2, y_idx=3)

    # Oscilador
    process_law(generate_oscillator, ["x", "F"], "oscillator")
    plot_law(generate_oscillator, ["x", "F"], "Oscillator (F vs x)", x_idx=0, y_idx=1)

    # Kepler
    process_law(generate_kepler, ["r", "T"], "kepler")
    plot_law(generate_kepler, ["r", "T"], "Kepler (T vs r)", x_idx=0, y_idx=1)

    # Gas ideal
    process_law(generate_ideal_gas, ["n", "T", "V", "P"], "ideal_gas")
    plot_law(generate_ideal_gas, ["n", "T", "V", "P"], "Ideal Gas (P vs V)", x_idx=2, y_idx=3)

    print("Datasets y plots generados correctamente.")