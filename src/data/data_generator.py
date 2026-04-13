#importo las librerías
import numpy as np
import matplotlib.pyplot as plt
import os

plots = False

np.random.seed(1) # Aseguramos reproducibilidad

N_SAMPLES = 1000 # puntos para cada ley

NOISE_LEVELS = {
    "no_noise": 0.0,
    "low_noise": 0.01,
    "high_noise": 0.1
} # diferentes niveles de ruido que luego aplicaremos para distinguir los casos limpios de los ruidosos
# Resulta en una ponderación del 

# Carpeta donde está este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta destino: symbolic-physics-regression/data/
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../..", "data"))

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
    #np.random.randn: Genera números aleatorios con distribución normal.
    #*y.shape asegura que el array de ruido e y tengan la misma forma 
    #np.std obtiene la desviación estándar de y, así se escala el ruido con la magnitud de los datos
    #noise_level sirve para modular cuánto ruido se quiere añadir
    noise = noise_level * np.std(y) * np.random.randn(*y.shape)
    return y + noise #devuelve y + el ruido


def add_pink_noise(y, noise_level):
    """
    Genera ruido con densidad espectral de potencia proporcional a 1/f.
    Típico para sistemas de astrofísica o electrónica de semiconductores.
    """
    n = len(y)
    # Generamos ruido blanco en el dominio de la frecuencia
    white_fft = np.fft.rfft(np.random.randn(n))
    f = np.fft.rfftfreq(n)
    
    # El ruido rosa escala la amplitud por 1/sqrt(f) (potencia 1/f)
    f[0] = f[1] # Evitar división por cero en la componente DC
    scaler = 1 / np.sqrt(f)
    pink_fft = white_fft * scaler
    
    # Volvemos al dominio del tiempo
    pink_noise = np.fft.irfft(pink_fft, n)
    
    # Normalización y escalado proporcional a la señal y
    pink_noise = (pink_noise / np.std(pink_noise)) * (noise_level * np.std(y))
    
    return y + pink_noise


# Generación de datos
# column_stack aplica arrays 1D para formar columnas de una matriz de 2D
def generate_coulomb():
    #elijo valores aleatorios de q1, q2, r y obtengo las fuerzas de coulomb
    q1 = np.random.uniform(1e-6, 1e-3, N_SAMPLES)
    q2 = np.random.uniform(1e-6, 1e-3, N_SAMPLES)
    r = np.random.uniform(0.01, 1.0, N_SAMPLES)
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
    V = np.random.uniform(0.1, 10, N_SAMPLES)
    P = ideal_gas_law(n, T, V)
    return np.column_stack((n, T, V, P))


# Guarda los datos en un .csv con el título que se desee, los encabezados deseados
# y para los datasets en matrices 2D como se han generado con los generate_*
def save_dataset(filename, headers, data):
    """Guarda un dataset 2D en un archivo CSV con encabezados"""
    
    os.makedirs(DATA_DIR, exist_ok=True) # Crea la carpeta 'data' si no existe
    
    full_path = os.path.join(DATA_DIR, filename) # ruta completa del archivo
    
    #np.savetxt guarda directamente en CSV usando coma como separador
    #header añade los nombres de columnas separados por comas
    np.savetxt(
        full_path,
        data,
        delimiter=",",
        header=",".join(headers),
        comments="",  # evita que ponga "#" delante del header
        fmt="%.6e"
    )
    
    return 0


# Función que unifica las funciones anteriores y crea 3 archivos .csv (1 por tipo de ruido) por ley
def process_law(generate_func, headers, law_name):
    base_data = generate_func() # Llama a la función generate_func() para generar el dataset base sin ruido    
    
    #llamaremos noise_name a las claves (strings) y noise_level a los valores (floats)
    for noise_name, noise_level in NOISE_LEVELS.items(): # Itera sobre los distintos niveles de ruido definidos en NOISE_LEVELS
        data = base_data.copy() # Copia el dataset base
        data[:, -1] = add_noise(data[:, -1], noise_level) # Añade ruido solo a la columna dependiente (última columna del dataset)
        
        filename = f"{law_name}_{noise_name}.csv" # ahora guardamos como CSV
        
        save_dataset(filename, headers, data) # guarda el dataset con la función definida anteriormente
    
    return 0


# Plots -- hay que tener en cuenta que ni el gas ideal ni la ley de Coulomb son dependientes de una única variable
# por eso las relaciones que se muestran en el plot no son exactas (el resto de valores varían)
PLOTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "plots"))
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_law(generate_func, headers, law_name, x_idx, y_idx):
    """Genera plot de línea con los 3 niveles de ruido"""
    
    base_data = generate_func() #generamos los datos base
    plt.figure() #comenzamos con la figura
    
    colors = {"no_noise": "red", "low_noise": "green", "high_noise": "blue"} # definir los colores para cada nivel de ruido
    alpha_values = {"no_noise": 1.0, "low_noise": 0.8, "high_noise": 0.6} # añadimos algo de transparencia

    for noise_name in ["high_noise", "low_noise", "no_noise"]: # Orden de plot
        data = base_data.copy() # cargamos los valores de los resultados de las leyes
        data[:, -1] = add_noise(data[:, -1], NOISE_LEVELS[noise_name]) # añadimos ruido
        
        sorted_idx = np.argsort(data[:, x_idx]) # ordenamos los valores por x para que las líneas se vean correctamente
        
        plt.plot(
            data[sorted_idx, x_idx],
            data[sorted_idx, y_idx],
            color=colors[noise_name],
            alpha=alpha_values[noise_name],
            label=noise_name
        ) #ploteamos los datos

    # el nombre de los ejes son los de los encabezados
    plt.xlabel(headers[x_idx]) 
    plt.ylabel(headers[y_idx])
    
    plt.title(law_name) # el título es el de la ley de la que obtenemos los datos
    plt.legend() # mostramos la leyenda en la figura
    plt.tight_layout() # estilo de la figura
    
    plt.savefig(os.path.join(PLOTS_DIR, f"{law_name}.png")) # guardamos en plots
    plt.show() # mostramos la figura
    plt.close() # la cerramos


if __name__ == "__main__":
    
    # Coulomb
    process_law(generate_coulomb, ["q1", "q2", "r", "F"], "coulomb")

    # Oscilador
    process_law(generate_oscillator, ["x", "F"], "oscillator")

    # Kepler
    process_law(generate_kepler, ["r", "T"], "kepler")

    # Gas ideal
    process_law(generate_ideal_gas, ["n", "T", "V", "P"], "ideal_gas")

    if plots:
        plot_law(generate_coulomb, ["q1", "q2", "r", "F"], "Coulomb (F vs r)", x_idx=2, y_idx=3)
        plot_law(generate_oscillator, ["x", "F"], "Oscillator (F vs x)", x_idx=0, y_idx=1)
        plot_law(generate_kepler, ["r", "T"], "Kepler (T vs r)", x_idx=0, y_idx=1)
        plot_law(generate_ideal_gas, ["n", "T", "V", "P"], "Ideal Gas (P vs V)", x_idx=2, y_idx=3)


    print("Datasets y plots generados correctamente.")