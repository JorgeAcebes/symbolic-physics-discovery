# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:11:53 2026

@author: jfcte
"""

import numpy as np

np.random.seed(42)

# =========================
# CONFIGURACIÓN GENERAL
# =========================
N_SAMPLES = 1000

NOISE_LEVELS = {
    "no_noise": 0.0,
    "low_noise": 0.01,
    "high_noise": 0.1
}

# =========================
# FUNCIONES DE LEYES FÍSICAS
# =========================

def coulomb_law(q1, q2, r):
    k = 8.99e9
    return k * (q1 * q2) / (r**2)

def harmonic_oscillator(x):
    k = 1.0
    return -k * x

def kepler_third_law(r):
    # T^2 = r^3  → T = sqrt(r^3)
    return np.sqrt(r**3)

def ideal_gas_law(n, T, V):
    R = 8.314
    return (n * R * T) / V


# =========================
# FUNCIÓN PARA AÑADIR RUIDO
# =========================

def add_noise(y, noise_level):
    noise = noise_level * np.std(y) * np.random.randn(*y.shape)
    return y + noise


# =========================
# GENERACIÓN DE DATOS
# =========================

def generate_coulomb():
    q1 = np.random.uniform(1e-6, 1e-3, N_SAMPLES)
    q2 = np.random.uniform(1e-6, 1e-3, N_SAMPLES)
    r = np.random.uniform(0.01, 1.0, N_SAMPLES)

    y = coulomb_law(q1, q2, r)
    return np.column_stack((q1, q2, r, y))


def generate_oscillator():
    x = np.random.uniform(-10, 10, N_SAMPLES)
    y = harmonic_oscillator(x)
    return np.column_stack((x, y))


def generate_kepler():
    r = np.random.uniform(0.1, 10, N_SAMPLES)
    T = kepler_third_law(r)
    return np.column_stack((r, T))


def generate_ideal_gas():
    n = np.random.uniform(0.1, 10, N_SAMPLES)
    T = np.random.uniform(100, 500, N_SAMPLES)
    V = np.random.uniform(0.1, 10, N_SAMPLES)

    P = ideal_gas_law(n, T, V)
    return np.column_stack((n, T, V, P))


# =========================
# GUARDADO DE DATOS
# =========================

def save_dataset(filename, headers, datasets):
    with open(filename, "w") as f:
        for label, data in datasets.items():
            f.write(f"# ===== {label} =====\n")
            f.write("# " + " ".join(headers) + "\n")
            np.savetxt(f, data, fmt="%.6e")
            f.write("\n\n")


# =========================
# PIPELINE PRINCIPAL
# =========================

def process_law(generate_func, headers, filename):
    base_data = generate_func()

    datasets = {}

    for noise_name, noise_level in NOISE_LEVELS.items():
        noisy_data = base_data.copy()
        y = noisy_data[:, -1]
        noisy_data[:, -1] = add_noise(y, noise_level)

        datasets[noise_name] = noisy_data

    save_dataset(filename, headers, datasets)


def main():
    # Ley de Coulomb
    process_law(
        generate_coulomb,
        headers=["q1", "q2", "r", "F"],
        filename="coulomb_data.txt"
    )

    # Oscilador armónico
    process_law(
        generate_oscillator,
        headers=["x", "F"],
        filename="oscillator_data.txt"
    )

    # Ley de Kepler
    process_law(
        generate_kepler,
        headers=["r", "T"],
        filename="kepler_data.txt"
    )

    # Gas ideal
    process_law(
        generate_ideal_gas,
        headers=["n", "T", "V", "P"],
        filename="ideal_gas_data.txt"
    )

    print("Datasets generados correctamente.")

import matplotlib.pyplot as plt
import os

# Crear carpeta para plots
os.makedirs("plots", exist_ok=True)


def plot_dataset(datasets, title, x_idx, y_idx, labels, filename):
    plt.figure()

    for name, data in datasets.items():
        x = data[:, x_idx]
        y = data[:, y_idx]
        plt.scatter(x, y, s=10, label=name)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"plots/{filename}")
    plt.show()
    plt.close()


def plot_all_laws():
    # Coulomb (usar r vs F para visualizar bien)
    base = generate_coulomb()
    datasets = {}
    for noise_name, noise_level in NOISE_LEVELS.items():
        d = base.copy()
        d[:, -1] = add_noise(d[:, -1], noise_level)
        datasets[noise_name] = d

    plot_dataset(
        datasets,
        title="Ley de Coulomb (F vs r)",
        x_idx=2,  # r
        y_idx=3,  # F
        labels=["r", "F"],
        filename="coulomb.png"
    )

    # Oscilador armónico
    base = generate_oscillator()
    datasets = {}
    for noise_name, noise_level in NOISE_LEVELS.items():
        d = base.copy()
        d[:, -1] = add_noise(d[:, -1], noise_level)
        datasets[noise_name] = d

    plot_dataset(
        datasets,
        title="Oscilador armónico (F vs x)",
        x_idx=0,
        y_idx=1,
        labels=["x", "F"],
        filename="oscillator.png"
    )

    # Kepler
    base = generate_kepler()
    datasets = {}
    for noise_name, noise_level in NOISE_LEVELS.items():
        d = base.copy()
        d[:, -1] = add_noise(d[:, -1], noise_level)
        datasets[noise_name] = d

    plot_dataset(
        datasets,
        title="Ley de Kepler (T vs r)",
        x_idx=0,
        y_idx=1,
        labels=["r", "T"],
        filename="kepler.png"
    )

    # Gas ideal (usar V vs P)
    base = generate_ideal_gas()
    datasets = {}
    for noise_name, noise_level in NOISE_LEVELS.items():
        d = base.copy()
        d[:, -1] = add_noise(d[:, -1], noise_level)
        datasets[noise_name] = d

    plot_dataset(
        datasets,
        title="Gas ideal (P vs V)",
        x_idx=2,  # V
        y_idx=3,  # P
        labels=["V", "P"],
        filename="ideal_gas.png"
    )

if __name__ == "__main__":
    main()
    plot_all_laws()