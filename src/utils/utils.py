import matplotlib.pyplot as plt

def set_plot_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "figure.titlesize": 14,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

# Colores corporativos para usar en los plots
COLORS = {
    'theory': '#003B5C',
    'pysr': '#EAAA00',
    'mlp': "#6FC05B",
    'noise': '#C60C30'
}


'''
Ejemplo de Uso

import numpy as np
from utils import set_plot_style, COLORS

set_plot_style()

x = np.linspace(0.1, 10, 100)
y = x**1.5 # Ley de Kepler

plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$T \propto a^{3/2}$", color=COLORS['theory'])
plt.xlabel(r"Semieje mayor $a$ (UA)")
plt.ylabel(r"Periodo orbital $T$ (años)")
plt.title(r"\textbf{Recuperación de la Tercera Ley de Kepler}")
plt.legend()
plt.show()
'''