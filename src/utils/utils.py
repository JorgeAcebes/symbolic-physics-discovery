import matplotlib.pyplot as plt

def set_plot_style(for_paper=False):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"], 
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": False if for_paper else True,
        "grid.alpha": 0 if for_paper else 0.3,
        "lines.linewidth": 1.5,
        "figure.titlesize": 14,
        "savefig.dpi": 300 if for_paper else 150,
        "savefig.bbox": "tight",
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{lmodern} \usepackage[T1]{fontenc}",
    })