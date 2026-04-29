"""
visualize_benchmark.py
======================
Reproduce todas las visualizaciones del dashboard de benchmark de
regresión simbólica:

  Fig 1 — Tarjetas de métricas globales
  Fig 2 — Heatmap de recuperación simbólica (3 niveles de ruido)
  Fig 3 — Scatter: tasa de recuperación vs. log-MSE global
  Fig 4 — Ranking global de modelos (barras horizontales)
  Fig 5 — Tabla de ecuaciones descubiertas (sin ruido)
  Fig 6 — MSE por ley y modelo (heatmap numérico)
  Fig 7 — Evolución del MSE con el nivel de ruido por modelo
  Fig 8 — Radar: perfil de recuperación por modelo
  Fig 9 — Análisis de robustez al ruido (degradación de MSE)

Uso:
    python visualize_benchmark.py                # guarda PNGs en ./figures/
    python visualize_benchmark.py --show         # además abre las ventanas
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.table import Table

warnings.filterwarnings("ignore")

# ── salida ──────────────────────────────────────────────────────────────────
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── paleta / estilo global ───────────────────────────────────────────────────
STYLE = {
    "bg":        "#FAFAF8",
    "card":      "#F3F2EE",
    "border":    "#D3D1C7",
    "text":      "#2C2C2A",
    "muted":     "#888780",
    "ok":        "#639922",   # verde  (correcto)
    "approx":    "#BA7517",   # ámbar  (aproximado)
    "wrong":     "#D3D1C7",   # gris   (incorrecto)
    "danger":    "#A32D2D",
}
MODEL_COLORS = {
    "GPLearn":     "#378ADD",
    "PySR":        "#639922",
    "QLattice":    "#BA7517",
    "PySINDy":     "#1D9E75",
    "Polynomial":  "#888780",
    "MLP_Standard":"#D4537E",
    "MLP_Sparse":  "#D85A30",
    "MLP_Dropout": "#7F77DD",
}

plt.rcParams.update({
    "figure.facecolor":  STYLE["bg"],
    "axes.facecolor":    STYLE["bg"],
    "axes.edgecolor":    STYLE["border"],
    "axes.labelcolor":   STYLE["text"],
    "text.color":        STYLE["text"],
    "xtick.color":       STYLE["muted"],
    "ytick.color":       STYLE["muted"],
    "grid.color":        STYLE["border"],
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

# ═══════════════════════════════════════════════════════════════════════════════
# DATOS
# ═══════════════════════════════════════════════════════════════════════════════

LAWS = [
    ("boltzmann",      "Boltzmann entropy"),
    ("coulomb",        "Coulomb"),
    ("ideal_gas",      "Ideal gas"),
    ("kepler",         "Kepler"),
    ("newton_cooling", "Newton cooling"),
    ("oscillator",     "Oscillator"),
    ("projectile",     "Projectile range"),
    ("radioactive",    "Radioactive decay"),
    ("time_dilation",  "Time dilation"),
]
LAW_KEYS  = [k for k, _ in LAWS]
LAW_NAMES = [n for _, n in LAWS]

MODELS = ["GPLearn", "PySR", "QLattice", "PySINDy",
          "Polynomial", "MLP_Standard", "MLP_Sparse", "MLP_Dropout"]
SYMBOLIC_MODELS = {"GPLearn", "PySR", "QLattice", "PySINDy"}

# ── Recuperación simbólica: 1=exacta, 0.5=aprox, 0=fallo ────────────────────
# Dimensión: (modelo, ley, nivel_ruido=[sin ruido, bajo, alto])
RECOVERY = {
    "GPLearn":    {"boltzmann":[1,1,1],"coulomb":[1,1,1],"ideal_gas":[1,1,1],
                   "kepler":[1,1,1],"newton_cooling":[0,0,0],"oscillator":[1,1,1],
                   "projectile":[0,0,0],"radioactive":[0,0,0],"time_dilation":[0,0,0]},
    "PySR":       {"boltzmann":[1,1,1],"coulomb":[1,1,1],"ideal_gas":[1,1,1],
                   "kepler":[1,1,1],"newton_cooling":[1,1,1],"oscillator":[1,1,1],
                   "projectile":[1,1,1],"radioactive":[1,1,1],"time_dilation":[0,0,0]},
    "QLattice":   {"boltzmann":[.5,.5,.5],"coulomb":[0,0,0],"ideal_gas":[1,.5,.5],
                   "kepler":[.5,.5,.5],"newton_cooling":[1,1,1],"oscillator":[.5,1,1],
                   "projectile":[0,0,0],"radioactive":[1,1,1],"time_dilation":[0,0,0]},
    "PySINDy":    {"boltzmann":[0,0,0],"coulomb":[0,0,0],"ideal_gas":[0,0,0],
                   "kepler":[0,0,0],"newton_cooling":[0,0,0],"oscillator":[1,1,1],
                   "projectile":[0,0,0],"radioactive":[0,0,0],"time_dilation":[0,0,0]},
    "Polynomial": {"boltzmann":[0,0,0],"coulomb":[0,0,0],"ideal_gas":[0,0,0],
                   "kepler":[0,0,0],"newton_cooling":[0,0,0],"oscillator":[1,1,1],
                   "projectile":[0,0,0],"radioactive":[0,0,0],"time_dilation":[0,0,0]},
    "MLP_Standard":{"boltzmann":[0,0,0],"coulomb":[0,0,0],"ideal_gas":[0,0,0],
                    "kepler":[0,0,0],"newton_cooling":[0,0,0],"oscillator":[0,0,0],
                    "projectile":[0,0,0],"radioactive":[0,0,0],"time_dilation":[0,0,0]},
    "MLP_Sparse": {"boltzmann":[0,0,0],"coulomb":[0,0,0],"ideal_gas":[0,0,0],
                   "kepler":[0,0,0],"newton_cooling":[0,0,0],"oscillator":[0,0,0],
                   "projectile":[0,0,0],"radioactive":[0,0,0],"time_dilation":[0,0,0]},
    "MLP_Dropout":{"boltzmann":[0,0,0],"coulomb":[0,0,0],"ideal_gas":[0,0,0],
                   "kepler":[0,0,0],"newton_cooling":[0,0,0],"oscillator":[0,0,0],
                   "projectile":[0,0,0],"radioactive":[0,0,0],"time_dilation":[0,0,0]},
}

# ── MSE promedio por (modelo, ley) ── todos los niveles de ruido ─────────────
MSE_GLOBAL = {
    "GPLearn":    {"boltzmann":1.18e-3,"coulomb":5.79e-3,"ideal_gas":3.28e-3,
                   "kepler":7.54e-3,"newton_cooling":9.16e-2,"oscillator":4.77e-3,
                   "projectile":4.58e-1,"radioactive":5.87e-2,"time_dilation":1.66e-1},
    "PySR":       {"boltzmann":1.18e-3,"coulomb":5.79e-3,"ideal_gas":3.27e-3,
                   "kepler":7.54e-3,"newton_cooling":3.18e-4,"oscillator":4.75e-3,
                   "projectile":1.61e-2,"radioactive":2.90e-4,"time_dilation":3.06e-3},
    "QLattice":   {"boltzmann":1.19e-3,"coulomb":6.60e-3,"ideal_gas":3.52e-3,
                   "kepler":7.61e-3,"newton_cooling":3.18e-4,"oscillator":4.81e-3,
                   "projectile":1.76e-2,"radioactive":2.90e-4,"time_dilation":2.41e-3},
    "PySINDy":    {"boltzmann":5.03e-3,"coulomb":4.34e-2,"ideal_gas":4.98e-3,
                   "kepler":7.69e-3,"newton_cooling":5.29e-4,"oscillator":4.72e-3,
                   "projectile":1.76e-2,"radioactive":4.87e-4,"time_dilation":1.25e-2},
    "Polynomial": {"boltzmann":1.20e-3,"coulomb":1.19e-2,"ideal_gas":3.33e-3,
                   "kepler":7.52e-3,"newton_cooling":3.28e-4,"oscillator":4.75e-3,
                   "projectile":1.62e-2,"radioactive":3.01e-4,"time_dilation":3.90e-3},
    "MLP_Standard":{"boltzmann":1.19e-3,"coulomb":6.47e-3,"ideal_gas":3.41e-3,
                    "kepler":8.25e-3,"newton_cooling":3.29e-4,"oscillator":4.76e-3,
                    "projectile":1.69e-2,"radioactive":3.08e-4,"time_dilation":2.42e-3},
    "MLP_Sparse": {"boltzmann":1.19e-3,"coulomb":6.70e-3,"ideal_gas":3.55e-3,
                   "kepler":7.75e-3,"newton_cooling":3.38e-4,"oscillator":4.80e-3,
                   "projectile":1.66e-2,"radioactive":3.07e-4,"time_dilation":2.41e-3},
    "MLP_Dropout":{"boltzmann":1.50e-3,"coulomb":1.93e-2,"ideal_gas":6.96e-3,
                   "kepler":1.07e-2,"newton_cooling":5.47e-4,"oscillator":7.27e-3,
                   "projectile":2.99e-2,"radioactive":5.02e-4,"time_dilation":4.96e-3},
}

# ── MSE por nivel de ruido (sin ruido / bajo / alto) ─────────────────────────
MSE_BY_NOISE = {
    "GPLearn":    {"boltzmann":[7.95e-14,3.48e-5,3.50e-3],"coulomb":[3.01e-13,1.76e-4,1.72e-2],
                   "ideal_gas":[3.21e-13,9.35e-5,9.73e-3],"kepler":[3.41e-13,2.32e-4,2.24e-2],
                   "newton_cooling":[9.13e-2,9.13e-2,9.21e-2],"oscillator":[3.51e-5,1.81e-4,1.41e-2],
                   "projectile":[4.41e-1,4.41e-1,4.92e-1],"radioactive":[5.85e-2,5.85e-2,5.92e-2],
                   "time_dilation":[1.64e-1,1.64e-1,1.71e-1]},
    "PySR":       {"boltzmann":[7.95e-14,3.48e-5,3.50e-3],"coulomb":[3.14e-13,1.76e-4,1.72e-2],
                   "ideal_gas":[3.21e-13,9.35e-5,9.73e-3],"kepler":[4.04e-11,2.32e-4,2.24e-2],
                   "newton_cooling":[8.78e-14,8.76e-6,9.44e-4],"oscillator":[0.0,1.42e-4,1.41e-2],
                   "projectile":[2.14e-12,4.71e-4,4.79e-2],"radioactive":[4.70e-15,8.55e-6,8.60e-4],
                   "time_dilation":[1.48e-4,1.25e-4,8.92e-3]},
    "QLattice":   {"boltzmann":[1.08e-12,3.56e-5,3.52e-3],"coulomb":[1.42e-3,2.72e-4,1.80e-2],
                   "ideal_gas":[1.24e-8,3.42e-4,1.02e-2],"kepler":[1.80e-7,2.38e-4,2.26e-2],
                   "newton_cooling":[6.22e-10,9.27e-6,9.43e-4],"oscillator":[5.37e-15,1.45e-4,1.43e-2],
                   "projectile":[3.97e-4,2.45e-3,4.98e-2],"radioactive":[1.48e-12,8.70e-6,8.61e-4],
                   "time_dilation":[1.52e-4,2.04e-4,6.88e-3]},
    "PySINDy":    {"boltzmann":[8.55e-5,1.23e-4,1.47e-2],"coulomb":[3.79e-2,3.81e-2,5.41e-2],
                   "ideal_gas":[1.76e-3,1.69e-3,1.15e-2],"kepler":[2.20e-5,6.22e-4,2.24e-2],
                   "newton_cooling":[2.01e-4,2.07e-4,1.18e-3],"oscillator":[1.92e-14,1.42e-4,1.40e-2],
                   "projectile":[9.44e-4,1.43e-3,5.04e-2],"radioactive":[1.97e-4,2.05e-4,1.06e-3],
                   "time_dilation":[1.01e-2,1.02e-2,1.72e-2]},
    "Polynomial": {"boltzmann":[1.49e-5,4.96e-5,3.53e-3],"coulomb":[6.54e-3,6.64e-3,2.37e-2],
                   "ideal_gas":[3.15e-5,1.27e-4,9.83e-3],"kepler":[3.45e-7,2.32e-4,2.23e-2],
                   "newton_cooling":[9.55e-6,1.83e-5,9.55e-4],"oscillator":[8.37e-14,1.42e-4,1.41e-2],
                   "projectile":[2.41e-5,5.00e-4,4.81e-2],"radioactive":[9.21e-6,1.78e-5,8.77e-4],
                   "time_dilation":[1.62e-3,1.70e-3,8.38e-3]},
    "MLP_Standard":{"boltzmann":[3.53e-3,5.47e-5,3.53e-3],"coulomb":[2.03e-4,3.21e-4,1.89e-2],
                    "ideal_gas":[1.81e-5,1.13e-4,1.01e-2],"kepler":[2.01e-6,2.48e-4,2.45e-2],
                    "newton_cooling":[2.93e-7,1.61e-5,9.72e-4],"oscillator":[4.09e-9,1.60e-4,1.41e-2],
                    "projectile":[5.52e-5,6.15e-4,5.00e-2],"radioactive":[9.74e-7,1.19e-5,9.12e-4],
                    "time_dilation":[3.00e-5,1.15e-4,7.11e-3]},
    "MLP_Sparse": {"boltzmann":[9.19e-6,4.60e-5,3.52e-3],"coulomb":[7.16e-4,7.11e-4,1.87e-2],
                   "ideal_gas":[9.73e-5,2.12e-4,1.03e-2],"kepler":[3.17e-5,2.66e-4,2.29e-2],
                   "newton_cooling":[9.82e-6,1.99e-5,9.83e-4],"oscillator":[2.23e-5,1.99e-4,1.42e-2],
                   "projectile":[2.79e-4,7.81e-4,4.89e-2],"radioactive":[1.70e-5,1.61e-5,8.88e-4],
                   "time_dilation":[5.76e-5,1.30e-4,7.04e-3]},
    "MLP_Dropout":{"boltzmann":[3.68e-4,2.63e-4,3.88e-3],"coulomb":[8.97e-3,6.70e-3,4.28e-2],
                   "ideal_gas":[2.25e-3,3.05e-3,1.59e-2],"kepler":[3.74e-3,1.59e-3,2.63e-2],
                   "newton_cooling":[1.41e-4,1.38e-4,1.36e-3],"oscillator":[1.36e-3,3.45e-3,1.70e-2],
                   "projectile":[1.43e-2,1.33e-2,6.31e-2],"radioactive":[3.12e-4,1.48e-4,1.05e-3],
                   "time_dilation":[2.22e-3,3.17e-3,9.49e-3]},
}

# ── Ecuaciones descubiertas (sin ruido) ───────────────────────────────────────
EQUATIONS = {
    "boltzmann": {
        "Verdad":    "log(ω)",
        "GPLearn":   ("log(ω)",               1),
        "PySR":      ("log(ω)",               1),
        "QLattice":  ("1.0·log(0.408ω) + 0.897", 0.5),
        "PySINDy":   ("0.231ω + 2.22/ω − 2.89/ω²", 0),
        "Polynomial":("0.0001ω⁵−0.004ω⁴+...−1.052", 0),
    },
    "coulomb": {
        "Verdad":    "q₁q₂/r²",
        "GPLearn":   ("q₁q₂/r²",              1),
        "PySR":      ("q₁q₂/r²",              1),
        "QLattice":  ("2.04·f(q₁,q₂,r) ≈ q₁q₂/r²", 0),
        "PySINDy":   ("poly(q₁,q₂,r) de grado 4", 0),
        "Polynomial":("poly de 56 términos",   0),
    },
    "ideal_gas": {
        "Verdad":    "nT/V",
        "GPLearn":   ("T·n/V",                1),
        "PySR":      ("T·n/V",                1),
        "QLattice":  ("−0.783·(4.93T)·(−0.584n)/(2.255V)", 1),
        "PySINDy":   ("poly(n,T,V) 35 términos", 0),
        "Polynomial":("0.0096T⁵−0.0007T⁴V+...", 0),
    },
    "kepler": {
        "Verdad":    "r^(3/2)",
        "GPLearn":   ("r^(3/2)",              1),
        "PySR":      ("r^1.4999973",          1),
        "QLattice":  ("−1.48·(−0.511r−0.009)·√(1.746r) ≈ r^1.5", 0.5),
        "PySINDy":   ("−0.052 + 0.579r + 0.514r² − 0.042r³", 0),
        "Polynomial":("−0.006r⁵+0.056r⁴+...", 0),
    },
    "newton_cooling": {
        "Verdad":    "1 + e^(−kt)",
        "GPLearn":   ("1.366  ← FALLO TOTAL", 0),
        "PySR":      ("1.0 + e^(−kt)",       1),
        "QLattice":  ("1.0·e^(−0.842t·1.187k) + 1.0", 1),
        "PySINDy":   ("2.0 + 0.20k² − 1.50kt + ...", 0),
        "Polynomial":("−0.032k⁵+0.074k⁴t+...", 0),
    },
    "oscillator": {
        "Verdad":    "−x",
        "GPLearn":   ("−0.995x",             1),
        "PySR":      ("−x−x+x = −x",         1),
        "QLattice":  ("≈ −x (forma implícita)", 0.5),
        "PySINDy":   ("−1.000x",             1),
        "Polynomial":("≈ −x (grado 5, coefs ~0)", 1),
    },
    "projectile": {
        "Verdad":    "v₀²·sin(2θ)",
        "GPLearn":   ("v₀^(3/2)·√(log(θ−0.761)) ← FALLO", 0),
        "PySR":      ("v₀²·sin(θ+θ)",        1),
        "QLattice":  ("11.79 − 15.61·e^(−2·(...))", 0),
        "PySINDy":   ("poly(v₀,θ) 8 términos", 0),
        "Polynomial":("poly(v₀,θ) 21 términos", 0),
    },
    "radioactive": {
        "Verdad":    "e^(−λt)",
        "GPLearn":   ("cos(√t)  ← FALLO TOTAL", 0),
        "PySR":      ("e^(λ)^(−t) = e^(−λt)", 1),
        "QLattice":  ("1.0·e^(−0.977t·1.023λ) ≈ e^(−λt)", 1),
        "PySINDy":   ("0.985 + 0.114λ − 1.51λt + ...", 0),
        "Polynomial":("−0.028λ⁵+0.077λ⁴t+...", 0),
    },
    "time_dilation": {
        "Verdad":    "t/√(1−v²)",
        "GPLearn":   ("t·(v+1)  ← FALLO TOTAL", 0),
        "PySR":      ("t·(1.69 + 1/cos(v^1.36+0.5))·0.352", 0),
        "QLattice":  ("−2.10·(1.04−0.79v)·(2.14t)/(4.61v−4.78)", 0),
        "PySINDy":   ("poly(t,v) 17 términos", 0),
        "Polynomial":("poly(t,v) 21 términos", 0),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"], edgecolor="none")
    print(f"  Guardado: {path}")


def rec_score(model):
    """Tasa de recuperación ponderada sobre las 27 condiciones (9 leyes × 3 ruidos)."""
    vals = []
    for lk in LAW_KEYS:
        vals.extend(RECOVERY[model][lk])
    return np.mean(vals)


def avg_log_mse(model):
    mse_vals = [MSE_GLOBAL[model][lk] for lk in LAW_KEYS]
    return np.mean(np.log10(mse_vals))


def section_title(ax, text):
    ax.set_title(text, loc="left", fontsize=9, fontweight="500",
                 color=STYLE["muted"], pad=10,
                 fontfamily="DejaVu Sans")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — TARJETAS DE MÉTRICAS GLOBALES
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_metric_cards():
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.2))
    fig.patch.set_facecolor(STYLE["bg"])

    cards = [
        ("23 / 27", "PySR — mejor recuperación simbólica (85%)", STYLE["ok"]),
        ("15 / 27", "GPLearn — falla catastróficamente en 4 leyes", STYLE["text"]),
        ("1 ley",   "Sin resolver por ningún modelo (Time dilation)", STYLE["danger"]),
        ("0 / 27",  "Recuperaciones simbólicas por MLPs", STYLE["muted"]),
    ]

    for ax, (val, sub, col) in zip(axes, cards):
        ax.set_facecolor(STYLE["card"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["border"])
            spine.set_linewidth(0.7)
        ax.set_xticks([]); ax.set_yticks([])

        ax.text(0.5, 0.65, val, transform=ax.transAxes,
                ha="center", va="center", fontsize=22, fontweight="500", color=col)
        ax.text(0.5, 0.22, sub, transform=ax.transAxes,
                ha="center", va="center", fontsize=8.5, color=STYLE["muted"],
                wrap=True)

    fig.suptitle("Benchmark de regresión simbólica — métricas globales",
                 fontsize=11, fontweight="500", y=1.04, color=STYLE["text"])
    fig.tight_layout(pad=0.6)
    savefig(fig, "fig1_metric_cards.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — HEATMAP DE RECUPERACIÓN SIMBÓLICA
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_recovery_heatmap():
    """
    Cada celda muestra 3 marcadores (sin ruido · bajo · alto).
    Color: verde=exacto, ámbar=aprox, gris=fallo.
    """
    n_models = len(MODELS)
    n_laws   = len(LAW_KEYS)
    NOISE_LABELS = ["sin ruido", "bajo", "alto"]

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_facecolor(STYLE["bg"])
    ax.set_aspect("equal")

    cell_w, cell_h = 1.0, 1.0
    dot_r = 0.11
    offsets_x = [-0.27, 0.0, 0.27]

    for mi, model in enumerate(MODELS):
        for li, lk in enumerate(LAW_KEYS):
            vals = RECOVERY[model][lk]
            cx = li * cell_w
            cy = (n_models - 1 - mi) * cell_h

            # fondo de celda
            rect = mpatches.FancyBboxPatch(
                (cx - 0.45, cy - 0.40), 0.90, 0.80,
                boxstyle="round,pad=0.04", linewidth=0,
                facecolor=STYLE["card"])
            ax.add_patch(rect)

            for ni, v in enumerate(vals):
                col = STYLE["ok"] if v == 1 else (STYLE["approx"] if v == 0.5 else STYLE["wrong"])
                circ = plt.Circle((cx + offsets_x[ni], cy), dot_r,
                                  color=col, zorder=3)
                ax.add_patch(circ)

    # etiquetas
    for li, name in enumerate(LAW_NAMES):
        ax.text(li * cell_w, n_models - 0.05, name, ha="center", va="bottom",
                fontsize=8, color=STYLE["muted"], rotation=30)
    for mi, model in enumerate(MODELS):
        ax.text(-0.6, (n_models - 1 - mi) * cell_h, model,
                ha="right", va="center", fontsize=9, color=STYLE["text"])

    # cabecera de ruido
    for ni, nl in enumerate(NOISE_LABELS):
        ax.text(-0.6 + (ni + 1) * 0.22, n_models + 0.7, nl,
                ha="center", va="center", fontsize=7, color=STYLE["muted"])

    # leyenda
    for label, col in [("Exacta", STYLE["ok"]),
                        ("Aproximada", STYLE["approx"]),
                        ("Incorrecta", STYLE["wrong"])]:
        ax.scatter([], [], s=60, color=col, label=label)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5,
              labelcolor=STYLE["muted"])

    ax.set_xlim(-1.1, n_laws - 0.5)
    ax.set_ylim(-0.6, n_models + 0.9)
    ax.axis("off")
    section_title(ax, "RECUPERACIÓN SIMBÓLICA POR MODELO Y LEY  ·  3 puntos = sin ruido · bajo · alto")

    fig.tight_layout()
    savefig(fig, "fig2_recovery_heatmap.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — SCATTER: TASA DE RECUPERACIÓN vs. LOG-MSE
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_scatter():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    offsets = {
        "GPLearn":     (4, -12),
        "PySR":        (4,   8),
        "QLattice":    (4,   0),
        "PySINDy":     (4,  -12),
        "Polynomial":  (-80, 6),
        "MLP_Standard":(4, -12),
        "MLP_Sparse":  (4,   4),
        "MLP_Dropout": (4,  14),
    }

    for model in MODELS:
        x = rec_score(model) * 100
        y = avg_log_mse(model)
        col = MODEL_COLORS[model]
        marker = "D" if model in SYMBOLIC_MODELS else "o"
        ax.scatter(x, y, s=140, color=col, marker=marker,
                   zorder=4, edgecolors="white", linewidths=0.8)
        dx, dy = offsets.get(model, (4, 0))
        ax.annotate(model, (x, y), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8, color=col, va="center")

    ax.axvline(50, color=STYLE["border"], lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Tasa de recuperación simbólica (%)", fontsize=9)
    ax.set_ylabel("log₁₀(MSE promedio global)", fontsize=9)
    ax.grid(True, alpha=0.5)

    ax.scatter([], [], marker="D", s=80, color=STYLE["text"],
               label="Regresor simbólico", edgecolors="white", lw=0.8)
    ax.scatter([], [], marker="o", s=80, color=STYLE["text"],
               label="Caja negra", edgecolors="white", lw=0.8)
    ax.legend(fontsize=8, frameon=False)

    section_title(ax, "TASA DE RECUPERACIÓN VS. PRECISIÓN NUMÉRICA GLOBAL")
    fig.tight_layout()
    savefig(fig, "fig3_scatter_recovery_vs_mse.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — RANKING GLOBAL (BARRAS HORIZONTALES)
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_ranking():
    stats = []
    for model in MODELS:
        score = rec_score(model)
        exact = sum(1 for lk in LAW_KEYS
                    for v in RECOVERY[model][lk] if v == 1)
        approx = sum(1 for lk in LAW_KEYS
                     for v in RECOVERY[model][lk] if v == 0.5)
        mse = np.mean([MSE_GLOBAL[model][lk] for lk in LAW_KEYS])
        composite = score * 0.7 - np.log10(mse) * 0.05
        stats.append((model, score, exact, approx, mse, composite))

    stats.sort(key=lambda r: r[5], reverse=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(stats))
    bar_h = 0.55

    for i, (model, score, exact, approx, mse, _) in enumerate(stats):
        col = MODEL_COLORS[model]
        # barra exactas
        w_ex = exact / 27
        # barra aprox (apilada)
        w_ap = approx / 27

        ax.barh(i, w_ex, height=bar_h, color=col, alpha=0.90, zorder=3)
        ax.barh(i, w_ap, left=w_ex, height=bar_h,
                color=col, alpha=0.35, hatch="//", zorder=3)

        pct = (w_ex + w_ap) * 100
        ax.text(w_ex + w_ap + 0.01, i, f"{pct:.0f}%",
                va="center", fontsize=8.5, color=col)
        ax.text(w_ex + w_ap + 0.08, i,
                f"MSE={mse:.2e}",
                va="center", fontsize=7.5, color=STYLE["muted"])

        sym_tag = "◆ Simbólico" if model in SYMBOLIC_MODELS else "● Caja negra"
        ax.text(-0.01, i, sym_tag, ha="right", va="center",
                fontsize=6.5, color=STYLE["muted"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in stats], fontsize=9)
    ax.set_xlim(-0.22, 1.08)
    ax.set_xlabel("Tasa de recuperación simbólica", fontsize=9)
    ax.axvline(0, color=STYLE["border"], lw=0.5)
    ax.grid(axis="x", alpha=0.4)
    ax.invert_yaxis()

    # leyenda hatching
    ax.barh([], [], color=STYLE["muted"], alpha=0.90, label="Recuperación exacta")
    ax.barh([], [], color=STYLE["muted"], alpha=0.35, hatch="//",
            label="Recuperación aprox.")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    section_title(ax, "RANKING GLOBAL DE MODELOS")
    fig.tight_layout()
    savefig(fig, "fig4_ranking.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — TABLA DE ECUACIONES DESCUBIERTAS (SIN RUIDO)
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_equations_table():
    cols_show = ["GPLearn", "PySR", "QLattice", "PySINDy", "Polynomial"]
    n_rows = len(LAWS)
    n_cols = len(cols_show) + 2  # Ley + Verdad + 5 modelos

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.axis("off")

    col_widths = [0.09, 0.08] + [0.165] * 5
    headers = ["Ley", "Verdad"] + cols_show
    header_colors = [STYLE["card"]] * n_cols
    row_data = []
    cell_colors = []

    for lk, lname in LAWS:
        eq_dict = EQUATIONS[lk]
        row = [lname, eq_dict["Verdad"]]
        ccol = [STYLE["card"], STYLE["card"]]
        for m in cols_show:
            eq_text, score = eq_dict[m]
            # truncar si es muy largo
            if len(eq_text) > 38:
                eq_text = eq_text[:36] + "…"
            row.append(eq_text)
            if score == 1:
                ccol.append("#E8F3D6")   # verde claro
            elif score == 0.5:
                ccol.append("#FAF0D7")   # ámbar claro
            else:
                ccol.append("#FBEBEB")   # rojo claro
        row_data.append(row)
        cell_colors.append(ccol)

    tbl = ax.table(
        cellText=row_data,
        colLabels=headers,
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    # header style
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(STYLE["card"])
        cell.set_text_props(color=STYLE["muted"], fontweight="bold")
        cell.set_edgecolor(STYLE["border"])

    # content style
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor(cell_colors[i - 1][j])
            cell.set_edgecolor(STYLE["border"])
            if j < 2:
                cell.set_text_props(color=STYLE["text"], fontweight="500")
            else:
                cell.set_text_props(color=STYLE["text"])
            cell.PAD = 0.04

    # leyenda
    for label, col in [("Exacta", "#E8F3D6"),
                        ("Aproximada", "#FAF0D7"),
                        ("Incorrecta", "#FBEBEB")]:
        ax.barh([], [], color=col, label=label)
    ax.legend(loc="lower right", fontsize=8, frameon=True,
              framealpha=0.9, edgecolor=STYLE["border"])

    section_title(ax, "ECUACIONES DESCUBIERTAS — EXPERIMENTOS SIN RUIDO")
    fig.tight_layout()
    savefig(fig, "fig5_equations_table.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — HEATMAP NUMÉRICO DE MSE POR LEY Y MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_mse_heatmap():
    # construir matriz log10(MSE)
    matrix = np.zeros((len(MODELS), len(LAW_KEYS)))
    for mi, m in enumerate(MODELS):
        for li, lk in enumerate(LAW_KEYS):
            matrix[mi, li] = np.log10(MSE_GLOBAL[m][lk])

    fig, ax = plt.subplots(figsize=(13, 5.5))

    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#E8F3D6", "#FAF0D7", "#FBEBEB"])
    # invertir: menor log-MSE = mejor = verde
    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   vmin=matrix.min(), vmax=matrix.max())

    ax.set_xticks(range(len(LAW_NAMES)))
    ax.set_xticklabels(LAW_NAMES, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=9)

    # anotar valores
    for mi in range(len(MODELS)):
        for li in range(len(LAW_KEYS)):
            v = MSE_GLOBAL[MODELS[mi]][LAW_KEYS[li]]
            ax.text(li, mi, f"{v:.1e}", ha="center", va="center",
                    fontsize=6.5, color=STYLE["text"])

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("log₁₀(MSE)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    section_title(ax, "MSE NUMÉRICO POR LEY Y MODELO  (verde = mejor)")
    fig.tight_layout()
    savefig(fig, "fig6_mse_heatmap.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — EVOLUCIÓN DEL MSE CON EL NIVEL DE RUIDO
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_noise_sensitivity():
    noise_levels = [0, 1, 2]
    noise_labels  = ["sin ruido", "ruido bajo", "ruido alto"]

    # MSE agregado por nivel de ruido para cada modelo
    n_rows, n_cols = 2, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6.5), sharey=False)
    axes = axes.flatten()

    for idx, (lk, lname) in enumerate(LAWS[:9]):
        ax = axes[idx]
        for model in MODELS:
            vals = [MSE_BY_NOISE[model][lk][ni] for ni in noise_levels]
            ylog = [np.log10(v) if v > 0 else -16 for v in vals]
            ax.plot(noise_levels, ylog,
                    color=MODEL_COLORS[model], linewidth=1.4,
                    marker="o", markersize=4, label=model)

        ax.set_xticks(noise_levels)
        ax.set_xticklabels(noise_labels, fontsize=6.5, rotation=15)
        ax.set_title(lname, fontsize=8.5, fontweight="500", color=STYLE["text"])
        ax.set_ylabel("log₁₀(MSE)", fontsize=7)
        ax.grid(True, alpha=0.4)

    # quitar el décimo subplot si sobra
    if len(LAWS) < len(axes):
        for ax in axes[len(LAWS):]:
            ax.axis("off")

    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=m) for m in MODELS]
    fig.legend(handles=handles, loc="lower right", ncol=2,
               fontsize=7.5, frameon=False)

    fig.suptitle("Evolución del MSE con el nivel de ruido — por ley física",
                 fontsize=11, fontweight="500", y=1.02, color=STYLE["text"])
    fig.tight_layout()
    savefig(fig, "fig7_noise_sensitivity.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — RADAR: PERFIL DE RECUPERACIÓN POR MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def fig8_radar():
    """Diagrama de araña: cada eje = una ley, valor = tasa de recuperación."""
    from matplotlib.patches import FancyArrowPatch

    n_axes = len(LAW_KEYS)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    # Sólo modelos simbólicos para no saturar
    selected = ["GPLearn", "PySR", "QLattice", "PySINDy"]

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True))
    ax.set_facecolor(STYLE["bg"])

    for model in selected:
        values = [np.mean(RECOVERY[model][lk]) for lk in LAW_KEYS]
        values += values[:1]
        ax.plot(angles, values, color=MODEL_COLORS[model],
                linewidth=1.8, label=model)
        ax.fill(angles, values, color=MODEL_COLORS[model], alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]),
                      labels=LAW_NAMES, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6.5,
                       color=STYLE["muted"])
    ax.grid(color=STYLE["border"], alpha=0.6)
    ax.spines["polar"].set_edgecolor(STYLE["border"])

    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12),
              fontsize=8.5, frameon=False)
    ax.set_title("Perfil de recuperación por ley — modelos simbólicos",
                 pad=20, fontsize=10, fontweight="500", color=STYLE["text"])

    fig.tight_layout()
    savefig(fig, "fig8_radar_recovery.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — ANÁLISIS DE ROBUSTEZ AL RUIDO (DEGRADACIÓN DE MSE)
# ═══════════════════════════════════════════════════════════════════════════════

def fig9_noise_robustness():
    """
    Para cada modelo, calcula cuánto sube el log-MSE al pasar
    de sin ruido → ruido alto. Muestra degradación media ± std entre leyes.
    """
    degradations = {}
    for model in MODELS:
        deltas = []
        for lk in LAW_KEYS:
            v_no   = MSE_BY_NOISE[model][lk][0]
            v_high = MSE_BY_NOISE[model][lk][2]
            if v_no > 0:
                delta = np.log10(v_high) - np.log10(v_no)
                deltas.append(delta)
        degradations[model] = (np.mean(deltas), np.std(deltas))

    models_sorted = sorted(MODELS,
                           key=lambda m: degradations[m][0])
    means = [degradations[m][0] for m in models_sorted]
    stds  = [degradations[m][1] for m in models_sorted]
    colors = [MODEL_COLORS[m] for m in models_sorted]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(range(len(models_sorted)), means,
                   color=colors, alpha=0.85, height=0.6, zorder=3)
    ax.errorbar(means, range(len(models_sorted)), xerr=stds,
                fmt="none", color=STYLE["text"], capsize=4,
                linewidth=1.2, zorder=4)

    ax.set_yticks(range(len(models_sorted)))
    ax.set_yticklabels(models_sorted, fontsize=9)
    ax.axvline(0, color=STYLE["border"], lw=0.8)
    ax.set_xlabel("Δ log₁₀(MSE) — de sin ruido a ruido alto  (menor = más robusto)",
                  fontsize=9)
    ax.grid(axis="x", alpha=0.4)

    for i, (m, val) in enumerate(zip(models_sorted, means)):
        tag = "±" + f"{stds[i]:.1f}"
        ax.text(val + 0.05, i, f"{val:+.1f}  {tag}",
                va="center", fontsize=7.5, color=STYLE["muted"])

    section_title(ax, "ROBUSTEZ AL RUIDO — DEGRADACIÓN DE MSE (sin ruido → ruido alto)")
    fig.tight_layout()
    savefig(fig, "fig9_noise_robustness.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(show: bool = False):
    print("\n── Generando figuras ──────────────────────────────────────────")
    figs = [
        ("1 — Tarjetas de métricas",           fig1_metric_cards),
        ("2 — Heatmap de recuperación",         fig2_recovery_heatmap),
        ("3 — Scatter recuperación vs. MSE",    fig3_scatter),
        ("4 — Ranking global",                  fig4_ranking),
        ("5 — Tabla de ecuaciones",             fig5_equations_table),
        ("6 — Heatmap numérico MSE",            fig6_mse_heatmap),
        ("7 — Evolución con nivel de ruido",    fig7_noise_sensitivity),
        ("8 — Radar de perfil por ley",         fig8_radar),
        ("9 — Robustez al ruido",               fig9_noise_robustness),
    ]

    for label, fn in figs:
        print(f"\n  Fig {label}")
        fn()

    print(f"\n✓ {len(figs)} figuras guardadas en ./{OUT_DIR}/")

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera todas las figuras del benchmark de regresión simbólica.")
    parser.add_argument("--show", action="store_true",
                        help="Además de guardar, abre las ventanas de matplotlib.")
    args = parser.parse_args()
    main(show=args.show)