import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "..", "results")

RESULTS_TXT = os.path.join(_RESULTS, "all_models", "combined_results.txt")
OOD_JSON = os.path.join(_RESULTS, "results_ood", "ood_metrics_summary.json")
OUT_DIR = os.path.join(_RESULTS, "all_models")

# ─────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "text.color": "#000000",
    "font.family": "serif",
    "mathtext.fontset": "cm",  # Renderiza matemáticas con tipografía tipo LM Roman
    "font.size": 14,           # Aumento general del tamaño de fuente
})

PERFECT_GREEN = "#1B5E20"
EXCELLENT_GREEN = "#66BB6A"
ORANGE = "#F57C00"
RED = "#C62828"

# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

MODEL_ORDER = [
    "GPLearn", "PySR", "QLattice", "PySINDy",
    "Polynomial", "MLP_Standard", "MLP_Sparse", "MLP_Dropout"
]

MODEL_ES = {
    "GPLearn": "GPLearn",
    "PySR": "PySR",
    "QLattice": "QLattice",
    "PySINDy": "PySINDy",
    "Polynomial": "Polinomial",
    "MLP_Standard": "Red Neuronal",
    "MLP_Sparse": "Red Neuronal Dispersa",
    "MLP_Dropout": "Red Neuronal Dropout"
}

MODEL_HEADER = {m: MODEL_ES.get(m, m) for m in MODEL_ORDER}

# ─────────────────────────────────────────────────────────────
# LEYES FÍSICAS EN ESPAÑOL
# ─────────────────────────────────────────────────────────────

LAW_ES = {
    "coulomb": "Ley de Coulomb",
    "harmonic_oscillator": "Oscilador armónico",
    "kepler_third": "Tercera ley de Kepler",
    "ideal_gas": "Ley de los gases ideales",
    "time_dilation": "Dilatación temporal",
    "projectile_range": "Rango proyectil",
    "radioactive_decay": "Decaimiento radiactivo",
    "newton_cooling": "Enfriamiento de Newton",
    "boltzmann_entropy": "Entropía de Boltzmann",
}

NOISE_KEYS = ["no_noise", "low_noise", "high_noise"]

# ─────────────────────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────────────────────

def _split_law_noise(dataset):
    for suffix in ("_no_noise", "_low_noise", "_high_noise"):
        if dataset.endswith(suffix):
            return dataset[:-len(suffix)], suffix[1:]
    return dataset, "unknown"

def parse_results(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    experiments = re.split(r"=+\r?\nEXPERIMENT:", text)[1:]
    rows = []

    for exp in experiments:
        blocks = re.findall(
            r"=== Model: (.*?) \| Dataset: (.*?) ===\r?\n\r?\n"
            r"Equation / Structure:\r?\n(.*?)\r?\n\r?\n"
            r"Metrics:\r?\nTest MSE: (.*?)\r?\nTest MAE: (.*?)\r?\n",
            exp, re.DOTALL,
        )

        for model, dataset, eq, mse, mae in blocks:
            law, noise = _split_law_noise(dataset.strip())

            rows.append({
                "model": model.strip(),
                "law": law,
                "noise": noise,
                "equation": eq.strip(),
                "mse": float(mse),
                "mae": float(mae),
            })

    return pd.DataFrame(rows)

def load_ood(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ood = {}
    for law, entries in raw.items():
        ood[law] = {}
        for key, metrics in entries.items():
            for ns in ("_no_noise", "_low_noise", "_high_noise"):
                if key.endswith(ns):
                    model = key[:-len(ns)]
                    noise = ns[1:]
                    mse = metrics.get("MSE")
                    ood[law].setdefault(model, {})[noise] = (
                        float(mse) if mse is not None else None
                    )
    return ood

# ─────────────────────────────────────────────────────────────
# CLEAN EQUATION & CELL FIT
# ─────────────────────────────────────────────────────────────

def compute_max_chars_per_col(fig_width, n_cols, base=200):
    return max(40, int(base / n_cols))

def format_latex_equation(eq: str, max_chars: int) -> str:
    if eq in ["", "--", "----", None]:
        return "--"

    eq = str(eq).replace("\n", " ")
    
    eq = eq.replace("**", "^")
    eq = eq.replace("omega", r"\Omega")
    eq = eq.replace("pi", r"\pi")
    eq = eq.replace("theta", r"\theta")
    
    eq = re.sub(r"([a-zA-Z])(\d+)", r"\1_{\2}", eq)
    eq = re.sub(r"sqrt\((.*?)\)", r"\\sqrt{\1}", eq)
    eq = re.sub(r"square\((.*?)\)", r"\1^2", eq)
    eq = re.sub(r"\s+", " ", eq).strip()

    if len(eq) > max_chars:
        eq = eq[:max_chars - 3].rstrip()
        
        eq = re.sub(r'\\[a-zA-Z]*$', '', eq)
        
        open_braces = eq.count('{') - eq.count('}')
        if open_braces > 0:
            eq += '}' * open_braces
            
        eq += "..."
    
    return f"${eq}$"
# ─────────────────────────────────────────────────────────────
# COLOR LOGIC
# ─────────────────────────────────────────────────────────────

def get_colour(mse):
    if mse is None or (isinstance(mse, float) and np.isnan(mse)):
        return RED

    if mse < 1e-20:
        return PERFECT_GREEN
    elif mse < 1e-4:
        return EXCELLENT_GREEN
    elif mse <= 1e-2:
        return ORANGE
    else:
        return RED

# ─────────────────────────────────────────────────────────────
# TABLE
# ─────────────────────────────────────────────────────────────

def build_table(df, ood, noise, laws, models, max_chars):
    text, colors = [], []

    for law in laws:
        row_t, row_c = [], []

        for model in models:
            sub = df[
                (df["model"] == model) &
                (df["law"] == law) &
                (df["noise"] == noise)
            ]

            is_nn = model.startswith("MLP")

            if is_nn:
                eq = "--"
            else:
                if sub.empty:
                    eq = "--"
                else:
                    eq = sub.iloc[0]["equation"]

            mse = ood.get(law, {}).get(model, {}).get(noise)

            eq_clean = format_latex_equation(eq, max_chars) if not is_nn else eq
            
            row_t.append(eq_clean)
            row_c.append(get_colour(mse))

        text.append(row_t)
        colors.append(row_c)

    return text, colors

# ─────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────

def render(cell_text, cell_colors, laws, models, out_path):
    headers = [MODEL_HEADER[m] for m in models]

    fig, ax = plt.subplots(figsize=(28, 12))
    ax.axis("off")

    table_text = [
        [LAW_ES.get(l, l)] + row for l, row in zip(laws, cell_text)
    ]
    table_colors = [["#F5F5F5"] + row for row in cell_colors]

    tbl = ax.table(
        cellText=table_text,
        colLabels=["Ley Física"] + headers,
        cellColours=table_colors,
        cellLoc="center",
        loc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)  # Letra más grande
    tbl.scale(1.0, 2.5)   # Celdas más altas para acomodar ecuaciones

    for (r, c), cell in tbl.get_celld().items():
        cell.get_text().set_wrap(True)
        cell.set_clip_on(True)
        cell.set_edgecolor('#D3D3D3') # Bordes más sutiles para estilo paper

        if r == 0:
            cell.set_facecolor("#E0E0E0")
            cell.set_text_props(weight="bold")

        if c == 0:
            cell.set_text_props(weight="bold")

    legend = [
        mpatches.Patch(color=PERFECT_GREEN,
                       label=r"Exacta ($\mathrm{MSE} < 10^{-20}$)"),
        mpatches.Patch(color=EXCELLENT_GREEN,
                       label=r"Excelente ($10^{-20} \leq \mathrm{MSE} < 10^{-4}$)"),
        mpatches.Patch(color=ORANGE,
                       label=r"Media ($10^{-4} \leq \mathrm{MSE} \leq 10^{-2}$)"),
        mpatches.Patch(color=RED,
                       label=r"Mala ($\mathrm{MSE} > 10^{-2}$)"),
    ]

    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02), # Reducido el espacio respecto a la tabla
        ncol=4,
        frameon=False, # Más elegante sin recuadro exterior
        fontsize=14
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight") # Subido a 300 DPI estándar
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(RESULTS_TXT) or not os.path.exists(OOD_JSON):
        sys.exit("ERROR: archivos no encontrados")

    df = parse_results(RESULTS_TXT)
    ood = load_ood(OOD_JSON)

    laws = list(df["law"].unique())
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    os.makedirs(OUT_DIR, exist_ok=True)
    
    n_cols = len(models) + 1
    max_chars = compute_max_chars_per_col(28, n_cols)

    for noise in NOISE_KEYS:
        text, colors = build_table(df, ood, noise, laws, models, max_chars)

        out = os.path.join(OUT_DIR, f"table_{noise}.png")
        render(text, colors, laws, models, out)

    print("Tablas generadas correctamente.")

if __name__ == "__main__":
    main()