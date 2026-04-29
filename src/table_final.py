r"""
table_final.py
Versión definitiva:
- Colores correctos (incluye MAE==0)
- Sin celdas blancas
- Omega → \Omega
- Leyenda completa con rangos MSE
- Leyenda cerca y con borde
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── RUTAS ────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_RESULTS    = os.path.join(_HERE, "..", "results")
RESULTS_TXT = os.path.join(_RESULTS, "all_models",   "combined_results.txt")
OOD_JSON    = os.path.join(_RESULTS, "results_ood",  "ood_metrics_summary.json")
OUT_DIR     = os.path.join(_RESULTS, "all_models")

# ── CONFIG ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor":   "#FFFFFF",
    "text.color":       "#000000",
    "font.family":      "serif",
    "font.size":        12,
    "mathtext.fontset": "cm",
})

# ── COLORES ──────────────────────────────────────────────────────────────────
PERFECT_GREEN   = "#1B5E20"
EXCELLENT_GREEN = "#66BB6A"
ORANGE          = "#F57C00"
RED             = "#C62828"
NULL_COLOUR     = "#EEEEEE"

# ── NOMBRES ──────────────────────────────────────────────────────────────────
LAW_ORDER = [
    "boltzmann_entropy", "coulomb", "ideal_gas", "kepler",
    "newton_cooling", "oscillator", "projectile_range",
    "radioactive_decay", "time_dilation",
]

LAW_DISPLAY = {
    "boltzmann_entropy": "Entropía de Boltzmann",
    "coulomb": "Ley de Coulomb",
    "ideal_gas": "Gas ideal",
    "kepler": "Tercera ley de Kepler",
    "newton_cooling": "Enfriamiento de Newton",
    "oscillator": "Oscilador armónico",
    "projectile_range": "Alcance de proyectil",
    "radioactive_decay": "Desintegración radiactiva",
    "time_dilation": "Dilatación temporal",
}

MODEL_ORDER = ["GPLearn", "PySR", "QLattice", "PySINDy", "Polynomial",
               "MLP_Standard", "MLP_Sparse", "MLP_Dropout"]

MODEL_HEADER = {m: m.replace("_", " ") for m in MODEL_ORDER}

NOISE_KEYS = ["no_noise", "low_noise", "high_noise"]

# ══════════════════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════════════════

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

        for model, dataset, equation, mse, mae in blocks:
            law, noise = _split_law_noise(dataset.strip())
            rows.append({
                "model": model.strip(),
                "law": law,
                "noise": noise,
                "equation": equation.strip(),
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
                    ood[law].setdefault(model, {})[noise] = float(mse) if mse else None
    return ood

# ══════════════════════════════════════════════════════════════════════════════
# LATEX
# ══════════════════════════════════════════════════════════════════════════════

def format_equation_latex(eq_str, is_mlp=False):
    if is_mlp or eq_str in ["—", "", "None", "--"]:
        return "--"

    eq = str(eq_str).replace("\n", " ")
    eq = eq.replace("omega", r"\Omega ")
    eq = eq.replace("**2", "^2").replace("**3", "^3")
    eq = re.sub(r"sqrt\(([^()]*)\)", r"\\sqrt{\1}", eq)

    return f"${eq}$"

def truncate_latex_eq(eq_latex, maxlen=45):
    if eq_latex == "--":
        return eq_latex

    inner = eq_latex.strip("$")
    if len(inner) <= maxlen:
        return eq_latex

    cut = inner[:maxlen]

    open_b = cut.count("{")
    close_b = cut.count("}")
    if open_b > close_b:
        cut += "}" * (open_b - close_b)

    return f"${cut} \\dots$"

# ══════════════════════════════════════════════════════════════════════════════
# COLOR LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def get_colour(mse, mae):
    # CASO 2: divergencia (Null / NaN / None)
    if mse is None or (isinstance(mse, float) and np.isnan(mse)):
        return RED  # ahora rojo = mala ecuación
    # CASO 1: ecuación perfecta
    if mae is not None and mae == 0:
        return PERFECT_GREEN


    # CASO 3: muy buena
    if mse < 1e-4:
        return EXCELLENT_GREEN

    # CASO 4: media
    elif mse <= 1e-2:
        return ORANGE

    # CASO 5: mala
    else:
        return RED
# ══════════════════════════════════════════════════════════════════════════════
# TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_table_data(df, ood, noise_key, laws, models):
    cell_text, cell_colors = [], []

    for lk in laws:
        row_t, row_c = [], []

        for model in models:
            sub = df[(df["model"] == model) &
                     (df["law"] == lk) &
                     (df["noise"] == noise_key)]

            if sub.empty:
                eq = "--"
                mae = None
            else:
                eq = sub.iloc[0]["equation"]
                mae = sub.iloc[0]["mae"]

            mse = ood.get(lk, {}).get(model, {}).get(noise_key)

            eq_final = truncate_latex_eq(format_equation_latex(eq, "MLP" in model))

            row_t.append(eq_final)
            row_c.append(get_colour(mse, mae))

        cell_text.append(row_t)
        cell_colors.append(row_c)

    return cell_text, cell_colors

# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render_png(cell_text, cell_colors, laws, models, law_names, out_path):
    headers = [MODEL_HEADER[m] for m in models]

    fig, ax = plt.subplots(figsize=(24, 10))
    ax.axis("off")

    tbl_text = [[law_names[l]] + cell_text[i] for i, l in enumerate(laws)]
    tbl_colors = [["#F5F5F5"] + cell_colors[i] for i in range(len(laws))]

    tbl = ax.table(
        cellText=tbl_text,
        colLabels=["Ley Física"] + headers,
        cellColours=tbl_colors,
        cellLoc="center",
        loc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.5)

    # CABECERAS
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E0E0E0")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_text_props(weight="bold")

        # texto blanco SOLO rojo oscuro
        if cell.get_facecolor() == plt.matplotlib.colors.to_rgba(RED):
            cell.set_text_props(color="white")

    # LEYENDA
    legend_items = [
        mpatches.Patch(color=PERFECT_GREEN, label="Perfecta (MAE = 0)"),
        mpatches.Patch(color=EXCELLENT_GREEN, label="Excelente (MSE < 1e-4)"),
        mpatches.Patch(color=ORANGE, label="Media (1e-4 ≤ MSE ≤ 1e-2)"),
        mpatches.Patch(color=RED, label="Mala (MSE > 1e-2)"),
    ]

    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.12),  # más cerca
        ncol=4,
        frameon=True
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not os.path.exists(RESULTS_TXT) or not os.path.exists(OOD_JSON):
        sys.exit("ERROR: archivos no encontrados")

    df = parse_results(RESULTS_TXT)
    ood = load_ood(OOD_JSON)

    laws = [l for l in LAW_ORDER if l in df["law"].unique()]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    law_names = {lk: LAW_DISPLAY.get(lk, lk) for lk in laws}

    os.makedirs(OUT_DIR, exist_ok=True)

    for noise in NOISE_KEYS:
        cell_text, cell_colors = build_table_data(df, ood, noise, laws, models)

        out = os.path.join(OUT_DIR, f"table_{noise}.png")
        render_png(cell_text, cell_colors, laws, models, law_names, out)

    print("✓ Tablas generadas correctamente")

if __name__ == "__main__":
    main()