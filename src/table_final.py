r"""
table_final.py - FIX FINAL ROBUSTO (SIN LATEX CRASH + WRAP + LEYENDA CIENTÍFICA)
"""

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
    "font.size": 11,
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

MODEL_HEADER = {m: m.replace("_", " ") for m in MODEL_ORDER}

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
# SAFE TEXT (SIN LATEX -> EVITA CRASHES)
# ─────────────────────────────────────────────────────────────

def clean_equation(eq: str) -> str:
    if eq in ["", "--", None]:
        return "--"

    eq = str(eq).replace("\n", " ")

    # simplificaciones seguras
    eq = eq.replace("**", "^")
    eq = eq.replace("omega", "Ω")

    # sqrt seguro básico
    eq = re.sub(r"sqrt\((.*?)\)", r"sqrt(\1)", eq)

    # limpiar basura
    eq = re.sub(r"\s+", " ", eq)

    return eq.strip()


def truncate_text(eq: str, maxlen=45) -> str:
    if eq == "--":
        return eq

    if len(eq) <= maxlen:
        return eq

    return eq[:maxlen] + "..."


# ─────────────────────────────────────────────────────────────
# COLOR LOGIC
# ─────────────────────────────────────────────────────────────

def get_colour(mse, mae):

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

def build_table(df, ood, noise, laws, models):

    text, colors = [], []

    for law in laws:
        row_t, row_c = [], []

        for model in models:
            sub = df[
                (df["model"] == model) &
                (df["law"] == law) &
                (df["noise"] == noise)
            ]

            if sub.empty:
                eq = "--"
                mae = None
            else:
                eq = sub.iloc[0]["equation"]
                mae = sub.iloc[0]["mae"]

            mse = ood.get(law, {}).get(model, {}).get(noise)

            eq_clean = clean_equation(eq)
            eq_final = truncate_text(eq_clean)

            row_t.append(eq_final)
            row_c.append(get_colour(mse, mae))

        text.append(row_t)
        colors.append(row_c)

    return text, colors


# ─────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────

def render(cell_text, cell_colors, laws, models, law_names, out_path):

    headers = [MODEL_HEADER[m] for m in models]

    fig, ax = plt.subplots(figsize=(26, 10))
    ax.axis("off")

    table_text = [[law_names.get(l, l)] + row for l, row in zip(laws, cell_text)]
    table_colors = [["#F5F5F5"] + row for row in cell_colors]

    tbl = ax.table(
        cellText=table_text,
        colLabels=["Ley Física"] + headers,
        cellColours=table_colors,
        cellLoc="center",
        loc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 2.2)

    # 🔥 FIX: wrap + clip + ajuste
    for (r, c), cell in tbl.get_celld().items():
        cell.get_text().set_wrap(True)
        cell.set_clip_on(True)

        if r == 0:
            cell.set_facecolor("#E0E0E0")
            cell.set_text_props(weight="bold")

        if c == 0:
            cell.set_text_props(weight="bold")

    # ─────────────────────────────────────────────
    # LEGEND (MEJORADA)
    # ─────────────────────────────────────────────

    legend = [
        mpatches.Patch(color=PERFECT_GREEN,
                       label=r"Perfecta ($\mathrm{MSE} < 10^{-20}$)"),
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
        bbox_to_anchor=(0.5, 0.15),
        ncol=2,
        frameon=True,
        fontsize=14
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
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

    law_names = {l: l for l in laws}

    os.makedirs(OUT_DIR, exist_ok=True)

    for noise in NOISE_KEYS:
        text, colors = build_table(df, ood, noise, laws, models)

        out = os.path.join(OUT_DIR, f"table_{noise}.png")
        render(text, colors, laws, models, law_names, out)

    print("✓ Tablas generadas correctamente")


if __name__ == "__main__":
    main()