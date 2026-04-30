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
OUT_DIR = os.path.join(_RESULTS, "results_ood")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from utils.utils import set_plot_style
    set_plot_style(for_paper=True)
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────
# STYLE & TRUE EQUATIONS
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
})

COLOR_EXACT = '#2ca02c'
COLOR_BUENO = 'limegreen'
COLOR_APPROX = '#ff7f0e'
COLOR_INCORRECT = '#d62728'

def get_true_eq(raw_law):
    mapping = {
        "coulomb": r"$q_1 q_2/r^2$",
        "oscillator": r"$-x$",
        "harmonic_oscillator": r"$-x$",
        "kepler": r"$r^{3/2}$",
        "kepler_third": r"$a^{3/2}$",
        "ideal_gas": r"$n T/V$",
        "time_dilation": r"$t/\sqrt{1 - v^2}$",
        "projectile_range": r"$v_0^2 \sin(2\theta)$",
        "radioactive_decay": r"$e^{-\lambda t}$",
        "newton_cooling": r"$1 + e^{-kt}$",
        "boltzmann_entropy": r"$\ln(\Omega)$",
    }
    return mapping.get(raw_law, "--")

def get_law_name(raw_law):
    mapping = {
        "coulomb": "Ley de Coulomb",
        "oscillator": "Oscilador armónico",
        "harmonic_oscillator": "Oscilador armónico",
        "kepler": "Tercera ley de Kepler",
        "kepler_third": "Tercera ley de Kepler",
        "ideal_gas": "Ley de los gases ideales",
        "time_dilation": "Dilatación temporal",
        "projectile_range": "Rango proyectil",
        "radioactive_decay": "Decaimiento radiactivo",
        "newton_cooling": "Enfriamiento de Newton",
        "boltzmann_entropy": "Entropía de Boltzmann",
    }
    return mapping.get(raw_law, raw_law)

LAW_ORDER = [
    "coulomb",
    "oscillator", 
    "kepler",
    "ideal_gas",
    "time_dilation",
    "projectile_range",
    "radioactive_decay",
    "newton_cooling",
    "boltzmann_entropy"
]

# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

MODEL_ORDER = [
 "MLP_Standard", "MLP_Sparse", "MLP_Dropout", "Polynomial",  "PySINDy" , "GPLearn", "PySR", "QLattice",
]

MODEL_ES = {
    "GPLearn": "GPLearn",
    "PySR": "PySR",
    "QLattice": "QLattice",
    "PySINDy": "PySINDy",
    "Polynomial": "Polinomial",
    "MLP_Standard": "Estándar",
    "MLP_Sparse": "Dispersa",
    "MLP_Dropout": "Dropout"
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

def get_max_chars(model: str) -> int:
    # Aumentamos el límite para los modelos priorizados
    if model in ["PySR"]:
        return 40
    if model in ["PySINDy", "QLattice", "Polynomial"]:
        return 35
    if model.startswith("MLP"):
        return 10
    return 35

def format_latex_equation(eq: str, max_chars: int) -> str:
    if eq in ["", "--", "----", None]:
        return "--"

    eq = str(eq).replace("\n", " ")

    def repl_sci(m):
        base = float(m.group(1))
        exp = int(m.group(2))
        if exp == 0:
            return f"{int(base)}" if base.is_integer() else f"{base}"
        if base == 1.0:
            return f"10^{{{exp}}}"
        return f"{base} \\times 10^{{{exp}}}"
        
    eq = re.sub(r"\b([0-9.]+)e([-+]?[0-9]+)\b", repl_sci, eq)
    

    for func in ["log", "exp", "sin", "cos", "tan"]:
        eq = re.sub(rf"\b{func}\b", rf"\\{func}", eq)
    
    eq = eq.replace("omega", r"\Omega")
    eq = eq.replace("pi", r"\pi")
    eq = eq.replace("theta", r"\theta")
    eq = eq.replace("lambd", r"\lambda")
    
    eq = re.sub(r"sqrt\((.*?)\)", r"\\sqrt{\1}", eq)
    eq = re.sub(r"square\((.*?)\)", r"{\1}^2", eq)
    
    eq = eq.replace("**", "^")
    eq = re.sub(r"\^\((.*?)\)", r"^{\1}", eq)
    eq = re.sub(r"\^([a-zA-Z0-9.\-]+)", r"^{\1}", eq)
    eq = re.sub(r"([a-zA-Z])(\d+)", r"\1_{\2}", eq)

    eq = re.sub(r"\b1(?:\.0+)?\s*\*\s*", "", eq)
    eq = eq.replace("*", " ")
    eq = re.sub(r"\s+", " ", eq).strip()

    m = re.match(r"^([-+]?\s*\d+(?:\.\d+)?(?: \\times 10\^\{-?\d+\})?)\s*([-+])\s*(.*)$", eq)
    if m:
        c, sign, rest = m.groups()
        if rest:
            if not c.strip().startswith("-"):
                c = "+ " + c.strip()
            else:
                c = c.strip()
                c = c.replace("-", "- ") if not c.startswith("- ") else c
            
            if sign == "+":
                eq = f"{rest} {c}"
            elif sign == "-":
                eq = f"- {rest} {c}"

    eq = eq.replace("+ -", "- ")
    eq = eq.replace("- -", "+ ")

    if len(eq) > max_chars:
        eq = eq[:max_chars - 4].rstrip()
        eq = re.sub(r'\\[a-zA-Z]*$', '', eq).rstrip()
        eq = re.sub(r'[\^\_\+\-\/\=]+$', '', eq).rstrip()
        
        open_braces = eq.count('{') - eq.count('}')
        if open_braces > 0:
            eq += '}' * open_braces
            
        eq += r" \dots"
    
    eq = re.sub(r"(?<=\d|\})\s+1(?=\s*[\+\-]|$)", "", eq) 
    return f"${eq}$"

# ─────────────────────────────────────────────────────────────
# COLOR LOGIC
# ─────────────────────────────────────────────────────────────

def get_colour(mse):
    if mse is None or (isinstance(mse, float) and np.isnan(mse)):
        return COLOR_INCORRECT

    if mse < 1e-20:
        return COLOR_EXACT
    elif mse < 1e-4:
        return COLOR_BUENO
    elif mse <= 1e-2:
        return COLOR_APPROX
    else:
        return COLOR_INCORRECT

# ─────────────────────────────────────────────────────────────
# TABLE
# ─────────────────────────────────────────────────────────────

def build_table_data(df, ood, noise, laws, models):
    text, colors = [], []

    # Super-cabecera definida a trozos
    row0_t = [""] * (len(models) + 2)
    row0_c = ["#FFFFFF", "#FFFFFF"]  # Ley, Expresión
    for m in models:
        if m in ["GPLearn", "PySR", "QLattice", "PySINDy"]:
            row0_c.append("#BDBDBD")
        elif m == "Polynomial":
            row0_c.append("#FFFFFF")
        elif m.startswith("MLP"):
            row0_c.append("#BDBDBD")
            
    text.append(row0_t)
    colors.append(row0_c)

    row1_t = ["Ley Física", "Forma Analítica"] + [MODEL_ES.get(m, m) for m in models]
    row1_c = ["#D6D6D6"] * (len(models) + 2)
    text.append(row1_t)
    colors.append(row1_c)

    for law in laws:
        row_t = [get_law_name(law), get_true_eq(law)]
        row_c = ["#F5F5F5", "#EAEAEA"]

        for model in models:
            sub = df[
                (df["model"] == model) &
                (df["law"] == law) &
                (df["noise"] == noise)
            ]

            is_nn = model.startswith("MLP")
            eq = "--" if is_nn or sub.empty else sub.iloc[0]["equation"]

            mse = ood.get(law, {}).get(model, {}).get(noise)
            eq_clean = format_latex_equation(eq, get_max_chars(model)) if not is_nn else eq
            
            row_t.append(eq_clean)
            row_c.append(get_colour(mse))

        text.append(row_t)
        colors.append(row_c)

    return text, colors

# ─────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────

def render(cell_text, cell_colors, models, out_path):
    # Altura del subplot reducida para evitar espacio inútil
    fig, ax = plt.subplots(figsize=(28, 6))
    ax.axis("off")

    widths = [0.11, 0.07] 
    for m in models:
        if m.startswith("MLP"):
            widths.append(0.04)
        elif m in ["QLattice"]:
            widths.append(0.14)
        elif m in ["PySR"]:
            widths.append(0.13)   # Incremento de área
        else:
            widths.append(0.12)
            
    w_sum = sum(widths)
    widths = [w / w_sum for w in widths]

    tbl = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colWidths=widths,
        cellLoc="center",
        loc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15) 
    tbl.scale(1.0, 2.2)

    for (r, c), cell in tbl.get_celld().items():
        cell.get_text().set_wrap(True)
        cell.set_clip_on(True)
        cell.set_edgecolor('#D3D3D3')

        if r == 0:
            if c in [2, 3, 4] or c in [6, 7, 8, 9]:
                cell.set_edgecolor("#BDBDBD")
                cell.set_facecolor("#BDBDBD")
            else:
                cell.set_edgecolor("#FFFFFF")
                cell.set_facecolor("#FFFFFF")
                
        elif r == 1:
            cell.set_text_props(weight="bold")

        if c == 0 and r > 1:
            cell.set_text_props(weight="bold")

    fig.canvas.draw()
    
    b_rs_L = tbl[0, 6].get_bbox()
    b_rs_R = tbl[0, 9].get_bbox()
    ax.text((b_rs_L.x0 + b_rs_R.x1) / 2, (b_rs_L.y0 + b_rs_L.y1) / 2, 
            "Regresores Simbólicos", ha='center', va='center', weight='bold', size=16, transform=ax.transAxes)

    b_nn_L = tbl[0, 2].get_bbox()
    b_nn_R = tbl[0, 4].get_bbox()
    ax.text((b_nn_L.x0 + b_nn_R.x1) / 2, (b_nn_L.y0 + b_nn_L.y1) / 2, 
            "Redes Neuronales", ha='center', va='center', weight='bold', size=16, transform=ax.transAxes)

    legend = [
        mpatches.Patch(color=COLOR_EXACT, label=r"Exacta ($\mathrm{MSE} < 10^{-20}$)"),
        mpatches.Patch(color=COLOR_BUENO, label=r"Precisa ($10^{-20} \leq \mathrm{MSE} < 10^{-4}$)"),
        mpatches.Patch(color=COLOR_APPROX, label=r"Aproximada ($10^{-4} \leq \mathrm{MSE} \leq 10^{-2}$)"),
        mpatches.Patch(color=COLOR_INCORRECT, label=r"Incorrecta ($\mathrm{MSE} > 10^{-2}$)"),
    ]

    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),  # Leyenda acortada al borde inferior de la tabla
        ncol=4,
        frameon=False,
        fontsize=20
    )

    # Padding nulo para recortar bordes blancos
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(RESULTS_TXT) or not os.path.exists(OOD_JSON):
        sys.exit("ERROR: archivos no encontrados")

    df = parse_results(RESULTS_TXT)
    ood = load_ood(OOD_JSON)

    laws = [law for law in LAW_ORDER if law in df["law"].values]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    os.makedirs(OUT_DIR, exist_ok=True)

    for noise in NOISE_KEYS:
        text, colors = build_table_data(df, ood, noise, laws, models)
        out = os.path.join(OUT_DIR, f"table_{noise}.pdf")
        render(text, colors, models, out)

    print("Tablas generadas correctamente (PDF).")

if __name__ == "__main__":
    main()