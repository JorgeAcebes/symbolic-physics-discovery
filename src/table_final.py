"""
table_final.py
==============
Genera tres tablas de ecuaciones descubiertas (sin ruido / ruido bajo /
ruido alto), donde el color de cada celda refleja el MAE de evaluación
fuera del dominio (OOD) del modelo correspondiente.

Ubicación esperada en el repositorio:
    symbolic-physics-discovery/src/table_final.py

Entradas (rutas relativas al directorio del script):
    ../results/all_models/combined_results.txt
    ../results/results_ood/ood_metrics_summary.json

Salidas (en ../results/all_models/):
    table_no_noise.png / .tex
    table_low_noise.png / .tex
    table_high_noise.png / .tex

Escala de colores (OOD MAE, escala log):
    ■ Verde intenso   MAE = 0  (recuperación simbólica exacta)
    ■ Verde           MAE < 0.01
    ■ Verde claro     MAE < 0.1
    ■ Amarillo        MAE < 1
    ■ Ámbar           MAE < 10
    ■ Naranja         MAE < 100
    ■ Rojo            MAE ≥ 100
    ■ Gris            sin dato OOD (NULL)

Dependencias: numpy, pandas, matplotlib
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

# ── rutas ────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_RESULTS    = os.path.join(_HERE, "..", "results")
RESULTS_TXT = os.path.join(_RESULTS, "all_models",   "combined_results.txt")
OOD_JSON    = os.path.join(_RESULTS, "results_ood",  "ood_metrics_summary.json")
OUT_DIR     = os.path.join(_RESULTS, "all_models")

# ── orden y nombres legibles ─────────────────────────────────────────────────
LAW_ORDER = [
    "boltzmann_entropy", "coulomb", "ideal_gas", "kepler",
    "newton_cooling", "oscillator", "projectile_range",
    "radioactive_decay", "time_dilation",
]
LAW_DISPLAY = {
    "boltzmann_entropy": "Boltzmann entropy",
    "coulomb":           "Coulomb",
    "ideal_gas":         "Ideal gas",
    "kepler":            "Kepler",
    "newton_cooling":    "Newton cooling",
    "oscillator":        "Oscillator",
    "projectile_range":  "Projectile range",
    "radioactive_decay": "Radioactive decay",
    "time_dilation":     "Time dilation",
}

# Modelos mostrados en las tablas y su cabecera corta para LaTeX
MODEL_ORDER = ["GPLearn", "PySR", "QLattice", "PySINDy", "Polynomial",
               "MLP_Standard", "MLP_Sparse", "MLP_Dropout"]
MODEL_HEADER = {
    "GPLearn":      "GPLearn",
    "PySR":         "PySR",
    "QLattice":     "QLattice",
    "PySINDy":      "PySINDy",
    "Polynomial":   "Polynomial",
    "MLP_Standard": "MLP Std.",
    "MLP_Sparse":   "MLP Sparse",
    "MLP_Dropout":  "MLP Drop.",
}

NOISE_KEYS   = ["no_noise", "low_noise", "high_noise"]
NOISE_LABELS = {
    "no_noise":   "sin ruido",
    "low_noise":  "ruido bajo",
    "high_noise": "ruido alto",
}
NOISE_LABEL_EN = {
    "no_noise":   "No noise",
    "low_noise":  "Low noise",
    "high_noise": "High noise",
}

# ── escala de colores OOD MAE ────────────────────────────────────────────────
# Cada entrada: (umbral_superior, hex_fondo, hex_latex, etiqueta)
# La celda recibe el color del primer tramo cuyo umbral sea > MAE.
COLOUR_SCALE = [
    (0.0,    "#1A6B0A", "ood0",   "MAE = 0 (exacto)"),   # verde intenso
    (1e-2,   "#4CAF50", "ood1",   "MAE < 0.01"),           # verde
    (1e-1,   "#A5D6A7", "ood2",   "MAE < 0.1"),            # verde claro
    (1e0,    "#FFF176", "ood3",   "MAE < 1"),               # amarillo
    (1e1,    "#FFB74D", "ood4",   "MAE < 10"),              # ámbar
    (1e2,    "#EF6C00", "ood5",   "MAE < 100"),             # naranja
    (np.inf, "#C62828", "ood6",   "MAE ≥ 100"),             # rojo
]
NULL_COLOUR   = "#BDBDBD"   # gris — sin dato OOD
NULL_TEX_NAME = "oodnull"

# Colores de texto (blanco sobre fondos oscuros, negro sobre claros)
_DARK_BKGS = {"ood0", "ood6"}   # verde intenso y rojo → texto blanco

STYLE = {
    "bg":     "#FAFAF8",
    "header": "#ECEBE5",
    "border": "#CCCCCC",
    "text":   "#1C1C1A",
    "muted":  "#888880",
}

plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["bg"],
    "text.color":       STYLE["text"],
    "font.family":      "DejaVu Sans",
    "font.size":        9,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. PARSER  combined_results.txt
# ══════════════════════════════════════════════════════════════════════════════

def _split_law_noise(dataset: str):
    """'boltzmann_entropy_no_noise' → ('boltzmann_entropy', 'no_noise')"""
    for suffix in ("_no_noise", "_low_noise", "_high_noise"):
        if dataset.endswith(suffix):
            return dataset[: -len(suffix)], suffix[1:]
    parts = dataset.rsplit("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (dataset, "unknown")


def parse_results(path: str) -> pd.DataFrame:
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
                "model":    model.strip(),
                "law":      law,
                "noise":    noise,
                "equation": equation.strip(),
                "mse":      float(mse),
                "mae":      float(mae),
            })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA OOD JSON  →  dict[law][model][noise] = MAE | None
# ══════════════════════════════════════════════════════════════════════════════

def load_ood(path: str) -> dict:
    """
    Transforma el JSON de estructura plana
        {law: {"Model_noise": {"MAE": v, "MSE": v}, ...}}
    a un dict jerárquico
        {law: {model: {noise: mae_value_or_None}}}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ood: dict = {}
    for law, entries in raw.items():
        ood[law] = {}
        for composite_key, metrics in entries.items():
            # composite_key = "ModelName_noise_level"  e.g. "GPLearn_high_noise"
            # Noise suffix is always one of the three known values
            model = noise = None
            for ns in ("_no_noise", "_low_noise", "_high_noise"):
                if composite_key.endswith(ns):
                    model = composite_key[: -len(ns)]
                    noise = ns[1:]
                    break
            if model is None:
                continue
            mae_val = metrics.get("MAE")
            ood[law].setdefault(model, {})[noise] = (
                float(mae_val) if mae_val is not None else None
            )
    return ood

# ══════════════════════════════════════════════════════════════════════════════
# 3. COLORES
# ══════════════════════════════════════════════════════════════════════════════

def mae_to_colour(mae) -> tuple[str, str, str]:
    """
    Devuelve (hex_fondo, tex_name, text_hex) para un valor de MAE OOD.
    mae puede ser None (dato ausente) o float.
    """
    if mae is None:
        return NULL_COLOUR, NULL_TEX_NAME, STYLE["text"]
    for threshold, hex_bg, tex_name, _ in COLOUR_SCALE:
        if mae <= threshold:
            text_col = "#FFFFFF" if tex_name in _DARK_BKGS else STYLE["text"]
            return hex_bg, tex_name, text_col
    # fallback: rojo
    return COLOUR_SCALE[-1][1], COLOUR_SCALE[-1][2], "#FFFFFF"


def _hex_to_rgb01(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRUNCADO DE ECUACIONES
# ══════════════════════════════════════════════════════════════════════════════

def _shorten(eq: str, maxlen: int = 46) -> str:
    """Acorta exponentes y trunca si sigue siendo muy larga."""
    eq = (eq.replace("**2", "²").replace("**3", "³")
            .replace("**4", "⁴").replace("**5", "⁵"))
    return eq if len(eq) <= maxlen else eq[: maxlen - 1] + "…"


def _latex_eq(eq: str, maxlen: int = 52) -> str:
    """
    Convierte la ecuación a una cadena apta para LaTeX:
      - escapa caracteres especiales de LaTeX
      - envuelve en \\texttt{}
      - trunca si hace falta
    """
    # No intentamos renderizar como math porque las ecuaciones son strings
    # en notación Python; se presentan en fuente monoespaciada.
    s = eq
    # Sustituir caracteres que romperían LaTeX
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("_",  "\\_")
    s = s.replace("^",  "\\^{}")
    s = s.replace("**", "\\^{}")   # después de escapar \
    s = s.replace("%",  "\\%")
    s = s.replace("$",  "\\$")
    s = s.replace("&",  "\\&")
    s = s.replace("#",  "\\#")
    s = s.replace("{",  "\\{")
    s = s.replace("}",  "\\}")
    s = s.replace("~",  "\\textasciitilde{}")
    # Truncar
    if len(s) > maxlen:
        s = s[: maxlen - 1] + "\\ldots"
    return f"\\texttt{{{s}}}"

# ══════════════════════════════════════════════════════════════════════════════
# 5. CONSTRUCCIÓN DE LAS MATRICES DE TABLA
# ══════════════════════════════════════════════════════════════════════════════

def build_table(df: pd.DataFrame, ood: dict,
                noise_key: str, laws: list, models: list):
    """
    Devuelve dos matrices paralelas (n_laws × n_models):
        cell_text   — ecuación (string corto para PNG)
        cell_meta   — (hex_bg, tex_name, text_hex, mae_value, full_equation)
    """
    cell_text = []
    cell_meta = []

    for lk in laws:
        row_text = []
        row_meta = []
        for model in models:
            sub = df[(df["model"] == model) &
                     (df["law"]   == lk)    &
                     (df["noise"] == noise_key)]
            if sub.empty:
                eq_full  = "—"
                eq_short = "—"
            else:
                eq_full  = sub.iloc[0]["equation"]
                eq_short = _shorten(eq_full)

            # OOD MAE — intentar con el nombre exacto y variantes
            mae = None
            law_ood = ood.get(lk, ood.get(lk.replace("_", ""), {}))
            if lk in ood:
                mae = ood[lk].get(model, {}).get(noise_key)

            hex_bg, tex_name, text_col = mae_to_colour(mae)
            row_text.append(eq_short)
            row_meta.append((hex_bg, tex_name, text_col, mae, eq_full))

        cell_text.append(row_text)
        cell_meta.append(row_meta)

    return cell_text, cell_meta

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURA PNG
# ══════════════════════════════════════════════════════════════════════════════

def _legend_patches():
    patches = []
    for _, hex_bg, _, label in COLOUR_SCALE:
        patches.append(mpatches.Patch(facecolor=hex_bg, edgecolor="#888",
                                      linewidth=0.4, label=label))
    patches.append(mpatches.Patch(facecolor=NULL_COLOUR, edgecolor="#888",
                                  linewidth=0.4, label="Sin dato OOD"))
    return patches


def render_png(cell_text, cell_meta, laws, models, noise_key,
               law_names, out_path: str):
    n_laws   = len(laws)
    n_models = len(models)
    headers  = [MODEL_HEADER.get(m, m) for m in models]

    # Tamaño de figura adaptado al contenido
    fig_w = max(22, n_models * 2.7)
    fig_h = max(5.5, n_laws * 0.72 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Filas: cabecera + n_laws
    n_rows_tbl = n_laws + 1
    n_cols_tbl = n_models + 1   # columna extra para el nombre de la ley

    col_w_law    = 0.12
    col_w_model  = (1.0 - col_w_law) / n_models
    col_widths   = [col_w_law] + [col_w_model] * n_models

    # Construir datos para ax.table
    tbl_text   = [[law_names[lk]] + cell_text[i]
                  for i, lk in enumerate(laws)]
    tbl_colors = [[STYLE["header"]] + [meta[0] for meta in cell_meta[i]]
                  for i, lk in enumerate(laws)]

    tbl = ax.table(
        cellText=tbl_text,
        colLabels=["Ley"] + headers,
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)

    # Estilo de cabecera
    for j in range(n_cols_tbl):
        c = tbl[0, j]
        c.set_facecolor(STYLE["header"])
        c.set_edgecolor(STYLE["border"])
        c.set_text_props(fontweight="bold", color=STYLE["muted"])
        c.set_height(0.09)

    # Estilo de celdas de contenido
    for i in range(n_laws):
        for j in range(n_cols_tbl):
            c = tbl[i + 1, j]
            c.set_edgecolor(STYLE["border"])
            c.set_height(0.072)
            c.PAD = 0.04
            if j == 0:
                c.set_facecolor(STYLE["header"])
                c.set_text_props(fontweight="500", color=STYLE["text"])
            else:
                hex_bg, _, text_col, mae, _ = cell_meta[i][j - 1]
                c.set_facecolor(hex_bg)
                c.set_text_props(color=text_col)

    # Leyenda
    fig.legend(handles=_legend_patches(),
               title="OOD MAE", title_fontsize=7.5,
               loc="lower center", ncol=len(COLOUR_SCALE) + 1,
               fontsize=7, frameon=True,
               framealpha=0.9, edgecolor=STYLE["border"],
               bbox_to_anchor=(0.5, -0.01))

    label = NOISE_LABELS[noise_key]
    ax.set_title(
        f"Ecuaciones descubiertas — {label}  "
        f"(color = OOD MAE)",
        loc="left", fontsize=9.5, fontweight="500",
        color=STYLE["muted"], pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"], edgecolor="none")
    plt.close(fig)
    print(f"    → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FICHERO LaTeX
# ══════════════════════════════════════════════════════════════════════════════

# Definiciones de colores para el preámbulo LaTeX
_LATEX_COLOUR_DEFS = "\n".join(
    f"\\definecolor{{{tex_name}}}{{HTML}}{{{hex_bg.lstrip('#')}}}"
    for _, hex_bg, tex_name, _ in COLOUR_SCALE
) + f"\n\\definecolor{{{NULL_TEX_NAME}}}{{HTML}}{{{NULL_COLOUR.lstrip('#')}}}"

_LATEX_PREAMBLE_COMMENT = """\
% ─────────────────────────────────────────────────────────────────────────────
% Tabla generada automáticamente por src/table_final.py
%
% Paquetes necesarios en el preámbulo del documento:
%   \\usepackage{booktabs}
%   \\usepackage{colortbl}
%   \\usepackage{xcolor}
%   \\usepackage{adjustbox}   % para \\begin{adjustbox}{max width=\\linewidth}
%   \\usepackage{array}
%
% Añade también en el preámbulo las siguientes definiciones de color:
%
"""

def _latex_colour_comment(indent="% "):
    return "\n".join(indent + line
                     for line in _LATEX_COLOUR_DEFS.split("\n"))


def render_latex(cell_text, cell_meta, laws, models, noise_key,
                 law_names, out_path: str):
    """
    Genera un fichero .tex con una tabla booktabs + colortbl lista para
    incluir en cualquier documento LaTeX con \\input{...} o \\include{...}.
    """
    headers = [MODEL_HEADER.get(m, m) for m in models]
    label   = NOISE_LABEL_EN[noise_key]
    n_laws  = len(laws)
    n_models = len(models)

    # Especificación de columnas: p{} para texto largo
    law_col    = "p{2.4cm}"
    model_cols = " ".join(["p{2.6cm}"] * n_models)
    col_spec   = f"{law_col} {model_cols}"

    lines = []

    # ── comentario con instrucciones ──────────────────────────────────────
    lines.append(_LATEX_PREAMBLE_COMMENT.rstrip())
    lines.append(_latex_colour_comment())
    lines.append("% ─────────────────────────────────────────────────────────────────────────────\n")

    # ── entorno de tabla ─────────────────────────────────────────────────
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{Discovered equations -- {label}. "
                 "Cell colour encodes OOD MAE (see legend).}}")
    lines.append(f"  \\label{{tab:equations_{noise_key}}}")
    lines.append("  \\begin{adjustbox}{max width=\\linewidth}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # ── cabecera ─────────────────────────────────────────────────────────
    header_row = "    \\textbf{Law} & " + \
                 " & ".join(f"\\textbf{{{h}}}" for h in headers) + \
                 " \\\\"
    lines.append(header_row)
    lines.append("    \\midrule")

    # ── filas de datos ────────────────────────────────────────────────────
    for i, lk in enumerate(laws):
        law_label = law_names[lk]
        cells = []
        for j, model in enumerate(models):
            hex_bg, tex_name, text_col, mae, eq_full = cell_meta[i][j]
            eq_latex = _latex_eq(eq_full)

            # Color de texto: blanco si el fondo es muy oscuro
            if text_col == "#FFFFFF":
                eq_latex = f"\\textcolor{{white}}{{{eq_latex}}}"

            cell_content = f"\\cellcolor{{{tex_name}}}{eq_latex}"
            cells.append(cell_content)

        row = f"    {law_label} & " + " & ".join(cells) + " \\\\"
        lines.append(row)
        if i < n_laws - 1:
            lines.append("    \\addlinespace[2pt]")

    # ── cierre ────────────────────────────────────────────────────────────
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\end{adjustbox}")

    # Leyenda de colores como nota al pie de tabla
    lines.append("  \\\\[4pt]")
    lines.append("  \\footnotesize")
    legend_entries = []
    for _, hex_bg, tex_name, label_str in COLOUR_SCALE:
        legend_entries.append(
            f"\\colorbox{{{tex_name}}}{{\\strut\\,{label_str}\\,}}"
        )
    legend_entries.append(
        f"\\colorbox{{{NULL_TEX_NAME}}}{{\\strut\\,No OOD data\\,}}"
    )
    lines.append("  " + "\\quad ".join(legend_entries))
    lines.append("\\end{table}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"    → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── verificar ficheros de entrada ────────────────────────────────────
    for path in (RESULTS_TXT, OOD_JSON):
        if not os.path.exists(path):
            sys.exit(f"\n  ERROR: fichero no encontrado:\n  {path}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n── Cargando datos ──────────────────────────────────────────────")
    df  = parse_results(RESULTS_TXT)
    ood = load_ood(OOD_JSON)

    laws_present   = [l for l in LAW_ORDER   if l in df["law"].unique()]
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    law_names      = {lk: LAW_DISPLAY.get(lk, lk) for lk in laws_present}

    print(f"  {len(laws_present)} leyes · {len(models_present)} modelos")
    print(f"  Leyes OOD disponibles: {list(ood.keys())}")

    # ── generar una tabla por nivel de ruido ─────────────────────────────
    print("\n── Generando tablas ────────────────────────────────────────────")
    for noise_key in NOISE_KEYS:
        label = NOISE_LABELS[noise_key]
        print(f"\n  Tabla — {label}")

        cell_text, cell_meta = build_table(
            df, ood, noise_key, laws_present, models_present
        )

        base = os.path.join(OUT_DIR, f"table_{noise_key}")
        render_png(cell_text, cell_meta, laws_present, models_present,
                   noise_key, law_names, base + ".png")
        render_latex(cell_text, cell_meta, laws_present, models_present,
                     noise_key, law_names, base + ".tex")

    print(f"\n✓ 6 ficheros generados en:\n  {OUT_DIR}\n")


if __name__ == "__main__":
    main()
