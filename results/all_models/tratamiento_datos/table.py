"""
visualize_benchmark.py
======================
Benchmark de regresión simbólica — todas las visualizaciones.

Lee los datos directamente del fichero combined_results.txt (mismo directorio
por defecto, o indicado con --results). No hay datos hardcodeados: MSE,
ecuaciones y tasas de recuperación se calculan a partir del fichero de texto.

Figuras generadas
-----------------
  Fig 1   — Tarjetas de métricas globales
  Fig 2   — Heatmap de recuperación simbólica (3 niveles de ruido)
  Fig 3   — Scatter: tasa de recuperación vs. log-MSE global
  Fig 4   — Ranking global de modelos (estilo dashboard web)
  Fig 5a  — Tabla de ecuaciones descubiertas — sin ruido
  Fig 5b  — Tabla de ecuaciones descubiertas — ruido bajo
  Fig 5c  — Tabla de ecuaciones descubiertas — ruido alto
  Fig 6   — Heatmap numérico de MSE por modelo × ley
  Fig 7   — Evolución del MSE con el nivel de ruido (subplots por ley)
  Fig 8   — Radar de perfil de recuperación (modelos simbólicos)
  Fig 9   — Robustez al ruido (degradación de MSE)

Uso
---
  python visualize_benchmark.py
  python visualize_benchmark.py --results path/to/combined_results.txt
  python visualize_benchmark.py --show        # abre ventanas matplotlib
  python visualize_benchmark.py --fast        # clasificación por MSE (sin SymPy)

Clasificación "aproximada" (score 0.5)
---------------------------------------
La verificación automática sólo distingue correcto / incorrecto.
Para casos donde el modelo captura la estructura pero con constantes
libres (p.ej. QLattice en Kepler), añade entradas en APPROX_OVERRIDES
con el formato  (modelo, law_key, noise_idx) → 0.5
donde noise_idx: 0 = sin ruido, 1 = ruido bajo, 2 = ruido alto.

Nombres internos de las leyes (law_key)
-----------------------------------------
  boltzmann_entropy, coulomb, ideal_gas, kepler, newton_cooling,
  oscillator, projectile_range, radioactive_decay, time_dilation
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# AJUSTA AQUÍ las clasificaciones manuales de "aproximado" (score 0.5)
# Formato: (modelo, law_key, noise_idx): 0.5
# La verificación automática marca estas celdas como incorrectas,
# pero estructuralmente capturan la ley física.
# ══════════════════════════════════════════════════════════════════════════════
APPROX_OVERRIDES: dict = {
    # Ejemplos — descomenta o añade los que consideres:
    # ("QLattice", "boltzmann_entropy", 0): 0.5,
    # ("QLattice", "boltzmann_entropy", 1): 0.5,
    # ("QLattice", "boltzmann_entropy", 2): 0.5,
    # ("QLattice", "kepler",            0): 0.5,
    # ("QLattice", "kepler",            1): 0.5,
    # ("QLattice", "kepler",            2): 0.5,
    # ("QLattice", "ideal_gas",         1): 0.5,
    # ("QLattice", "ideal_gas",         2): 0.5,
    # ("QLattice", "oscillator",        0): 0.5,
}

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN VISUAL
# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

STYLE = {
    "bg":     "#FAFAF8",
    "card":   "#F3F2EE",
    "border": "#D3D1C7",
    "text":   "#2C2C2A",
    "muted":  "#888780",
    "ok":     "#639922",
    "approx": "#BA7517",
    "wrong":  "#D3D1C7",
    "danger": "#A32D2D",
    "ok_bg":  "#E8F3D6",
    "ap_bg":  "#FAF0D7",
    "no_bg":  "#FBEBEB",
}

MODEL_COLORS = {
    "GPLearn":      "#378ADD",
    "PySR":         "#639922",
    "QLattice":     "#BA7517",
    "PySINDy":      "#1D9E75",
    "Polynomial":   "#888780",
    "MLP_Standard": "#D4537E",
    "MLP_Sparse":   "#D85A30",
    "MLP_Dropout":  "#7F77DD",
}

# Orden de presentación de modelos
MODEL_ORDER = [
    "GPLearn", "PySR", "QLattice", "PySINDy",
    "Polynomial", "MLP_Standard", "MLP_Sparse", "MLP_Dropout",
]

SYMBOLIC_MODELS = {"GPLearn", "PySR", "QLattice", "PySINDy"}

# Orden y nombres legibles de las leyes
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

# Índice numérico de cada nivel de ruido
NOISE_IDX = {"no_noise": 0, "low_noise": 1, "high_noise": 2}
NOISE_LABELS = ["sin ruido", "ruido bajo", "ruido alto"]

# Modelos que aparecen en las tablas de ecuaciones (MLPs omitidos — no interpretables)
EQ_TABLE_MODELS = ["GPLearn", "PySR", "QLattice", "PySINDy", "Polynomial"]

plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["bg"],
    "axes.edgecolor":   STYLE["border"],
    "axes.labelcolor":  STYLE["text"],
    "text.color":       STYLE["text"],
    "xtick.color":      STYLE["muted"],
    "ytick.color":      STYLE["muted"],
    "grid.color":       STYLE["border"],
    "grid.linewidth":   0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})

# ══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════════════

GROUND_TRUTH = {
    "boltzmann_entropy": "log(omega)",
    "coulomb":           "q1*q2/r**2",
    "ideal_gas":         "n*T/V",
    "kepler":            "sqrt(r**3)",
    "newton_cooling":    "1 + exp(-k*t)",
    "oscillator":        "-x",
    "projectile_range":  "v0**2*sin(2*theta)",
    "radioactive_decay": "exp(-lambda_*t)",
    "time_dilation":     "t/sqrt(1 - v**2)",
}

# ══════════════════════════════════════════════════════════════════════════════
# PARSER  (basado en documento.py)
# ══════════════════════════════════════════════════════════════════════════════

def parse_results(file_path: str) -> pd.DataFrame:
    """Lee combined_results.txt y devuelve un DataFrame con todas las filas."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    experiments = re.split(r"=+\nEXPERIMENT:", text)[1:]
    rows = []
    for exp in experiments:
        exp_name = exp.split("\n")[0].strip()
        blocks = re.findall(
            r"=== Model: (.*?) \| Dataset: (.*?) ===\n\n"
            r"Equation / Structure:\n(.*?)\n\n"
            r"Metrics:\nTest MSE: (.*?)\nTest MAE: (.*?)\n",
            exp, re.DOTALL,
        )
        for model, dataset, equation, mse, mae in blocks:
            rows.append({
                "experiment": exp_name,
                "dataset":    dataset.strip(),
                "model":      model.strip(),
                "equation":   equation.strip(),
                "mse":        float(mse),
                "mae":        float(mae),
            })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN SIMBÓLICA  (basado en documento.py)
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess(eq: str) -> str:
    """Normaliza nombres de variables antes del parsing simbólico."""
    return re.sub(r"\blambd\b", "lambda_", eq)


def _sympy_env():
    try:
        import sympy as sp
        syms = sp.symbols("q1 q2 r x n T V v0 theta t v lambda_ k omega")
        return {str(s): s for s in syms}, sp
    except ImportError:
        return None, None


def symbolic_equivalence(pred: str, true: str) -> bool:
    env, sp = _sympy_env()
    if sp is None:
        return False
    try:
        p = sp.sympify(pred, locals=env)
        t = sp.sympify(true, locals=env)
        r = sp.simplify(p / t)
        return r.is_number or len(r.free_symbols) == 0
    except Exception:
        return False


def numeric_equivalence(pred: str, true: str, n: int = 50) -> bool:
    env, sp = _sympy_env()
    if sp is None:
        return False
    try:
        p = sp.sympify(pred, locals=env)
        t = sp.sympify(true, locals=env)
        vs = list(p.free_symbols | t.free_symbols)
        pf = sp.lambdify(vs, p, "numpy")
        tf = sp.lambdify(vs, t, "numpy")
        ratios = []
        rng = np.random.default_rng(42)
        for _ in range(n):
            vals = rng.uniform(0.5, 2.0, len(vs))
            pv, tv = pf(*vals), tf(*vals)
            if abs(tv) > 1e-8:
                ratios.append(pv / tv)
        ratios = np.array(ratios)
        return len(ratios) > 5 and np.std(ratios) < 1e-2
    except Exception:
        return False


def check_correct(eq: str, law_key: str) -> bool:
    """Devuelve True si la ecuación predicha es equivalente a la verdad."""
    if "Red Neuronal" in eq:
        return False
    true = GROUND_TRUTH.get(law_key)
    if true is None:
        return False
    eq = _preprocess(eq)
    return symbolic_equivalence(eq, true) or numeric_equivalence(eq, true)


def check_correct_fast(eq: str, mse: float, threshold: float = 1e-6) -> bool:
    """Clasificación rápida por umbral de MSE (sin SymPy)."""
    return ("Red Neuronal" not in eq) and (mse < threshold)

# ══════════════════════════════════════════════════════════════════════════════
# CARGA Y CONSTRUCCIÓN DE ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_law_noise(dataset: str):
    """
    'boltzmann_entropy_no_noise' → ('boltzmann_entropy', 'no_noise')
    La ley puede contener guiones bajos, el ruido siempre es el último token o par.
    """
    for suffix in ("_no_noise", "_low_noise", "_high_noise"):
        if dataset.endswith(suffix):
            return dataset[: -len(suffix)], suffix[1:]
    # fallback: último token
    parts = dataset.rsplit("_", 1)
    return parts[0], parts[1] if len(parts) == 2 else ("unknown", "no_noise")


def load_data(results_path: str, fast: bool = False) -> dict:
    """
    Parsea combined_results.txt y construye todos los diccionarios
    necesarios para generar las figuras.

    Devuelve:
        laws        — lista ordenada de law_keys presentes en los datos
        models      — lista ordenada de modelos presentes
        law_names   — {law_key: nombre legible}
        mse_by_noise— {model: {law: [mse_no, mse_low, mse_high]}}
        mse_global  — {model: {law: media de los 3 niveles}}
        recovery    — {model: {law: [score_no, score_low, score_high]}}
                      score: 1=exacto, 0.5=aprox (override), 0=incorrecto
        equations   — {noise_key: {law: {model: (eq_str, score)}}}
        df          — DataFrame completo con columna is_correct
    """
    print(f"  Leyendo: {results_path}")
    df = parse_results(results_path)

    # Separar ley y nivel de ruido
    extracted = df["dataset"].apply(_extract_law_noise)
    df["law"]   = extracted.apply(lambda x: x[0])
    df["noise"] = extracted.apply(lambda x: x[1])
    df["noise_idx"] = df["noise"].map(NOISE_IDX).fillna(-1).astype(int)

    # Ordenar leyes y modelos según los órdenes preferidos
    laws_present   = [l for l in LAW_ORDER   if l in df["law"].unique()]
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]

    # ── Verificación de corrección ────────────────────────────────────────
    if fast:
        print("  Clasificando ecuaciones (modo rápido — umbral MSE)…")
        df["is_correct"] = df.apply(
            lambda r: check_correct_fast(r["equation"], r["mse"]), axis=1
        )
    else:
        print("  Verificando equivalencia simbólica", end="", flush=True)
        np.random.seed(42)
        res = []
        for _, row in df.iterrows():
            res.append(check_correct(row["equation"], row["law"]))
            print(".", end="", flush=True)
        print()
        df["is_correct"] = res

    # ── MSE_BY_NOISE ──────────────────────────────────────────────────────
    mse_by_noise: dict = {}
    for model in models_present:
        mse_by_noise[model] = {}
        for law in laws_present:
            vals = [np.nan, np.nan, np.nan]
            sub = df[(df["model"] == model) & (df["law"] == law)]
            for _, row in sub.iterrows():
                ni = int(row["noise_idx"])
                if 0 <= ni <= 2:
                    vals[ni] = float(row["mse"])
            mse_by_noise[model][law] = vals

    # ── MSE_GLOBAL (media de los 3 niveles) ───────────────────────────────
    mse_global: dict = {}
    for model in models_present:
        mse_global[model] = {}
        for law in laws_present:
            v = [x for x in mse_by_noise[model][law] if not np.isnan(x)]
            mse_global[model][law] = float(np.mean(v)) if v else np.nan

    # ── RECOVERY ──────────────────────────────────────────────────────────
    recovery: dict = {}
    for model in models_present:
        recovery[model] = {}
        for law in laws_present:
            scores = [0, 0, 0]
            sub = df[(df["model"] == model) & (df["law"] == law)]
            for _, row in sub.iterrows():
                ni = int(row["noise_idx"])
                if 0 <= ni <= 2 and row["is_correct"]:
                    scores[ni] = 1
            # Overrides manuales de "aproximado"
            for ni in range(3):
                key = (model, law, ni)
                if key in APPROX_OVERRIDES and scores[ni] == 0:
                    scores[ni] = APPROX_OVERRIDES[key]
            recovery[model][law] = scores

    # ── EQUATIONS ────────────────────────────────────────────────────────
    equations: dict = {nk: {} for nk in NOISE_IDX}
    for noise_key in NOISE_IDX:
        for law in laws_present:
            equations[noise_key][law] = {"Verdad": GROUND_TRUTH.get(law, "?")}
            for model in models_present:
                sub = df[
                    (df["model"] == model) &
                    (df["law"]   == law)   &
                    (df["noise"] == noise_key)
                ]
                if not sub.empty:
                    row = sub.iloc[0]
                    ni  = NOISE_IDX[noise_key]
                    equations[noise_key][law][model] = (
                        row["equation"],
                        recovery[model][law][ni],
                    )
                else:
                    equations[noise_key][law][model] = ("—", 0)

    return {
        "df":           df,
        "laws":         laws_present,
        "models":       models_present,
        "law_names":    {lk: LAW_DISPLAY.get(lk, lk) for lk in laws_present},
        "mse_by_noise": mse_by_noise,
        "mse_global":   mse_global,
        "recovery":     recovery,
        "equations":    equations,
    }

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def savefig(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"], edgecolor="none")
    print(f"    → {path}")


def section_title(ax, text: str):
    ax.set_title(text, loc="left", fontsize=8.5, fontweight="500",
                 color=STYLE["muted"], pad=10)


def recovery_rate(recovery: dict, model: str, laws: list) -> float:
    """Tasa de recuperación ponderada (n_laws × 3 condiciones)."""
    return float(np.mean([v for lk in laws for v in recovery[model][lk]]))


def trunc(s: str, n: int = 46) -> str:
    """Trunca una ecuación para que quepa en una celda de tabla."""
    s = (s.replace("**2", "²").replace("**3", "³")
          .replace("**4", "⁴").replace("**5", "⁵"))
    return s if len(s) <= n else s[: n - 1] + "…"

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — TARJETAS DE MÉTRICAS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

def fig1_metric_cards(data: dict):
    laws, models, recovery, mse_global, law_names = (
        data["laws"], data["models"], data["recovery"],
        data["mse_global"], data["law_names"],
    )
    total = len(laws) * 3

    # Mejor modelo simbólico
    best_m, best_s = max(
        ((m, recovery_rate(recovery, m, laws)) for m in models),
        key=lambda x: x[1],
    )
    best_exact = sum(1 for lk in laws for v in recovery[best_m][lk] if v == 1)

    # GPLearn exactas
    gp_exact = (sum(1 for lk in laws for v in recovery["GPLearn"][lk] if v == 1)
                if "GPLearn" in recovery else "—")

    # Leyes sin resolver por nadie
    unsolved = [
        law_names[lk] for lk in laws
        if all(recovery[m][lk][ni] == 0
               for m in models for ni in range(3))
    ]

    # MLPs exactas
    mlp_exact = sum(
        1 for m in models if "MLP" in m
        for lk in laws for v in recovery[m][lk] if v == 1
    )

    cards = [
        (f"{best_exact}/{total}",
         f"{best_m} — mejor recuperación simbólica ({best_s*100:.0f}%)",
         STYLE["ok"]),
        (f"{gp_exact}/{total}",
         "GPLearn — falla catastróficamente en algunas leyes",
         STYLE["text"]),
        (f"{len(unsolved)} ley{'es' if len(unsolved) != 1 else ''}",
         ("Sin resolver: " + ", ".join(unsolved)) if unsolved else "Todas resueltas por ≥1 modelo",
         STYLE["danger"] if unsolved else STYLE["ok"]),
        (f"{mlp_exact}/{total}",
         "Recuperaciones simbólicas por MLPs (0 — nunca interpretan la ley)",
         STYLE["muted"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 2.4))
    fig.patch.set_facecolor(STYLE["bg"])
    for ax, (val, sub, col) in zip(axes, cards):
        ax.set_facecolor(STYLE["card"])
        for sp in ax.spines.values():
            sp.set_edgecolor(STYLE["border"]); sp.set_linewidth(0.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.65, val, transform=ax.transAxes,
                ha="center", va="center", fontsize=20, fontweight="500", color=col)
        ax.text(0.5, 0.22, sub, transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=STYLE["muted"], wrap=True)

    fig.suptitle("Benchmark de regresión simbólica — métricas globales",
                 fontsize=11, fontweight="500", y=1.04, color=STYLE["text"])
    fig.tight_layout(pad=0.6)
    savefig(fig, "fig1_metric_cards.png")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — HEATMAP DE RECUPERACIÓN SIMBÓLICA
# ══════════════════════════════════════════════════════════════════════════════

def fig2_recovery_heatmap(data: dict):
    laws, models, recovery, law_names = (
        data["laws"], data["models"], data["recovery"], data["law_names"]
    )
    n_m, n_l = len(models), len(laws)
    offsets_x = [-0.27, 0.0, 0.27]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_facecolor(STYLE["bg"])
    ax.set_aspect("equal")

    for mi, model in enumerate(models):
        for li, lk in enumerate(laws):
            cx, cy = float(li), float(n_m - 1 - mi)
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx - 0.45, cy - 0.40), 0.90, 0.80,
                boxstyle="round,pad=0.04", linewidth=0, facecolor=STYLE["card"]
            ))
            for ni, v in enumerate(recovery[model][lk]):
                col = (STYLE["ok"] if v == 1 else
                       STYLE["approx"] if v == 0.5 else STYLE["wrong"])
                ax.add_patch(plt.Circle(
                    (cx + offsets_x[ni], cy), 0.11, color=col, zorder=3
                ))

    for li, lk in enumerate(laws):
        ax.text(float(li), float(n_m) - 0.1, law_names[lk],
                ha="center", va="bottom", fontsize=7.5,
                color=STYLE["muted"], rotation=30)
    for mi, model in enumerate(models):
        ax.text(-0.6, float(n_m - 1 - mi), model,
                ha="right", va="center", fontsize=9, color=STYLE["text"])

    for label, col in [("Exacta", STYLE["ok"]),
                        ("Aproximada", STYLE["approx"]),
                        ("Incorrecta", STYLE["wrong"])]:
        ax.scatter([], [], s=60, color=col, label=label)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5,
              labelcolor=STYLE["muted"])

    ax.set_xlim(-1.2, n_l - 0.4)
    ax.set_ylim(-0.6, n_m + 1.2)
    ax.axis("off")
    section_title(ax, "RECUPERACIÓN SIMBÓLICA POR MODELO Y LEY  ·  "
                  "3 puntos = sin ruido · ruido bajo · ruido alto")
    fig.tight_layout()
    savefig(fig, "fig2_recovery_heatmap.png")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — SCATTER: TASA DE RECUPERACIÓN vs. LOG-MSE
# ══════════════════════════════════════════════════════════════════════════════

def fig3_scatter(data: dict):
    laws, models, recovery, mse_global = (
        data["laws"], data["models"], data["recovery"], data["mse_global"]
    )
    label_offsets = {
        "GPLearn":      (5, -12), "PySR":         (5,   8),
        "QLattice":     (5,   0), "PySINDy":      (5, -12),
        "Polynomial":   (-84, 6), "MLP_Standard": (5, -12),
        "MLP_Sparse":   (5,   4), "MLP_Dropout":  (5,  14),
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for model in models:
        x   = recovery_rate(recovery, model, laws) * 100
        mse = [mse_global[model][lk] for lk in laws
               if not np.isnan(mse_global[model][lk])]
        if not mse:
            continue
        y   = np.log10(np.mean(mse))
        col = MODEL_COLORS.get(model, "#888780")
        ax.scatter(x, y, s=140, color=col,
                   marker="D" if model in SYMBOLIC_MODELS else "o",
                   zorder=4, edgecolors="white", linewidths=0.8)
        dx, dy = label_offsets.get(model, (5, 0))
        ax.annotate(model, (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color=col, va="center")

    ax.axvline(50, color=STYLE["border"], lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Tasa de recuperación simbólica (%)", fontsize=9)
    ax.set_ylabel("log₁₀(MSE promedio global)", fontsize=9)
    ax.grid(True, alpha=0.5)
    ax.scatter([], [], marker="D", s=80, color=STYLE["text"],
               label="Regresor simbólico", edgecolors="white", lw=0.8)
    ax.scatter([], [], marker="o", s=80, color=STYLE["text"],
               label="Caja negra", edgecolors="white", lw=0.8)
    ax.legend(fontsize=8.5, frameon=False)
    section_title(ax, "TASA DE RECUPERACIÓN VS. PRECISIÓN NUMÉRICA GLOBAL")
    fig.tight_layout()
    savefig(fig, "fig3_scatter_recovery_vs_mse.png")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — RANKING GLOBAL (ESTILO DASHBOARD WEB)
# ══════════════════════════════════════════════════════════════════════════════

def fig4_ranking(data: dict):
    laws, models, recovery, mse_global = (
        data["laws"], data["models"], data["recovery"], data["mse_global"]
    )
    total = len(laws) * 3

    stats = []
    for model in models:
        exact  = sum(1 for lk in laws for v in recovery[model][lk] if v == 1)
        approx = sum(1 for lk in laws for v in recovery[model][lk] if v == 0.5)
        score  = (exact + 0.5 * approx) / total
        mse_v  = [mse_global[model][lk] for lk in laws
                  if not np.isnan(mse_global[model][lk])]
        avg_mse = float(np.mean(mse_v)) if mse_v else np.nan
        std_mse = float(np.std(mse_v))  if mse_v else np.nan
        # Score compuesto: prioriza recuperación, penaliza error
        composite = score * 0.7 - (np.log10(avg_mse) * 0.05
                                    if avg_mse and avg_mse > 0 else 0)
        stats.append(dict(model=model, exact=exact, approx=approx,
                          score=score, avg_mse=avg_mse, std_mse=std_mse,
                          composite=composite))

    stats.sort(key=lambda r: (r["composite"], -r["avg_mse"]), reverse=True)
    n = len(stats)

    fig, ax = plt.subplots(figsize=(11, 0.75 * n + 2.0))
    bar_h = 0.52

    for rank, s in enumerate(stats):
        model = s["model"]
        col   = MODEL_COLORS.get(model, "#888780")
        y     = n - 1 - rank

        w_ex = s["exact"]  / total
        w_ap = s["approx"] / total

        # Barra de exactas
        ax.barh(y, w_ex, height=bar_h, color=col, alpha=0.90, zorder=3)
        # Barra de aproximadas (apilada)
        if w_ap > 0:
            ax.barh(y, w_ap, height=bar_h, color=col, alpha=0.35,
                    hatch="///", zorder=3, left=w_ex)

        # Porcentaje
        pct = (w_ex + w_ap) * 100
        ax.text(w_ex + w_ap + 0.013, y,
                f"{pct:.0f}%",
                va="center", fontsize=8.5, fontweight="500", color=col)

        # Conteo exactas + aproximadas
        ax.text(0.575, y,
                f"{s['exact']}/{total}  +{s['approx']}≈",
                va="center", ha="left", fontsize=7.5, color=STYLE["muted"],
                transform=ax.get_yaxis_transform())

        # MSE
        ax.text(0.735, y,
                f"MSE = {s['avg_mse']:.2e}",
                va="center", ha="left", fontsize=7.5, color=STYLE["muted"],
                transform=ax.get_yaxis_transform())

        # Badge tipo de modelo
        is_sym = model in SYMBOLIC_MODELS
        badge  = "◆ Simbólico" if is_sym else "● Caja negra"
        ax.text(-0.015, y, badge, ha="right", va="center",
                fontsize=6.5, color=col if is_sym else STYLE["muted"])

        # Posición en ranking
        ax.text(-0.165, y, f"#{rank + 1}", ha="center", va="center",
                fontsize=8.5, fontweight="500",
                color=STYLE["ok"] if rank == 0 else STYLE["muted"])

    ax.set_yticks(range(n))
    ax.set_yticklabels([s["model"] for s in reversed(stats)], fontsize=9)
    ax.set_xlim(-0.22, 1.08)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Tasa de recuperación simbólica (fracción del total de condiciones)",
                  fontsize=9)
    ax.axvline(0, color=STYLE["border"], lw=0.5)
    ax.grid(axis="x", alpha=0.35)

    ax.barh([], [], color=STYLE["muted"], alpha=0.90,
            label="Recuperación exacta")
    ax.barh([], [], color=STYLE["muted"], alpha=0.35, hatch="///",
            label="Recuperación aproximada")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    section_title(ax, "RANKING GLOBAL DE MODELOS")
    fig.tight_layout()
    savefig(fig, "fig4_ranking.png")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5a / 5b / 5c — TABLAS DE ECUACIONES (genérica por nivel de ruido)
# ══════════════════════════════════════════════════════════════════════════════

def _equations_table(data: dict, noise_key: str, title_suffix: str, filename: str):
    """
    Tabla: Ley | Verdad | GPLearn | PySR | QLattice | PySINDy | Polynomial
    Cada celda de modelo se colorea según score: verde/ámbar/rojo.
    """
    laws, law_names, equations, models = (
        data["laws"], data["law_names"], data["equations"], data["models"]
    )
    eq_data   = equations[noise_key]
    cols_show = [m for m in EQ_TABLE_MODELS if m in models]
    headers   = ["Ley", "Verdad"] + cols_show
    n_rows    = len(laws)
    n_cols    = len(headers)

    # Anchuras de columna proporcionales al contenido esperado
    col_widths = [0.10, 0.11] + [0.155] * len(cols_show)

    row_data    = []
    cell_colors = []

    for lk in laws:
        eq_dict = eq_data.get(lk, {})
        row   = [law_names[lk], GROUND_TRUTH.get(lk, "?")]
        ccols = [STYLE["card"], STYLE["card"]]
        for m in cols_show:
            eq_str, score = eq_dict.get(m, ("—", 0))
            row.append(trunc(eq_str))
            ccols.append(
                STYLE["ok_bg"] if score == 1 else
                STYLE["ap_bg"] if score == 0.5 else
                STYLE["no_bg"]
            )
        row_data.append(row)
        cell_colors.append(ccols)

    fig_h = max(5.5, n_rows * 0.95 + 2.2)
    fig, ax = plt.subplots(figsize=(19, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=row_data,
        colLabels=headers,
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    # Cabecera
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(STYLE["card"])
        cell.set_text_props(color=STYLE["muted"], fontweight="bold")
        cell.set_edgecolor(STYLE["border"])
        cell.set_height(0.085)

    # Celdas de contenido
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor(cell_colors[i - 1][j])
            cell.set_edgecolor(STYLE["border"])
            cell.set_text_props(
                color=STYLE["text"],
                fontweight="500" if j < 2 else "normal"
            )
            cell.PAD = 0.045
            cell.set_height(0.072)

    # Leyenda
    for label, col in [("Exacta",     STYLE["ok_bg"]),
                        ("Aproximada", STYLE["ap_bg"]),
                        ("Incorrecta", STYLE["no_bg"])]:
        ax.barh([], [], color=col, label=label,
                edgecolor=STYLE["border"], linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True,
              framealpha=0.95, edgecolor=STYLE["border"])

    section_title(ax, f"ECUACIONES DESCUBIERTAS — {title_suffix.upper()}")
    fig.tight_layout()
    savefig(fig, filename)
    return fig


def fig5a_equations_no_noise(data: dict):
    return _equations_table(
        data, "no_noise", "sin ruido", "fig5a_equations_no_noise.png"
    )


def fig5b_equations_low_noise(data: dict):
    return _equations_table(
        data, "low_noise", "ruido bajo", "fig5b_equations_low_noise.png"
    )


def fig5c_equations_high_noise(data: dict):
    return _equations_table(
        data, "high_noise", "ruido alto", "fig5c_equations_high_noise.png"
    )

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — HEATMAP NUMÉRICO DE MSE
# ══════════════════════════════════════════════════════════════════════════════

def fig6_mse_heatmap(data: dict):
    laws, models, mse_global, law_names = (
        data["laws"], data["models"], data["mse_global"], data["law_names"]
    )
    matrix = np.array([
        [np.log10(mse_global[m][lk])
         if not np.isnan(mse_global[m][lk]) and mse_global[m][lk] > 0
         else np.nan
         for lk in laws]
        for m in models
    ])

    fig, ax = plt.subplots(figsize=(13, 5.5))
    cmap = LinearSegmentedColormap.from_list(
        "mse_cmap", [STYLE["ok_bg"], STYLE["ap_bg"], STYLE["no_bg"]]
    )
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(laws)))
    ax.set_xticklabels([law_names[lk] for lk in laws],
                       rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    for mi, m in enumerate(models):
        for li, lk in enumerate(laws):
            v = mse_global[m][lk]
            if not np.isnan(v):
                ax.text(li, mi, f"{v:.1e}",
                        ha="center", va="center", fontsize=6, color=STYLE["text"])

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("log₁₀(MSE promedio)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    section_title(ax, "MSE NUMÉRICO POR LEY Y MODELO  (verde = menor error)")
    fig.tight_layout()
    savefig(fig, "fig6_mse_heatmap.png")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — EVOLUCIÓN DEL MSE CON EL NIVEL DE RUIDO
# ══════════════════════════════════════════════════════════════════════════════

def fig7_noise_sensitivity(data: dict):
    laws, models, mse_by_noise, law_names = (
        data["laws"], data["models"],
        data["mse_by_noise"], data["law_names"]
    )
    ncols = 3
    nrows = (len(laws) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.2, nrows * 3.3),
                             sharey=False)
    axes = np.array(axes).flatten()

    for idx, lk in enumerate(laws):
        ax = axes[idx]
        for model in models:
            vals = mse_by_noise[model][lk]
            ylog = [np.log10(v) if v and v > 0 else np.nan for v in vals]
            ax.plot([0, 1, 2], ylog,
                    color=MODEL_COLORS.get(model, "#888780"),
                    linewidth=1.4, marker="o", markersize=4, label=model)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(NOISE_LABELS, fontsize=7, rotation=15)
        ax.set_title(law_names[lk], fontsize=9, fontweight="500",
                     color=STYLE["text"])
        ax.set_ylabel("log₁₀(MSE)", fontsize=7.5)
        ax.grid(True, alpha=0.4)

    for ax in axes[len(laws):]:
        ax.axis("off")

    handles = [mpatches.Patch(color=MODEL_COLORS.get(m, "#888780"), label=m)
               for m in models]
    fig.legend(handles=handles, loc="lower right", ncol=2,
               fontsize=7.5, frameon=False, bbox_to_anchor=(1.0, 0.01))
    fig.suptitle("Evolución del MSE con el nivel de ruido — por ley física",
                 fontsize=11, fontweight="500", y=1.01, color=STYLE["text"])
    fig.tight_layout()
    savefig(fig, "fig7_noise_sensitivity.png")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — RADAR DE PERFIL DE RECUPERACIÓN (modelos simbólicos)
# ══════════════════════════════════════════════════════════════════════════════

def fig8_radar(data: dict):
    laws, recovery, law_names, models = (
        data["laws"], data["recovery"], data["law_names"], data["models"]
    )
    selected = [m for m in ["GPLearn", "PySR", "QLattice", "PySINDy"]
                if m in models]

    n_ax   = len(laws)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.set_facecolor(STYLE["bg"])

    for model in selected:
        values = [float(np.mean(recovery[model][lk])) for lk in laws] + [0]
        values[-1] = values[0]   # cerrar el polígono
        col = MODEL_COLORS.get(model, "#888780")
        ax.plot(angles, values, color=col, linewidth=1.8, label=model)
        ax.fill(angles, values, color=col, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]),
                      labels=[law_names[lk] for lk in laws], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"],
                       fontsize=6.5, color=STYLE["muted"])
    ax.grid(color=STYLE["border"], alpha=0.6)
    ax.spines["polar"].set_edgecolor(STYLE["border"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.36, 1.14),
              fontsize=8.5, frameon=False)
    ax.set_title("Perfil de recuperación por ley — modelos simbólicos",
                 pad=20, fontsize=10, fontweight="500", color=STYLE["text"])
    fig.tight_layout()
    savefig(fig, "fig8_radar_recovery.png")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — ROBUSTEZ AL RUIDO
# ══════════════════════════════════════════════════════════════════════════════

def fig9_noise_robustness(data: dict):
    laws, models, mse_by_noise = (
        data["laws"], data["models"], data["mse_by_noise"]
    )
    degradations = {}
    for model in models:
        deltas = []
        for lk in laws:
            v0, v2 = mse_by_noise[model][lk][0], mse_by_noise[model][lk][2]
            if v0 and v2 and v0 > 0 and v2 > 0:
                deltas.append(np.log10(v2) - np.log10(v0))
        degradations[model] = (
            float(np.mean(deltas)) if deltas else 0.0,
            float(np.std(deltas))  if deltas else 0.0,
        )

    ordered = sorted(models, key=lambda m: degradations[m][0])
    means   = [degradations[m][0] for m in ordered]
    stds    = [degradations[m][1] for m in ordered]
    colors  = [MODEL_COLORS.get(m, "#888780") for m in ordered]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.barh(range(len(ordered)), means, color=colors, alpha=0.85,
            height=0.6, zorder=3)
    ax.errorbar(means, range(len(ordered)), xerr=stds,
                fmt="none", color=STYLE["text"], capsize=4,
                linewidth=1.2, zorder=4)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=9)
    ax.axvline(0, color=STYLE["border"], lw=0.8)
    ax.set_xlabel(
        "Δ log₁₀(MSE)  de sin ruido → ruido alto  (menor = más robusto)",
        fontsize=9)
    ax.grid(axis="x", alpha=0.4)
    for i, (val, std) in enumerate(zip(means, stds)):
        ax.text(val + 0.1, i, f"{val:+.1f} ±{std:.1f}",
                va="center", fontsize=7.5, color=STYLE["muted"])
    section_title(ax, "ROBUSTEZ AL RUIDO — DEGRADACIÓN DE MSE (sin ruido → ruido alto)")
    fig.tight_layout()
    savefig(fig, "fig9_noise_robustness.png")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Genera todas las figuras del benchmark de regresión simbólica."
    )
    parser.add_argument(
        "--results", default=None,
        help="Ruta al fichero combined_results.txt. Por defecto busca en el "
             "mismo directorio del script y un nivel arriba.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Abre las ventanas de matplotlib además de guardar los PNGs.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Usa umbral de MSE en vez de verificación simbólica con SymPy "
             "(mucho más rápido, menos preciso).",
    )
    args = parser.parse_args()

    # ── Localizar combined_results.txt ───────────────────────────────────
    if args.results:
        results_path = args.results
    else:
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        candidates   = [
            os.path.join(script_dir, "combined_results.txt"),
            os.path.join(os.path.dirname(script_dir), "combined_results.txt"),
        ]
        results_path = next((p for p in candidates if os.path.exists(p)), None)

    if results_path is None or not os.path.exists(results_path):
        sys.exit(
            "\n  ERROR: no se encontró combined_results.txt\n"
            "  Indica la ruta con:  --results /ruta/al/fichero.txt\n"
        )

    # ── Cargar y procesar datos ──────────────────────────────────────────
    print("\n── Cargando datos ─────────────────────────────────────────────")
    data = load_data(results_path, fast=args.fast)
    print(f"  {len(data['laws'])} leyes · {len(data['models'])} modelos")

    # ── Generar figuras ──────────────────────────────────────────────────
    all_figs = [
        ("1   — Tarjetas de métricas globales",           fig1_metric_cards),
        ("2   — Heatmap de recuperación simbólica",        fig2_recovery_heatmap),
        ("3   — Scatter recuperación vs. MSE",             fig3_scatter),
        ("4   — Ranking global (estilo web)",              fig4_ranking),
        ("5a  — Tabla de ecuaciones — sin ruido",          fig5a_equations_no_noise),
        ("5b  — Tabla de ecuaciones — ruido bajo",         fig5b_equations_low_noise),
        ("5c  — Tabla de ecuaciones — ruido alto",         fig5c_equations_high_noise),
        ("6   — Heatmap numérico MSE",                     fig6_mse_heatmap),
        ("7   — Evolución MSE con nivel de ruido",         fig7_noise_sensitivity),
        ("8   — Radar de perfil por ley",                  fig8_radar),
        ("9   — Robustez al ruido",                        fig9_noise_robustness),
    ]

    print(f"\n── Generando {len(all_figs)} figuras en ./{OUT_DIR}/ "
          "────────────────────")
    for label, fn in all_figs:
        print(f"\n  Fig {label}")
        fn(data)

    print(f"\n✓ Completado — {len(all_figs)} figuras guardadas en ./{OUT_DIR}/\n")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()