import re
import pandas as pd
from pathlib import Path
import sympy as sp
import numpy as np

# =========================
# 1. GROUND TRUTH
# =========================
GROUND_TRUTH = {
    "coulomb": "q1*q2/r**2",
    "oscillator": "-x",
    "kepler": "sqrt(r**3)",
    "ideal_gas": "n*T/V",
    "projectile_range": "v0**2*sin(2*theta)",
    "time_dilation": "t/sqrt(1 - v**2)",
    "radioactive_decay": "exp(-lambda_*t)",
    "newton_cooling": "1 + exp(-k*t)",
    "boltzmann_entropy": "log(omega)"
}

# =========================
# 2. PARSER
# =========================
def parse_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    experiments = re.split(r"=+\nEXPERIMENT:", text)[1:]
    rows = []

    for exp in experiments:
        exp_name = exp.split("\n")[0].strip()

        model_blocks = re.findall(
            r"=== Model: (.*?) \| Dataset: (.*?) ===\n\n"
            r"Equation / Structure:\n(.*?)\n\n"
            r"Metrics:\nTest MSE: (.*?)\nTest MAE: (.*?)\n",
            exp,
            re.DOTALL
        )

        for model, dataset, equation, mse, mae in model_blocks:
            rows.append({
                "experiment": exp_name,
                "dataset": dataset,
                "model": model,
                "equation": equation.strip(),
                "mse": float(mse),
                "mae": float(mae)
            })

    return pd.DataFrame(rows)

# =========================
# 3. EXTRAER LEY
# =========================
def extract_law(dataset_name):
    for key in GROUND_TRUTH.keys():
        if key in dataset_name:
            return key
    return None

# =========================
# 4. VERIFICACIÓN
# =========================
def symbolic_equivalence(pred_eq, true_eq):
    try:
        symbols = sp.symbols('q1 q2 r x n T V v0 theta t v lambda_ k omega')
        sym_dict = {str(s): s for s in symbols}

        pred = sp.sympify(pred_eq, locals=sym_dict)
        true = sp.sympify(true_eq, locals=sym_dict)

        ratio = sp.simplify(pred / true)

        return ratio.is_number or len(ratio.free_symbols) == 0

    except Exception:
        return False


def numeric_equivalence(pred_eq, true_eq, n_tests=50):
    try:
        symbols = sp.symbols('q1 q2 r x n T V v0 theta t v lambda_ k omega')
        sym_dict = {str(s): s for s in symbols}

        pred = sp.sympify(pred_eq, locals=sym_dict)
        true = sp.sympify(true_eq, locals=sym_dict)

        vars_used = list(pred.free_symbols.union(true.free_symbols))

        pred_f = sp.lambdify(vars_used, pred, "numpy")
        true_f = sp.lambdify(vars_used, true, "numpy")

        ratios = []

        for _ in range(n_tests):
            vals = np.random.uniform(0.5, 2.0, len(vars_used))

            p = pred_f(*vals)
            t = true_f(*vals)

            if abs(t) < 1e-8:
                continue

            ratios.append(p / t)

        ratios = np.array(ratios)

        return len(ratios) > 0 and np.std(ratios) < 1e-2

    except Exception:
        return False


def is_equation_correct(row):
    law = extract_law(row["dataset"])
    if law is None:
        return False

    true_eq = GROUND_TRUTH[law]
    pred_eq = row["equation"]

    if symbolic_equivalence(pred_eq, true_eq):
        return True

    return numeric_equivalence(pred_eq, true_eq)

# =========================
# 5. LATEX TABLE
# =========================
def generate_latex_table(df, output_path):
    table = df.pivot_table(
        index=["law", "noise"],
        columns="model",
        values="is_correct",
        aggfunc="first"
    )

    table = table.fillna(False).astype(bool)
    table = table.replace({True: r"\checkmark", False: ""})

    latex_str = table.to_latex(
        escape=False,
        multirow=True,
        caption="Symbolic regression performance across physical laws.",
        label="tab:symbolic_results"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

# =========================
# 6. ANÁLISIS POR LEY
# =========================
def analyze_per_law(df):
    grouped = df.groupby(["law", "model"])

    stats = grouped.agg(
        accuracy=("is_correct", "mean"),
        avg_mse=("mse", "mean")
    ).reset_index()

    results = []

    for law in stats["law"].unique():
        subset = stats[stats["law"] == law]

        best = subset.sort_values(
            by=["accuracy", "avg_mse"],
            ascending=[False, True]
        ).iloc[0]

        results.append({
            "law": law,
            "best_model": best["model"],
            "accuracy": best["accuracy"],
            "avg_mse": best["avg_mse"]
        })

    return pd.DataFrame(results)


def generate_latex_winners(law_df, output_path):
    latex_str = law_df.to_latex(
        index=False,
        caption="Best model per physical law",
        label="tab:best_models"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

# =========================
# 8. RANKING LATEX
# =========================
def generate_latex_ranking(ranking_df, output_path):
    # redondear para que quede bonito en paper
    ranking_df = ranking_df.copy()
    ranking_df["accuracy"] = ranking_df["accuracy"].round(3)
    ranking_df["avg_mse"] = ranking_df["avg_mse"].round(4)
    ranking_df["std_mse"] = ranking_df["std_mse"].round(4)
    ranking_df["score"] = ranking_df["score"].round(4)

    # reset index para que "model" sea columna
    ranking_df = ranking_df.reset_index()

    # ordenar explícitamente
    ranking_df = ranking_df.sort_values("score", ascending=False)

    latex_str = ranking_df.to_latex(
        index=False,
        caption="Global ranking of symbolic regression models.",
        label="tab:ranking_models",
        float_format="%.4f"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

# =========================
# 7. MAIN
# =========================
def main():
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir.parent / "combined_results.txt"

    df = parse_results(file_path)

    # separar ley y ruido
    df[["law", "noise"]] = df["dataset"].str.rsplit("_", n=1, expand=True)

    # verificar ecuaciones
    df["is_correct"] = df.apply(is_equation_correct, axis=1)

    # =========================
    # TABLAS CSV
    # =========================
    table_eq = df.pivot_table(
        index=["law", "noise"],
        columns="model",
        values="equation",
        aggfunc="first"
    )

    table_mse = df.pivot_table(
        index=["law", "noise"],
        columns="model",
        values="mse"
    )

    table_mae = df.pivot_table(
        index=["law", "noise"],
        columns="model",
        values="mae"
    )

    # =========================
    # RANKING GLOBAL
    # =========================
    ranking = (
        df.groupby("model")
        .agg(
            accuracy=("is_correct", "mean"),
            avg_mse=("mse", "mean"),
            std_mse=("mse", "std")
        )
    )

    ranking["score"] = (
        ranking["accuracy"] * 0.7
        - ranking["avg_mse"] * 0.2
        - ranking["std_mse"] * 0.1
    )

    ranking = ranking.sort_values("score", ascending=False)

    # =========================
    # EXPORTAR CSV
    # =========================
    df.to_csv(base_dir / "raw_data.csv", index=False)
    table_eq.to_csv(base_dir / "equations.csv")
    table_mse.to_csv(base_dir / "mse.csv")
    table_mae.to_csv(base_dir / "mae.csv")
    ranking.to_csv(base_dir / "ranking_models.csv")

    # =========================
    # LATEX + ANÁLISIS
    # =========================
    generate_latex_table(df, base_dir / "results_table.tex")

    law_analysis = analyze_per_law(df)
    law_analysis.to_csv(base_dir / "per_law_analysis.csv", index=False)

    generate_latex_winners(law_analysis, base_dir / "best_models.tex")

    # =========================
    print("\nRANKING GLOBAL:\n")
    print(ranking)

    print("\nMEJOR MODELO POR LEY:\n")
    print(law_analysis)

    print("\nTodo generado en:", base_dir)
    
    generate_latex_ranking(ranking, base_dir / "ranking_models.tex")


# =========================
if __name__ == "__main__":
    main()
