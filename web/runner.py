"""
symbolic-physics-discovery / web / runner.py

Adapter between the Flask web app and the src/ model library.
- Adds src/ to sys.path so wrappers can be imported.
- Redirects ALL file outputs to web/outputs/<job_id>/
- Never writes into results/, data/, or any other project folder.
"""

import os
import sys
import random
import json
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt

# ── Resolución estricta de topología de directorios ────────────────────────────
WEB_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEB_DIR.parent

SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = ROOT_DIR / "data"

# ── Seeding ────────────────────────────────────────────────────────────────────
def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

# ── Main entry called by app.py ────────────────────────────────────────────────
def run_experiment_internal(job_id: str, config: dict, output_dir: Path, log_fn) -> list:
    """
    config shape:
    {
        dataset: {
            type: "default" | "upload",
            name: "oscillator_no_noise.csv",
            file_id: "<uuid>",
            target_col: "F",
            column_map: {"old_name": "new_name"}
        },
        models: ["MLP_Standard", "PySR", ...],
        hyperparams: {
            "MLP_Standard": { epochs: 100, lr: 0.001 },
            ...
        }
    }
    Returns list of result dicts: [{ model, mse, mae, equation }, ...]
    """
    _set_seed(42)

    # ── 1. Locate dataset file ─────────────────────────────────────────────────
    ds = config.get("dataset", {})

    if ds.get("type") == "upload":
        filepath = str(WEB_DIR / "uploads" / f"{ds['file_id']}.csv")
    else:
        filepath = str(DATA_DIR / ds["name"])

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset no encontrado: {filepath}. Verifica la ruta {DATA_DIR}.")

    col_map = ds.get("column_map", {})
    target_col = ds["target_col"]

    if col_map:
        import pandas as pd
        df = pd.read_csv(filepath)
        df = df.rename(columns=col_map)
        tmp = output_dir / "_dataset_renamed.csv"
        df.to_csv(tmp, index=False)
        filepath = str(tmp)

    dataset_name = Path(ds.get("name", filepath)).stem

    log_fn(f"Dataset : {Path(filepath).name}")
    log_fn(f"   Target  : {target_col}")

    # ── 2. Load with PhysicalDataset ──────────────────────────────────────────
    from data.loader import PhysicalDataset # Aparecerá como un error de PyLance pero no pasa nada
    dataset = PhysicalDataset(filepath, target_col=target_col, scale=True)

    X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_latent_arrays()
    Xp_train, Xp_val, Xp_test, yp_train, yp_val, yp_test = dataset.get_physical_arrays()

    log_fn(f"   Split   : train={len(X_train)} / val={len(X_val)} / test={len(X_test)}")

    # ── 3. Run each selected model ────────────────────────────────────────────
    selected = config.get("models", [])
    hyperparams = config.get("hyperparams", {})
    summary = []

    for model_name in selected:
        log_fn(f"\n{model_name}")
        hp = hyperparams.get(model_name, {})

        try:
            model = _build_model(model_name, dataset, hp, output_dir)

            # ── Fit ────────────────────────────────────────────────────────────
            if "MLP" in model_name:
                import torch
                from torch.utils.data import TensorDataset, DataLoader
                
                train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                         torch.tensor(y_train, dtype=torch.float32))
                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                
                val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                       torch.tensor(y_val, dtype=torch.float32))
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
                
                model.fit(train_loader, val_loader)
                
                y_pred = dataset.scaler_y.inverse_transform(
                    model.predict(X_test)
                ).flatten()
                
            elif model_name == "Polynomial":
                model.fit(X_train, y_train)
                y_pred = dataset.scaler_y.inverse_transform(
                    model.predict(X_test)
                ).flatten()
                
            elif model_name in ("PySR", "QLattice"):
                model.fit(Xp_train, yp_train, X_val=Xp_val, y_val=yp_val)
                y_pred = model.predict(Xp_test).flatten()
                
            else:
                model.fit(Xp_train, yp_train)
                y_pred = model.predict(Xp_test).flatten()

            y_true = yp_test.flatten()

            # ── Metrics ────────────────────────────────────────────────────────
            from utils.metrics import evaluate_physical_space
            mse, mae = evaluate_physical_space(y_true, y_pred)

            log_fn(f"   MSE : {mse:.4e}  |  MAE : {mae:.4e}")
            log_fn(f"   Eq  : {model.equation}")

            # ── Save outputs ───────────────────────────────────────────────────
            _save_txt(output_dir, model_name, dataset_name, mse, mae, model.equation)
            _save_residuals(output_dir, model_name, dataset_name, y_true, y_pred, target_col)

            hist = getattr(model, "history", None)
            if hist and hist.get("train_loss"):
                _save_loss_curve(output_dir, model_name, dataset_name, hist)

            summary.append({
                "model":    model_name,
                "mse":      float(mse),
                "mae":      float(mae),
                "equation": str(model.equation),
            })

        except Exception as exc:
            log_fn(f"   [ERROR] {exc}")
            log_fn(traceback.format_exc())
            summary.append({
                "model":    model_name,
                "error":    str(exc),
            })

    # ── 4. Save combined summary ───────────────────────────────────────────────
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary

# ── Model factory ──────────────────────────────────────────────────────────────
def _build_model(model_name: str, dataset, hp: dict, output_dir: Path):
    from models.mlp        import MLPWrapper
    from models.pysr_sr    import PySRWrapper
    from models.gplearn_sr import GPLearnWrapper
    from models.pysindy_sr import PySINDyWrapper
    from models.polynomial import PolynomialWrapper

    if model_name == "MLP_Standard":
        return MLPWrapper(
            input_dim  = dataset.X_train.shape[1],
            model_type = "standard",
            epochs     = int(hp.get("epochs", 100)),
            lr         = float(hp.get("lr", 1e-3)),
        )

    if model_name == "MLP_Sparse":
        return MLPWrapper(
            input_dim  = dataset.X_train.shape[1],
            model_type = "sparse",
            epochs     = int(hp.get("epochs", 1000)),
            lr         = float(hp.get("lr", 1e-3)),
            l1_alpha   = float(hp.get("l1_alpha", 1e-3)),
        )

    if model_name == "MLP_Dropout":
        return MLPWrapper(
            input_dim  = dataset.X_train.shape[1],
            model_type = "dropout",
            epochs     = int(hp.get("epochs", 500)),
            lr         = float(hp.get("lr", 1e-3)),
            mc_samples = int(hp.get("mc_samples", 100)),
        )

    if model_name == "Polynomial":
        return PolynomialWrapper(
            feature_names = dataset.feature_names,
            scaler_X      = dataset.scaler_X,
            scaler_y      = dataset.scaler_y,
            degree        = int(hp.get("degree", 3)),
        )

    if model_name == "PySR":
        model = PySRWrapper(
            feature_names = dataset.feature_names,
            niterations   = int(hp.get("niterations", 50)),
        )
        try:
            model.model.set_params(
                temp_equation_file = str(output_dir / "pysr_hall_of_fame.csv"),
                output_directory   = str(output_dir),
            )
        except Exception:
            pass
        try:
            if "populations" in hp:
                model.model.set_params(populations=int(hp["populations"]))
            if "maxsize" in hp:
                model.model.set_params(maxsize=int(hp["maxsize"]))
        except Exception:
            pass
        return model

    if model_name == "GPLearn":
        return GPLearnWrapper(
            feature_names   = dataset.feature_names,
            generations     = int(hp.get("generations", 30)),
            population_size = int(hp.get("population_size", 2000)),
        )

    if model_name == "PySINDy":
        model = PySINDyWrapper(
            feature_names = dataset.feature_names,
            degree        = int(hp.get("degree", 3)),
        )
        if "threshold" in hp:
            try:
                from pysindy.optimizers import STLSQ
                model.optimizer = STLSQ(threshold=float(hp["threshold"]))
            except Exception:
                pass
        return model

    if model_name == "QLattice":
        try:
            from models.qlattice_sr import QLatticeWrapper
        except ImportError:
            raise RuntimeError("QLattice (feyn) no está instalado en este entorno.")
        return QLatticeWrapper(
            feature_names  = dataset.feature_names,
            target_name    = hp.get("target_name", "y"),
            epochs         = int(hp.get("epochs", 15)),
            max_complexity = int(hp.get("max_complexity", 7)),
        )

    raise ValueError(f"Modelo desconocido: {model_name}")

# ── Output helpers (isolated) ──────────────────────────────────────────────────
def _model_dir(output_dir: Path, model_name: str) -> Path:
    d = output_dir / model_name
    d.mkdir(exist_ok=True)
    return d

def _save_txt(output_dir, model_name, dataset_name, mse, mae, equation):
    path = _model_dir(output_dir, model_name) / f"{dataset_name}_result.txt"
    with open(path, "w") as f:
        f.write(f"=== Model: {model_name} | Dataset: {dataset_name} ===\n\n")
        f.write(f"Equation / Structure:\n{equation}\n\n")
        f.write("Metrics:\n")
        f.write(f"Test MSE : {mse:.6e}\n")
        f.write(f"Test MAE : {mae:.6e}\n")

def _save_residuals(output_dir, model_name, dataset_name, y_true, y_pred, target_name):
    eps       = 1e-8
    abs_err   = np.abs(y_true - y_pred)
    rel_err   = abs_err / (np.abs(y_true) + eps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(y_true, abs_err, alpha=0.6, color="#38bdf8", s=10, rasterized=True)
    ax1.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    ax1.set_title(f"Error Absoluto vs {target_name}")
    ax1.set_xlabel(f"{target_name} real")
    ax1.set_ylabel(f"|{target_name} − pred|")
    ax1.set_facecolor("#0a0f1e")
    ax1.tick_params(colors="#94a3b8")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1e2d45")

    ax2.scatter(y_true, rel_err, alpha=0.6, color="#f87171", s=10)
    ax2.axhline(1, color="#64748b", linestyle="--", linewidth=1)
    ax2.set_title(f"Error Relativo vs {target_name}")
    ax2.set_xlabel(f"{target_name} real")
    ax2.set_ylabel("Error relativo")
    ax2.set_yscale("log")
    ax2.set_facecolor("#0a0f1e")
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e2d45")

    fig.patch.set_facecolor("#070a12")
    plt.suptitle(f"{model_name} — Residuos ({dataset_name})", color="#e2e8f0", fontsize=13)
    plt.tight_layout()

    out = _model_dir(output_dir, model_name) / f"{dataset_name}_residuals.png"
    plt.savefig(str(out), bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close()

def _save_loss_curve(output_dir, model_name, dataset_name, history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train", color="#38bdf8", linewidth=1.5)
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="Validacion", color="#818cf8", linewidth=1.5)
    ax.set_title(f"Convergencia: {model_name}", color="#e2e8f0")
    ax.set_xlabel("Epoca", color="#94a3b8")
    ax.set_ylabel("MSE", color="#94a3b8")
    ax.set_yscale("log")
    ax.legend(facecolor="#0a0f1e", edgecolor="#1e2d45", labelcolor="#e2e8f0")
    ax.set_facecolor("#0a0f1e")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d45")
    fig.patch.set_facecolor("#070a12")
    plt.tight_layout()

    out = _model_dir(output_dir, model_name) / f"{dataset_name}_loss.png"
    plt.savefig(str(out), bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close()