import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.utils import set_plot_style

set_plot_style(for_paper=False) # To be set on 'True' when obtaining the graphs for paper or presentation

def save_experiment_results(model_name, dataset_name, metrics, equation, history=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", model_name))
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Guardar informe en texto plano
    txt_path = os.path.join(base_dir, f"{dataset_name}_result.txt")
    with open(txt_path, "w") as f:
        f.write(f"=== Model: {model_name} | Dataset: {dataset_name} ===\n\n")
        f.write(f"Equation / Structure:\n{equation}\n\n")
        f.write("Metrics:\n")
        f.write(f"Test MSE: {metrics['mse']:.6e}\n")
        f.write(f"Test MAE: {metrics['mae']:.6e}\n")

    # 2. Guardar curva de convergencia si existe
    if history and len(history.get("train_loss", [])) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="Train Loss")
        if "val_loss" in history and history["val_loss"]:
            plt.plot(history["val_loss"], label="Validation Loss")
                    
        safe_model = model_name.replace('_', r'\_')
        safe_dataset = dataset_name.replace('_', r'\_')

        plt.title(f"Convergence: {safe_model} ({safe_dataset})")
        plt.xlabel("Epochs / Generations")
        plt.ylabel("MSE")
        plt.yscale("log") # Escala logarítmica para dinámicas físicas
        plt.legend()
        
        plot_path = os.path.join(base_dir, f"{dataset_name}_loss.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()


def plot_residual_analysis(y_true, y_pred, model_name, dataset_name, results_dir, target_name=None):
    """Genera diagramas de dispersión del error absoluto y relativo en escala logarítmica."""
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    epsilon = 1e-8
    abs_error = np.abs(y_true - y_pred)
    rel_error = abs_error / (np.abs(y_true) + epsilon) # Evitamos divergencias
    
    exp_dir = os.path.join(results_dir, model_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Análisis de Error Absoluto
    ax1.scatter(y_true, abs_error, alpha=0.6, color='#003B5C', s=15, rasterized=True)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5) # Baseline (LÍNEA DE ERROR ABSOLUTO NULO)
    ax1.set_title(r"\textbf{Error Absoluto vs Magnitud}" if target_name is None else rf"\textbf{{Error Absoluto vs ${target_name}$}}")
    ax1.set_xlabel(r"$y_{\text{real}}$" if target_name is None else rf"${target_name}$")
    ax1.set_ylabel(r"$|y_{\text{real}} - y_{\text{pred}}|$" if target_name is None else rf"$|{target_name} - \hat{{{target_name}}}|$")
    
    # Análisis de Error Relativo (Escala Logarítmica)
    ax2.scatter(y_true, rel_error, alpha=0.6, color='#C60C30', s=15)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5) # Baseline (LÍNEA DE ERROR RELATIVO EN ESPACIO LOG NULO)
    ax2.set_title(r"\textbf{Error Relativo vs Magnitud}" if target_name is None else rf"\textbf{{Error Relativo vs ${target_name}$}}")
    ax2.set_xlabel(r"$y_{\text{real}}$" if target_name is None else rf"${target_name}$")
    ax2.set_ylabel(r"$\epsilon_{\text{rel}}$" if target_name is None else rf"$\epsilon_{{\text{{rel}}}} = \frac{{|{target_name} - \hat{{{target_name}}}|}}{{|{target_name}|}}$")
    ax2.set_yscale("log")
    safe_model = model_name.replace('_', r'\_')
    safe_dataset = dataset_name.replace('_', r'\_')

    # TODO: Hay que mejorar la salida del título. Actualmente es funcional pero poco profesional
    
    plt.suptitle(f"{safe_model} - Análisis de Residuos ({safe_dataset})", fontsize=14)    
    plt.tight_layout()
    
    save_path = os.path.join(exp_dir, f"{dataset_name}_residuals.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()