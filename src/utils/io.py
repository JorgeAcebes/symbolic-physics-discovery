#%%
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
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


    
    plt.suptitle(f"{safe_model} - Análisis de Residuos ({safe_dataset})", fontsize=14)    
    plt.tight_layout()
    
    save_path = os.path.join(exp_dir, f"{dataset_name}_residuals.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()





def report_all_models():
    print('~'*50)
    print(f'Iniciando reporte')
    
    results_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results")))
    output_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "all_models")))
    output_dir.mkdir(parents=True, exist_ok=True)

    models = set()
    experiments = set()
    
    # Estructura: dict[model][experiment] = path
    loss_imgs = {}
    res_imgs = {}
    
    # Estructura: dict[experiment][model] = path
    txt_files = {}

    # Patrón para extraer el experimento, el tipo (loss/residuals/result) y la extensión
    pattern = re.compile(r'^(.*)_(loss|residuals|result)\.(png|txt)$')

    # Exploración recursiva y clasificación
    for model_dir in [d for d in results_dir.iterdir() if d.is_dir()]:
        model = model_dir.name
        if model != 'all_models':
            models.add(model)
            
            for file_path in model_dir.rglob('*.*'):
                match = pattern.match(file_path.name)
                if match:
                    exp, ftype, ext = match.groups()
                    experiments.add(exp)
                    
                    if ftype == 'loss' and ext == 'png':
                        loss_imgs.setdefault(model, {})[exp] = file_path
                    elif ftype == 'residuals' and ext == 'png':
                        res_imgs.setdefault(model, {})[exp] = file_path
                    elif ftype == 'result' and ext == 'txt':
                        txt_files.setdefault(exp, {})[model] = file_path

    models = sorted(list(models))
    experiments = sorted(list(experiments))

    def create_gallery(data_dict, out_filename):
        n_rows = len(models)
        n_cols = len(experiments)
        if n_rows == 0 or n_cols == 0:
            return
        
        # Dimensionado dinámico: 4 pulgadas por columna, 3 por fila
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        
        # Garantizar que axes sea un array 2D independientemente del tamaño
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        for i, model in enumerate(models):
            for j, exp in enumerate(experiments):
                ax = axes[i, j]
                ax.axis('off')
                
                img_path = data_dict.get(model, {}).get(exp)
                if img_path and img_path.exists():
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                
                # Encabezados de columna (Experimentos)
                if i == 0:
                    ax.set_title(exp, fontsize=12, fontweight='bold')
                
                # Encabezados de fila (Modelos)
                if j == 0:
                    ax.text(-0.1, 0.5, model, rotation=90, va='center', ha='right', 
                            transform=ax.transAxes, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / out_filename, dpi=300, bbox_inches='tight')
        plt.close()

    # Generar las galerías de imágenes
    create_gallery(loss_imgs, "gallery_loss.png")
    create_gallery(res_imgs, "gallery_residuals.png")

    print(f'Galerías Generadas en {output_dir}')

    # Generar el archivo de texto unificado
    with open(output_dir / 'combined_results.txt', 'w', encoding='utf-8') as outfile:
        for exp in experiments:
            outfile.write(f"{'='*20}\n")
            outfile.write(f"EXPERIMENT: {exp}\n")
            outfile.write(f"{'='*20}\n\n")
            
            for model in models:
                txt_path = txt_files.get(exp, {}).get(model)
                if txt_path and txt_path.exists():
                    with open(txt_path, 'r', encoding='utf-8') as infile:
                        # Si el archivo ya trae la cabecera '=== Model...', simplemente se adjunta.
                        content = infile.read().strip()
                        outfile.write(content + "\n\n")
            outfile.write("\n")
    
    print(f'Resumen de loss y fórmulas generadas en {output_dir}')
    print('~'*50)


