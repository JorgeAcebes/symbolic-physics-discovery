import os
import matplotlib.pyplot as plt

def save_experiment_results(model_name, dataset_name, metrics, equation, history=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", model_name))
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Guardar informe en texto plano
    txt_path = os.path.join(base_dir, f"{dataset_name}_result.txt")
    with open(txt_path, "w") as f:
        f.write(f"=== Modelo: {model_name} | Dataset: {dataset_name} ===\n\n")
        f.write(f"Ecuación / Estructura:\n{equation}\n\n")
        f.write("Métricas (Espacio Físico):\n")
        f.write(f"Test MSE: {metrics['mse']:.6e}\n")
        f.write(f"Test MAE: {metrics['mae']:.6e}\n")

    # 2. Guardar curva de convergencia si existe
    if history and len(history.get("train_loss", [])) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="Train Loss")
        if "val_loss" in history and history["val_loss"]:
            plt.plot(history["val_loss"], label="Validation Loss")
            
        plt.title(f"Convergencia: {model_name} ({dataset_name})")
        plt.xlabel("Épocas / Generaciones")
        plt.ylabel("MSE")
        plt.yscale("log") # Escala logarítmica para dinámicas físicas
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        
        plot_path = os.path.join(base_dir, f"{dataset_name}_loss.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()