# %%
import pysr
import os
from data.loader import PhysicalDataset
from models.mlp import MLPWrapper
from models.pysr_sr import PySRWrapper
from models.pysindy_sr import PySINDyWrapper
from models.qlattice_sr import QLatticeWrapper
from models.gplearn_sr import GPLearnWrapper
from models.polynomial import PolynomialWrapper
from utils.metrics import evaluate_physical_space
from utils.io import save_experiment_results, plot_residual_analysis

def run_all_experiments():
    datasets_info = [
        {"file": "oscillator_no_noise.csv", "target": "F"},
        # {"file": "kepler_no_noise.csv", "target": "T"},
        # {"file": "coulomb_no_noise.csv", "target": "F"},
        {"file": "ideal_gas_no_noise.csv", "target": "P"}
    ]

    for ds in datasets_info:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        filepath = os.path.join(base_dir, "data", ds["file"])
        if not os.path.exists(filepath):
            print(f"Archivo no encontrado: {filepath}")
            continue
            
        dataset_name = ds["file"].replace(".csv", "")
        print(f"\n" + "="*50)
        print(f" DATASET: {dataset_name} ")
        print("="*50)

        # Carga de datos
        dataset = PhysicalDataset(filepath, target_col=ds["target"], scale=True)
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_latent_arrays()        

        X_train_phys, X_val_phys, X_test_phys, y_train_phys, y_val_phys, y_test_phys = dataset.get_physical_arrays()

        models = {
            "MLP_Standard": MLPWrapper(input_dim=X_train.shape[1], model_type='standard', epochs=100),
            "MLP_Sparse":   MLPWrapper(input_dim=X_train.shape[1], model_type='sparse', epochs=1000, l1_alpha=1e-3),
            "MLP_Dropout":  MLPWrapper(input_dim=X_train.shape[1], model_type='dropout', epochs=500),
            "Polynomial":   PolynomialWrapper(degree=3, feature_names=dataset.feature_names, scaler_X=dataset.scaler_X, scaler_y=dataset.scaler_y),
            "PySR":         PySRWrapper(feature_names=dataset.feature_names),
            "GPLearn":      GPLearnWrapper(generations=30),
            "PySINDy":      PySINDyWrapper(feature_names=dataset.feature_names),
            "QLattice":     QLatticeWrapper(feature_names=dataset.feature_names, target_name=ds["target"], epochs=15)
        }

        for model_name, model in models.items():
            print(f"\n--- Iniciando entrenamiento: {model_name} ---\n")
            
            if "MLP" in model_name:
                # El gradiente descendente exige el hipercubo normalizado
                train_loader, val_loader, _ = dataset.get_dataloaders()
                model.fit(train_loader, val_loader)
                
                y_pred_physical = dataset.scaler_y.inverse_transform(model.predict(X_test)).flatten()
                y_test_physical = y_test_phys.flatten()
                mse, mae = evaluate_physical_space(model, X_test, y_test, dataset.scaler_y)
                
            elif model_name == "Polynomial":
                # El Wrapper extrae la ecuación retro-proyectando algebraicamente.
                # REQUIERE entrenamiento estricto en el espacio escalado.
                model.fit(X_train, y_train)
                
                y_pred_physical = dataset.scaler_y.inverse_transform(model.predict(X_test)).flatten()
                y_test_physical = y_test_phys.flatten()
                mse, mae = evaluate_physical_space(model, X_test, y_test, dataset.scaler_y)
                
            else:
                # Regresión Simbólica: Optimización directa sobre la variedad física
                if model_name in ["PySR", "QLattice"]:
                    model.fit(X_train_phys, y_train_phys, X_val=X_val_phys, y_val=y_val_phys)
                else:
                    model.fit(X_train_phys, y_train_phys)
                
                y_pred_physical = model.predict(X_test_phys).flatten()
                y_test_physical = y_test_phys.flatten()
                
                # Evaluación directa asumiendo identidad en la métrica
                mse, mae = evaluate_physical_space(model, X_test_phys, y_test_phys)

            # Generación de gráficas
            results_base = os.path.abspath(os.path.join(base_dir, "results"))
            plot_residual_analysis(y_test_physical, y_pred_physical, model_name, dataset_name, results_base)

            # Persistencia de resultados
            print(f"Expresión descubierta: {model.equation}")
            print(f"Métricas en Espacio Físico -> MSE: {mse:.4e} | MAE: {mae:.4e}")

            metrics = {"mse": float(mse), "mae": float(mae)}
            history = getattr(model, "history", None)
            save_experiment_results(model_name, dataset_name, metrics, model.equation, history)
            
if __name__ == "__main__":
    run_all_experiments()