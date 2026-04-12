# %%
import os
from data.loader import PhysicalDataset
from models.mlp import MLPWrapper
from models.pysindy_sr import PySINDyWrapper
from models.qlattice_sr import QLatticeWrapper
from models.gplearn_sr import GPLearnWrapper
from models.polynomial import PolynomialWrapper
from utils.metrics import evaluate_physical_space
from utils.io import save_experiment_results, plot_residual_analysis

def run_all_experiments():
    datasets_info = [
        # {"file": "oscillator_no_noise.csv", "target": "F"},
        # {"file": "kepler_no_noise.csv", "target": "T"},
        {"file": "coulomb_no_noise.csv", "target": "F"},
        {"file": "ideal_gas_low_noise.csv", "target": "P"}
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
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_arrays()
        
        # Extracción de X sin escalar para recuperación analítica en Polynomial
        X_train_unscaled = dataset.scaler_X.inverse_transform(X_train) if dataset.scaler_X else X_train
        X_test_unscaled = dataset.scaler_X.inverse_transform(X_test) if dataset.scaler_X else X_test

        models = {
            "MLP_Standard": MLPWrapper(input_dim=X_train.shape[1], model_type='standard', epochs=100),
            "MLP_Sparse":   MLPWrapper(input_dim=X_train.shape[1], model_type='sparse', epochs=1000, l1_alpha=1e-3),
            "MLP_Dropout":  MLPWrapper(input_dim=X_train.shape[1], model_type='dropout', epochs=500),
            "Polynomial":   PolynomialWrapper(degree=3, feature_names=dataset.feature_names, scaler_y=dataset.scaler_y),
            "GPLearn":      GPLearnWrapper(generations=30),
            "PySINDy":      PySINDyWrapper(feature_names=dataset.feature_names),
            "QLattice":     QLatticeWrapper(feature_names=dataset.feature_names, target_name=ds["target"], epochs=15)
        }
        for model_name, model in models.items():
            print(f"\n--- Iniciando entrenamiento: {model_name} ---\n")
            
            # Entrenamiento con gestión de métricas diferenciada
            if "MLP" in model_name:
                train_loader, val_loader, _ = dataset.get_dataloaders()
                model.fit(train_loader, val_loader)
                mse, mae = evaluate_physical_space(model, X_test, y_test, dataset.scaler_y)
                
            elif model_name == "Polynomial":
                model.fit(X_train_unscaled, y_train)
                mse, mae = evaluate_physical_space(model, X_test_unscaled, y_test, dataset.scaler_y)
                
            elif model_name in ["GPLearn", "QLattice"]:
                # Inyección de validación para registro de historial
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                mse, mae = evaluate_physical_space(model, X_test, y_test, dataset.scaler_y)
                
            else: # PySINDy y otros Regresores Simbólicos básicos
                model.fit(X_train, y_train)
                mse, mae = evaluate_physical_space(model, X_test, y_test, dataset.scaler_y)

            
            # Aseguramos la entrada correcta según si es Polynomial (usa unscaled en tu código) o el resto
            X_eval = X_test_unscaled if model_name == "Polynomial" else X_test
            
            # Predicción cruda del modelo
            y_pred_raw = model.predict(X_eval)
            
            # Proyección al espacio físico si hay un scaler aplicado a Y
            if dataset.scaler_y:
                y_pred_physical = dataset.scaler_y.inverse_transform(y_pred_raw).flatten()
                y_test_physical = dataset.scaler_y.inverse_transform(y_test).flatten()
            else:
                y_pred_physical = y_pred_raw.flatten()
                y_test_physical = y_test.flatten()
                
            # Generación de las gráficas
            results_base = os.path.abspath(os.path.join(base_dir, "results"))
            plot_residual_analysis(y_test_physical, y_pred_physical, model_name, dataset_name, results_base)

            # Salida de resultados y persistencia
            print(f"Expresión descubierta: {model.equation}")
            print(f"Métricas en Espacio Físico -> MSE: {mse:.4e} | MAE: {mae:.4e}")

            metrics = {"mse": mse, "mae": mae}
            history = getattr(model, "history", None)
            save_experiment_results(model_name, dataset_name, metrics, model.equation, history)

if __name__ == "__main__":
    run_all_experiments()