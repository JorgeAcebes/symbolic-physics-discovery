import os
from data.loader import PhysicalDataset
from models.mlp import MLPWrapper
from models.pysindy_sr import PySINDyWrapper
from models.qlattice_sr import QLatticeWrapper
from models.gplearn_sr import GPLearnWrapper
from models.polynomial import PolynomialWrapper
from utils.metrics import evaluate_physical_space
from utils.io import save_experiment_results

def run_all_experiments():
    datasets_info = [
        {"file": "oscillator_no_noise.csv", "target": "F"},
        {"file": "kepler_no_noise.csv", "target": "T"},
        {"file": "coulomb_no_noise.csv", "target": "F"},
        {"file": "ideal_gas_no_noise.csv", "target": "P"}
    ]

    for ds in datasets_info:
        filepath = os.path.join("data", ds["file"])
        if not os.path.exists(filepath):
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
            "MLP": MLPWrapper(input_dim=X_train.shape[1], epochs=50),
            "Polynomial": PolynomialWrapper(degree=3, feature_names=dataset.feature_names, scaler_y=dataset.scaler_y),
            "GPLearn": GPLearnWrapper(generations=30),
            "PySINDy": PySINDyWrapper(feature_names=dataset.feature_names),
            # "QLattice": QLatticeWrapper(feature_names=dataset.feature_names, target_name=ds["target"])
        }

        for model_name, model in models.items():
            print(f"\n--- Ejecutando: {model_name} ---")
            
            # Entrenamiento diferenciado según arquitectura
            if model_name == "MLP":
                train_loader, val_loader, _ = dataset.get_dataloaders()
                model.fit(train_loader, val_loader)
                mse, mae = evaluate_physical_space(model, X_test, y_test, dataset.scaler_y)
                
            elif model_name == "Polynomial":
                # Entrena con X en espacio físico y retornará y escalado
                model.fit(X_train_unscaled, y_train)
                mse, mae = evaluate_physical_space(model, X_test_unscaled, y_test, dataset.scaler_y)
                
            else:
                model.fit(X_train, y_train)
                mse, mae = evaluate_physical_space(model, X_test, y_test, dataset.scaler_y)

            # Consola
            print(f"Ecuación: {model.equation}")
            print(f"Test MSE: {mse:.4e} | Test MAE: {mae:.4e}")

            # Exportación a disco
            metrics = {"mse": mse, "mae": mae}
            save_experiment_results(model_name, dataset_name, metrics, model.equation, model.history)

if __name__ == "__main__":
    run_all_experiments()