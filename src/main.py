# %% <-- Útil para correrlo en IPython
# Done by Jorge Acebes, Andrés López and Lorenzo Ji

# Imports del instrumental analítico + envoltorios de los modelos
import pysr #<--- No tocar. Debe ser llamado lo primero de todo 
import os
from data.loader import PhysicalDataset
from models.mlp import MLPWrapper
from models.pysr_sr import PySRWrapper
from models.pysindy_sr import PySINDyWrapper
from models.qlattice_sr import QLatticeWrapper
from models.gplearn_sr import GPLearnWrapper
from models.polynomial import PolynomialWrapper
from utils.metrics import evaluate_physical_space
from utils.io import save_experiment_results, plot_residual_analysis, report_all_models

# Conjunto de modelos con los que se correrá

models_to_run = [
    "MLP_Standard",
    # "MLP_Sparse",
    # "MLP_Dropout",
    # "Polynomial",
    # "PySR",
    # "GPLearn",
    # "PySINDy",
    # "QLattice" 
    ]

if len(models_to_run) == 0:
    raise ValueError("Debes escoger al menos 1 modelo en 'models_to_run'")
# Conjunto de archivos que se procesarán. Deben estar ubicados en la carpeta data 
def run_all_experiments():
    datasets_info = [
        {"file": "oscillator_no_noise.csv", "target": "F"},
        # {"file": "kepler_no_noise.csv", "target": "T"},
        # {"file": "coulomb_no_noise.csv", "target": "F"},
        # {"file": "ideal_gas_no_noise.csv", "target": "P"}
    ]


    for ds in datasets_info:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Establecemos la ruta de la raíz del proyecto
        filepath = os.path.join(base_dir, "data", ds["file"]) # Creamos la ruta carpeta_raíz/data/nombre_del_archivo.csv

        # Cortafuegos de seguridad. Si no se encuentra el archivo en esa ruta, lo mostramos en la terminal y pasamos al siguiente
        if not os.path.exists(filepath):
            print(f"Archivo no encontrado: {filepath}")
            continue
            
        # I/O En la terminal
        dataset_name = ds["file"].replace(".csv", "")
        print(f"\n" + "="*50)
        print(f" DATASET: {dataset_name} ")
        print("="*50)

        # Carga de datos mediante la clase PhysicalDataset de loader.py
        dataset = PhysicalDataset(filepath, target_col=ds["target"], scale=True)

        # Obtención del espacio "latente" (abuso de nomeclatura, sería más correcto z-score o N(0,1)) <-- Los usaremos para MLP
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_latent_arrays()        

        # Obtención de los datos físicos
        X_train_phys, X_val_phys, X_test_phys, y_train_phys, y_val_phys, y_test_phys = dataset.get_physical_arrays()

        
        # Inyección dinámica de los modelos. Se instancian dentro del bucle para garantizar que parámetros como input_dim o scaler_X no se hereden del experimento previo.
        # Se decide qué inputs tiene cada modelo (si espacio z-score, para MLP y Polynomial, o si es el espacio físic, para SR), así como otros inputs
        models = {
            "MLP_Standard": MLPWrapper(input_dim=X_train.shape[1], model_type='standard', epochs=100),
            "MLP_Sparse":   MLPWrapper(input_dim=X_train.shape[1], model_type='sparse', epochs=1000, l1_alpha=1e-3),
            "MLP_Dropout":  MLPWrapper(input_dim=X_train.shape[1], model_type='dropout', epochs=500, mc_samples=100),
            "Polynomial":   PolynomialWrapper(feature_names=dataset.feature_names, scaler_X=dataset.scaler_X, scaler_y=dataset.scaler_y, degree=3),
            "PySR":         PySRWrapper(feature_names=dataset.feature_names),
            "GPLearn":      GPLearnWrapper(feature_names=dataset.feature_names, generations=30),
            "PySINDy":      PySINDyWrapper(feature_names=dataset.feature_names),
            "QLattice":     QLatticeWrapper(feature_names=dataset.feature_names, target_name=ds["target"], epochs=15)
        }
        

        for model_name, model in models.items():
            if model_name in models_to_run:
                print(f"\n--- Iniciando entrenamiento: {model_name} ---\n")
                
                # Bifurcación según topología.
                if "MLP" in model_name: 
                    # Los MLP requieren de "alimentación" por lotes. Llamamos a get_dataloaders
                    train_loader, val_loader, _ = dataset.get_dataloaders() # Esto se realizó de manera correcta en el loader.py de tal manera que estamos en "espacio latente"
                    model.fit(train_loader, val_loader) # Realizamos el fit en el espacio latente, donde trabajan bien los MLPs.
                    
                    # Predicción en el espacio latente y proyección a la variedad física (espacio real)
                    y_pred_physical = dataset.scaler_y.inverse_transform(model.predict(X_test)).flatten()
                    y_test_physical = y_test_phys.flatten()
                    
                elif model_name == "Polynomial":
                    model.fit(X_train, y_train) # De manera similar, trabajamos en el espacio latente. En este caso no requiere de lotes.
                    y_pred_physical = dataset.scaler_y.inverse_transform(model.predict(X_test)).flatten() # Revertimos la transformación al espacio físico.
                    y_test_physical = y_test_phys.flatten()
                    
                else:
                    # Regresión Simbólica: Ajuste y evaluación estricta en el hiperespacio físico original
                    if model_name in ["PySR", "QLattice"]:
                        # PySR y QLattice están preparados para tomar X_val
                        model.fit(X_train_phys, y_train_phys, X_val=X_val_phys, y_val=y_val_phys)
                    else:
                        model.fit(X_train_phys, y_train_phys)
                    
                    y_pred_physical = model.predict(X_test_phys).flatten()
                    y_test_physical = y_test_phys.flatten()

                # Empleo de flatten() para garantizar vector unidimensional en R^n, no matrices.
                
                
                # Cálculo de la distancia métrica en el espacio FÍSICO
                mse, mae = evaluate_physical_space(y_test_physical, y_pred_physical)


                # Generación de gráficas
                results_base = os.path.abspath(os.path.join(base_dir, "results")) # Ruta a la carpeta de resultados
                plot_residual_analysis(y_test_physical, y_pred_physical, model_name, dataset_name, results_base, target_name=ds["target"]) # Gráfica de residuos (con el test FÍSICO)
                
                # I/O de resultados:
                print(f"Expresión descubierta: {model.equation}")
                print(f"Métricas en Espacio Físico -> MSE: {mse:.4e} | MAE: {mae:.4e}")


                metrics = {"mse": float(mse), "mae": float(mae)}
                history = getattr(model, "history", None) # En caso de que el modelo tenga el atributo "history", la obtenemos. Esto nos permitirá graficar el loss.
                save_experiment_results(model_name, dataset_name, metrics, model.equation, history) # Guardamos el mae y mse obtenido,
                # expresión (si es posible) y ploteo de loss (si es posible)
                
if __name__ == "__main__":
    run_all_experiments()
    report_all_models()
# %%
