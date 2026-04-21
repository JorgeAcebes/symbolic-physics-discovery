# Importamos la clase Physical Dataset que nos permite tener el mismo split train-validation-test
# que tenemos en main, de tal manera que la optimización mediante optuna no produce data leakage.

import os
import time
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error

# Importamos tu inyector de datos
from data.loader import PhysicalDataset

# Importamos todos tus Wrappers
from models.mlp import MLPWrapper
from models.pysr_sr import PySRWrapper
from models.pysindy_sr import PySINDyWrapper
from models.qlattice_sr import QLatticeWrapper
from models.gplearn_sr import GPLearnWrapper
from models.polynomial import PolynomialWrapper

# ==========================================
# CONFIGURACIÓN DE LA BÚSQUEDA
# ==========================================
models_to_run = [
    "MLP_Standard",
    "MLP_Sparse",
    "MLP_Dropout",
    "Polynomial",
    "PySR",
    "GPLearn",
    "PySINDy",
    "QLattice"
]

datasets_info = [
    {"file": "oscillator_no_noise.csv", "target": "F"},
    # {"file": "kepler_no_noise.csv", "target": "T"},
]

def run_hyperparameter_search():
    # Rutas dinámicas
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(base_dir, "results", "optuna_hyperparams") # Guardamos el resultado en resutls/optuna_hiperparams
    os.makedirs(results_dir, exist_ok=True)
    
    for ds in datasets_info: 
        filepath = os.path.join(base_dir, "data", ds["file"]) 
        if not os.path.exists(filepath): # Verificamos existencia del archivo csv
            print(f"Archivo no encontrado: {filepath}")
            continue
            
        dataset_name = ds["file"].replace(".csv", "")
        print(f"\n{'='*50}\n OPTIMIZANDO DATASET: {dataset_name} \n{'='*50}") # Output en la terminal

        # 1. Carga de datos unificada (garantiza mismos splits que main.py) <--- Esta es la clave
        dataset = PhysicalDataset(filepath, target_col=ds["target"], scale=True)
        
        # Espacio latente (MLP, Polynomial)
        X_train, X_val, _, y_train, y_val, _ = dataset.get_latent_arrays()
        # Espacio físico (PySR, GPLearn, etc)
        X_train_phys, X_val_phys, _, y_train_phys, y_val_phys, _ = dataset.get_physical_arrays()
        # DataLoaders para MLPs
        train_loader, val_loader, _ = dataset.get_dataloaders()
        
        # Array objetivo de validación en espacio físico para comparar de forma justa
        y_val_true_phys = y_val_phys.flatten()

        for model_name in models_to_run:
            log_file_path = os.path.join(results_dir, f"optuna_{model_name}_{dataset_name}.txt") # txt donde guardaremos los hiperparámetros
            print(f"\n--- Iniciando Optuna para: {model_name} ---")

            def objective(trial):
                # ::::::::::::::::::::::::::::::::::::::::::::::::::::::
                # A) Definición del Espacio de Búsqueda según el Modelo
                # ::::::::::::::::::::::::::::::::::::::::::::::::::::::
                model_kwargs = {}
                
                if "MLP" in model_name:
                    # Tasa de aprendizaje: Es el parámetro más crítico de cualquier red neuronal.
                    model_kwargs['lr'] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                    model_kwargs['epochs'] = trial.suggest_int("epochs", 50, 300, step=50)
                    
                    if model_name == "MLP_Sparse":
                        # Controla cuántos pesos se fuerzan a cero (dispersión)
                        model_kwargs['l1_alpha'] = trial.suggest_float("l1_alpha", 1e-5, 1e-2, log=True)
                    elif model_name == "MLP_Dropout":
                        # Muestras para promediar la incertidumbre de Monte Carlo
                        model_kwargs['mc_samples'] = trial.suggest_int("mc_samples", 50, 200, step=50)
                        
                elif model_name == "Polynomial":
                    # Más allá del grado 5, los polinomios suelen sobreajustar (fenómeno de Runge)
                    model_kwargs['degree'] = trial.suggest_int("degree", 2, 5)
                    
                elif model_name == "GPLearn":
                    # Número de ciclos de evolución
                    model_kwargs['generations'] = trial.suggest_int("generations", 10, 50, step=10)
                    # Tamaño del ecosistema: más individuos = más diversidad, pero más lento
                    model_kwargs['population_size'] = trial.suggest_int("population_size", 500, 3000, step=500)
                    # Parsimonia: Castigo matemático a las ecuaciones largas para evitar sobreajuste
                    model_kwargs['parsimony_coefficient'] = trial.suggest_float("parsimony_coefficient", 0.001, 0.1, log=True)
                    
                    # 1. Optuna sugiere "pesos" relativos. Les damos rangos lógicos
                    # (Crossover suele ser el dominante, mutaciones suelen ser menores)
                    w_cross = trial.suggest_float("w_crossover", 0.4, 1.0)
                    w_sub = trial.suggest_float("w_subtree", 0.01, 0.3)
                    w_hoist = trial.suggest_float("w_hoist", 0.01, 0.3)
                    w_point = trial.suggest_float("w_point", 0.01, 0.3)
                    
                    # El remanente hasta 1.0 en GPLearn se considera "p_reproduction" 
                    # (clonación directa sin mutar). Le damos también un peso.
                    w_repro = trial.suggest_float("w_repro", 0.01, 0.2) 
                    
                    # 2. Calculamos la masa total
                    total_w = w_cross + w_sub + w_hoist + w_point + w_repro
                    
                    # 3. Normalizamos. Matemáticamente es imposible que la suma supere 1.0
                    model_kwargs['p_crossover'] = w_cross / total_w
                    model_kwargs['p_subtree_mutation'] = w_sub / total_w
                    model_kwargs['p_hoist_mutation'] = w_hoist / total_w
                    model_kwargs['p_point_mutation'] = w_point / total_w
                    
                elif model_name == "PySR":
                    model_kwargs['niterations'] = trial.suggest_int("niterations", 20, 80)
                    # Cantidad de sub-poblaciones aisladas que compiten
                    model_kwargs['populations'] = trial.suggest_int("populations", 10, 50)
                    # Individuos por población
                    model_kwargs['population_size'] = trial.suggest_int("population_size", 20, 100)
                    # Castigo a la complejidad. Log=True porque varía por órdenes de magnitud
                    model_kwargs['parsimony'] = trial.suggest_float("parsimony", 1e-5, 1e-1, log=True)
                    # Tamaño máximo del árbol de la ecuación (nodos)
                    model_kwargs['maxsize'] = trial.suggest_int("maxsize", 10, 25)
                    
                elif model_name == "PySINDy":
                    # Grado de la biblioteca de polinomios base
                    model_kwargs['degree'] = trial.suggest_int("degree", 2, 4)
                    # Umbral de corte (STLSQ): Elimina términos físicos que tengan coeficientes menores a esto. CRÍTICO para el ruido.
                    model_kwargs['threshold'] = trial.suggest_float("threshold", 0.001, 0.5, log=True)
                    # Regularización L2 interna para suavizar matrices mal condicionadas
                    model_kwargs['alpha'] = trial.suggest_float("alpha", 1e-4, 1.0, log=True)

                elif model_name == "QLattice":
                    model_kwargs['epochs'] = trial.suggest_int("epochs", 5, 30)
                    # Define la longitud y complejidad máxima de los grafos generados en Feyn
                    model_kwargs['max_complexity'] = trial.suggest_int("max_complexity", 4, 10)


                # ::::::::::::::::::::::::::::::::::::::::::::::::::::::
                # B) Simulación Monte Carlo (Varios seeds para robustez)
                # ::::::::::::::::::::::::::::::::::::::::::::::::::::::
                num_seeds = 3 if "MLP" in model_name else 1 # 3 seeds en MLPs, 1 en el resto (ya implementan competiciones internas)
                losses = []

                for seed in range(num_seeds):
                    # Inyección dinámica de los wrappers con los hiperparámetros de Optuna
                    if model_name == "MLP_Standard":
                        model = MLPWrapper(input_dim=X_train.shape[1], model_type='standard', **model_kwargs)
                    elif model_name == "MLP_Sparse":
                        model = MLPWrapper(input_dim=X_train.shape[1], model_type='sparse', **model_kwargs)
                    elif model_name == "MLP_Dropout":
                        model = MLPWrapper(input_dim=X_train.shape[1], model_type='dropout', **model_kwargs)
                    elif model_name == "Polynomial":
                        model = PolynomialWrapper(feature_names=dataset.feature_names, scaler_X=dataset.scaler_X, scaler_y=dataset.scaler_y, **model_kwargs)
                    elif model_name == "PySR":
                        model = PySRWrapper(feature_names=dataset.feature_names,  **model_kwargs) 
                    elif model_name == "GPLearn":
                        model = GPLearnWrapper(feature_names=dataset.feature_names, **model_kwargs)
                    elif model_name == "PySINDy":
                        model = PySINDyWrapper(feature_names=dataset.feature_names,  **model_kwargs) 
                    elif model_name == "QLattice":
                        model = QLatticeWrapper(feature_names=dataset.feature_names, target_name=ds["target"], **model_kwargs)


                    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::
                    # C) Entrenamiento y Predicción (Bifurcación latente/físico)
                    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::
                    if "MLP" in model_name:
                        model.fit(train_loader, val_loader)
                        # Predecir en espacio latente y des-escalar a físico
                        y_pred_lat = model.predict(X_val)
                        y_pred_phys = dataset.scaler_y.inverse_transform(y_pred_lat).flatten()
                        
                    elif model_name == "Polynomial":
                        model.fit(X_train, y_train)
                        y_pred_lat = model.predict(X_val)
                        y_pred_phys = dataset.scaler_y.inverse_transform(y_pred_lat).flatten()
                        
                    else: # Regresadores Simbólicos (espacio físico)
                        if model_name in ["PySR", "QLattice"]:
                            model.fit(X_train_phys, y_train_phys, X_val=X_val_phys, y_val=y_val_phys)
                        else:
                            model.fit(X_train_phys, y_train_phys)
                            
                        y_pred_phys = model.predict(X_val_phys).flatten()

                    # Cálculo del Error en espacio real FÍSICO
                    mse_val = mean_squared_error(y_val_true_phys, y_pred_phys)
                    losses.append(mse_val)

                # Optuna optimiza el MSE medio de los distintos seeds
                mean_loss = np.float64(np.mean(losses))
                std_loss = float(np.std(losses))
                trial.set_user_attr("loss_std", std_loss)
                alpha = 1.0 # Peso de penalización a la std
                return mean_loss + alpha * std_loss
            
            # ::::::::::::::::::::::::::::::::::::::::::::::::::::::
            # D) Ejecución del Study de Optuna
            # ::::::::::::::::::::::::::::::::::::::::::::::::::::::
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=42))

            
            # Número de iteraciones por modelo. (Bájalo para PySR/QLattice si tardan mucho)
            n_trials = 10 if "MLP" in model_name else 5 
            study.optimize(objective, n_trials=n_trials)

            # ::::::::::::::::::::::::::::::::::::::::::::::::::::::
            # E) Guardado de los Mejores Resultados en TXT independiente
            # ::::::::::::::::::::::::::::::::::::::::::::::::::::::
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(f"=== RESULTADOS OPTIMIZACIÓN: {model_name} en {dataset_name} ===\n\n")
                f.write(f"Mejor Trial: #{study.best_trial.number}\n")
                f.write(f"MSE Medio (Espacio Físico): {study.best_value:.4e}\n")
                f.write(f"Desviación Estándar (MSE):  {study.best_trial.user_attrs['loss_std']:.4e}\n\n")
                
                f.write("Mejores Hiperparámetros:\n")
                for key, value in study.best_trial.params.items():
                    f.write(f"  - {key}: {value}\n")

            print(f"[{model_name}] Optimizado. Resultados guardados en: {log_file_path}")

if __name__ == "__main__":
    run_hyperparameter_search()