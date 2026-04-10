# %%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Regresores Simbólicos
import feyn # QLattice
from feyn.losses import squared_error
from gplearn.genetic import SymbolicRegressor
import pysindy as ps
from gplearn.functions import make_function

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../..", "data"))
RESULTS_BASE = os.path.abspath(os.path.join(BASE_DIR, "../../results/testing_other_sr"))

RES_QLATTICE = os.path.join(RESULTS_BASE, "testing_qlattice")
RES_GPLEARN = os.path.join(RESULTS_BASE, "testing_gplearn")
RES_PYSINDY = os.path.join(RESULTS_BASE, "testing_pysindy")


# ================================================================================================
# This should be temporary. We should implemet a confing.json in where we would choose what models

qlattice = 0
gplearn = 1
sindy = 1

# Only dataset containing these words will be used. Using OR, not AND.
names = ['ideal_gas_low_noise'] 

# TODO: Implement logic so that when "all" is written, all datasets are used
# =============================S===================================================================


for path in [RES_QLATTICE, RES_GPLEARN, RES_PYSINDY]:
    os.makedirs(path, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Modelos de Regresión Simbólica
# -------------------------------------------------------------------------

def plot_losses(train_losses, val_losses, epoch, loss_type = None, filename=None):
    x0 = list(range(1, epoch+1))
    plt.figure(figsize=(10, 5))
    plt.plot(x0, train_losses, label='Train loss')
    plt.plot(x0, val_losses, label='Validation loss')
    plt.title('Model loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss' if not loss_type else f'Loss ({loss_type})')
    plt.legend()
    if filename:
        plt.savefig(filename)
    plt.show()

# Lo anterior era un modelo bastante poco modular. Quiero poder modificar a mi gusto ciertas cosas. Esto es un intento:
def run_qlattice(df, target_col, dataset_name, filename=None):
    random_state = 42
    kind = 'regression'
    loss_function = 'squared_error' # MSE (realmente no hacemos la media pero para optimizar es idéntico)
    output_name = target_col
    n_epochs = 10 # 10 épocas (default)
    threads = 4 
    criterion = 'bic' # Bayesian Information Criterion para penalizar sobreajuste
    max_complexity = 7  # Forzamos parsimonia para física 


    feyn.validate_data(df, kind, output_name) # Valida el DataFrame
    train, val, test = feyn.tools.split(df, ratio=[0.6, 0.2, 0.2], random_state=random_state)

    print(type(train))

    ql = feyn.QLattice(random_seed=random_state)

    train_losses = []
    val_losses = []
    models = []

    for epoch in range(1, n_epochs+1):
        models += ql.sample_models(train, output_name, kind, max_complexity=max_complexity)

        models = feyn.fit_models(models, train, threads=threads, loss_function=loss_function, criterion=criterion)
        models = feyn.prune_models(models)

        # Append the latest loss value of the top model and display the loss with our function
        train_losses.append(models[0].loss_value)

        val_loss = np.mean(squared_error(val[output_name], models[0].predict(val)))
        val_losses.append(val_loss)


        # Note: because we use IPython.display (update_display=True) in show_model, the order here is important.

        feyn.show_model(
            models[0],
            # Just a simple label. Auto_run is more sophisticated.
            label=f"Epoch {epoch}/{n_epochs}.",
            update_display=True,
        )

        ql.update(models)

    models = feyn.get_diverse_models(models)
    
    out_path = os.path.join(RES_QLATTICE, f"{dataset_name}_{filename}_loss.png")
    # Display the final model and the loss graph
    plot_losses(train_losses, val_losses, epoch, loss_type='MSE', filename=out_path)
    best_model = models[0]  # El índice 0 contiene el modelo topológico óptimo
    best_model.show(update_display=True)
    best_model.plot(train, filename=os.path.join(RES_QLATTICE, f"{dataset_name}_plot.html"))

    best_expr = str(best_model.sympify(signif=4))
    
    # Guardar resultados
    out_path = os.path.join(RES_QLATTICE, f"{dataset_name}_result_manual.txt")
    with open(out_path, "w") as f:
        f.write(f"Equation obtained for {dataset_name}:\n")
        f.write(best_expr + "\n")
    
    print(f"[QLattice] {dataset_name}: {best_expr}")




# %%


#---- Working on it ------

##Creación de función exponencial (gplearn no la tiene definida por defecto)
# def _protected_exp(x):
#     with np.errstate(over='ignore'):
#         return np.where(np.abs(x) < 100, np.exp(x), 0.0)

# exp_func = make_function(function=_protected_exp, name='exp', arity=1)

def run_gplearn(X, y, dataset_name):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SymbolicRegressor(
        population_size=2000,
        generations=1,          # Controlamos el avance manualmente
        warm_start=True,        # Mantiene la población para la siguiente iteración
        function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos'),
        metric='mse',
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        n_jobs=-1,
        random_state=42
    )

    train_losses = []
    val_losses = []
    n_generations = 30

    for gen in range(1, n_generations + 1):
        model.set_params(generations=gen)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        train_losses.append(mean_squared_error(y_train, y_pred_train))
        val_losses.append(mean_squared_error(y_val, y_pred_val))

    best_expr = str(model._program)

    out_plot = os.path.join(RES_GPLEARN, f"{dataset_name}_gplearn_loss.png")
    plot_losses(train_losses, val_losses, n_generations, loss_type='MSE', filename=out_plot)

    with open(os.path.join(RES_GPLEARN, f"{dataset_name}_result.txt"), "w") as f:
        f.write(f"gplearn expr:\n{best_expr}\n")
    print(f"[gplearn] {dataset_name}: {best_expr}")



def run_pysindy(X, y, dataset_name, feature_names):
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    library_functions = [
        lambda x: x,
        lambda x: 1.0 / (x + 1e-8),
        lambda x: 1.0 / (x**2 + 1e-8)
    ]
    function_names = [
        lambda x: x,
        lambda x: f"(1/{x})",
        lambda x: f"(1/{x}^2)"
    ]
    lib_custom = ps.CustomLibrary(
        library_functions=library_functions, 
        function_names=function_names
    )
    
    lib_poly = ps.PolynomialLibrary(degree=2, include_interaction=True)
    
    try:
        combined_lib = ps.TensoredLibrary([lib_poly, lib_custom])
    except AttributeError:
        combined_lib = ps.GeneralizedLibrary([lib_poly, lib_custom])

    optimizer = ps.STLSQ(threshold=0.01)
    model = ps.SINDy(feature_library=combined_lib, optimizer=optimizer)

    try:
        # Se ajusta rigurosamente sobre el conjunto de entrenamiento
        model.fit(X_train, t=1.0, x_dot=y_train, feature_names=feature_names)
        
        eqs = model.equations()
        best_expr = eqs[0] if eqs else "0"
        
        # Validación post-ajuste (sin épocas)
        train_mse = mean_squared_error(y_train, model.predict(X_train, t=1.0))
        val_mse = mean_squared_error(y_val, model.predict(X_val, t=1.0))
        
        output_path = os.path.join(RES_PYSINDY, f"{dataset_name}_result.txt")
        with open(output_path, "w") as f:
            f.write(f"PySINDy expr:\n{best_expr}\n")
            f.write(f"MSE Train: {train_mse}\n")
            f.write(f"MSE Val: {val_mse}\n")
            
        print(f"[PySINDy] {dataset_name}: y = {best_expr} | MSE_val = {val_mse:.4e}")
        
    except Exception as e:
        print(f"Error en SINDy ({dataset_name}): {e}")


# -------------------------------------------------------------------------
# 2. Ejecución Principal
# -------------------------------------------------------------------------


def main():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("Fallo de I/O: No se encontraron archivos CSV en el directorio.")
        return

    for file in csv_files:
        if any(name in file for name in names):            
            dataset_name = os.path.basename(file).replace('.csv', '')
        else:
            continue
        
        df = pd.read_csv(file)
        
        # Asumimos que la última columna es el target
        target_col = df.columns[-1]
        feature_cols = df.columns[:-1].tolist()
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Filtrado de NaN e Infinitos crítico para la inversión matricial en SINDy
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isinf(X).any(axis=1) & ~np.isinf(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"--- Procesando {dataset_name} ---")
        
        if qlattice:
            run_qlattice(df, target_col, dataset_name, filename='qlattice')

        
        if gplearn:
            run_gplearn(X, y, dataset_name)
        
        if sindy:
            run_pysindy(X, y, dataset_name, feature_cols)
        

if __name__ == "__main__":
    main()