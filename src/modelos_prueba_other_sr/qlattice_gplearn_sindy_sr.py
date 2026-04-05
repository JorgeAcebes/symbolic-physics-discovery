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


# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../..", "data"))
RESULTS_BASE = os.path.abspath(os.path.join(BASE_DIR, "../../results/testing_other_sr"))

RES_QLATTICE = os.path.join(RESULTS_BASE, "testing_qlattice")
RES_GPLEARN = os.path.join(RESULTS_BASE, "testing_gplearn")
RES_PYSINDY = os.path.join(RESULTS_BASE, "testing_pysindy")


# ================================================================================================
# This should be temporary. We should implemet a confing.json in where we would choose what models

qlattice = 0 # Currently working only on it
qlattice_manual = 1
gplearn = 0
sindy = 0

# Only dataset containing these words will be used. Using OR, not AND.
names = ['kepler_low_noise'] 

# TODO: Implement logic so that when "all" is written, all datasets are used
# ================================================================================================


for path in [RES_QLATTICE, RES_GPLEARN, RES_PYSINDY]:
    os.makedirs(path, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Modelos de Regresión Simbólica
# -------------------------------------------------------------------------

def run_qlattice(df, target_col, dataset_name):
    """
    QLattice: Explora un hipergrafo probabilístico para extraer subgrafos
    que representan las leyes físicas.
    """

    random_state = 42

    # Dividimos nuestra data. Actualmente lo realizo dentro de la función run_qlattice,
    # sería interesante poder realizar el split fuera para que fuese igual para todos.
    train_data, val_data, test_data = feyn.tools.split(df, ratio=[0.6, 0.2, 0.2], random_state=random_state)

    # Instanciamos el hipergrafo. Escogemos QLattice en vez de connect_qlattice() porque queremos correr en local + no necesitar API de Abzu
    ql = feyn.QLattice(random_seed=random_state)


    # auto_run maneja el ciclo de entrenamiento (epoching) internamente
    models = ql.auto_run(
        data=train_data,
        output_name=target_col,
        loss_function='squared_error', # MSE (realmente no hacemos la media pero para optimizar es idéntico)
        n_epochs=10,      # 10 épocas (default)
        max_complexity=7, # Forzamos parsimonia para física
        criterion="bic"  # Bayesian Information Criterion para penalizar sobreajuste
    )

    # El índice 0 contiene el modelo topológico óptimo
    best_model = models[0]
    best_model.plot(train_data, filename=os.path.join(RES_QLATTICE, f"{dataset_name}_plot.html"))
    best_model.plot_signal(train_data, filename=os.path.join(RES_QLATTICE, f"{dataset_name}_plot_signal.html"))
    feyn.plots.plot_model_summary(best_model, train_data, filename=os.path.join(RES_QLATTICE, f"{dataset_name}_plot_summary.html"))
    best_expr = str(best_model.sympify(signif=4))
    
    # Guardar resultados
    out_path = os.path.join(RES_QLATTICE, f"{dataset_name}_result.txt")
    with open(out_path, "w") as f:
        f.write(f"Equation obtained for {dataset_name}:\n")
        f.write(best_expr + "\n")
    
    print(f"[QLattice] {dataset_name}: {best_expr}")



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
def run_qlattice_manual(df, target_col, dataset_name, filename=None):
    random_state = 42
    kind = 'regression'
    loss_function = 'squared_error'
    output_name = target_col
    n_epochs = 10
    threads = 4 
    criterion = 'bic'
    max_complexity = 7   


    feyn.validate_data(df, kind, output_name) # Valida el DataFrame
    train, val, test = feyn.tools.split(df, ratio=[0.6, 0.2, 0.2], random_state=random_state)


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
    
    out_path = os.path.join(RES_QLATTICE, f"{dataset_name}_{filename}.png")
    # Display the final model and the loss graph
    plot_losses(train_losses, val_losses, epoch, loss_type='MSE', filename=out_path)
    best_model = models[0]
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
    model = SymbolicRegressor(
        population_size=2000,
        generations=30,
        function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos'),
        metric='mse',
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05, # Control de parsimonia (poda)
        p_point_mutation=0.1,
        n_jobs=-1 # Paralelización multinúcleo
    )
    
    model.fit(X, y)
    best_expr = str(model._program)
    
    with open(os.path.join(RES_GPLEARN, f"{dataset_name}_result.txt"), "w") as f:
        f.write(f"gplearn expr:\n{best_expr}\n")
    print(f"[gplearn] {dataset_name}: {best_expr}")



def run_pysindy(X, y, dataset_name, feature_names):
    # Corrección 1: Asegurar que y es un vector columna (N, 1) para evitar AxesWarning
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Corrección 2: Definir nombres analíticos para la CustomLibrary
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
    
    # Intentamos crear un producto tensorial de librerías para capturar (q1 * q2 * 1/r^2).
    # Si la versión de PySINDy es antigua, hace fallback a la concatenación lineal.
    try:
        combined_lib = ps.TensoredLibrary([lib_poly, lib_custom])
    except AttributeError:
        combined_lib = ps.GeneralizedLibrary([lib_poly, lib_custom])

    optimizer = ps.STLSQ(threshold=0.01)
    
    # Corrección 3: Pasamos la biblioteca combinada correctamente
    model = ps.SINDy(feature_library=combined_lib, optimizer=optimizer)

    try:
        # Corrección 4: Se requiere el dummy t=1.0 para que el validador estático funcione
        model.fit(X, t=1.0, x_dot=y, feature_names=feature_names)
        
        eqs = model.equations()
        best_expr = eqs[0] if eqs else "0"
        
        # Asume que RES_PYSINDY está definido globalmente en tu script
        output_path = os.path.join(RES_PYSINDY, f"{dataset_name}_result.txt")
        with open(output_path, "w") as f:
            f.write(f"PySINDy expr:\n{best_expr}\n")
    
        print(f"[PySINDy] {dataset_name}: y = {best_expr}")
        
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
            run_qlattice(df, target_col, dataset_name)
        
        if qlattice_manual:
            run_qlattice_manual(df, target_col, dataset_name, filename='manual')

        
        if gplearn:
            run_gplearn(X, y, dataset_name)
        
        if sindy:
            run_pysindy(X, y, dataset_name, feature_cols)
        

if __name__ == "__main__":
    main()