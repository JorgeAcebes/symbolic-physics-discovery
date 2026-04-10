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

# TODO: Asemejarlo al código de Andrés para que sea el mismo pipeline
# TODO: Plot en log (y) si es loss muy grande.
# TODO: Implementar funciones en gplearn y en syndy. 



# ================================================================================================
# This should be temporary. We should implemet a confing.json in where we would choose what models

qlattice = 0
gplearn = 0
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

def plot_losses(train_losses, val_losses, epochs, loss_type=None, filename=None, model=None):
    x0 = list(range(1, epochs + 1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(x0, train_losses, label='Train loss', linewidth=1.5)
    plt.plot(x0, val_losses, label='Validation loss', linewidth=1.5)
    
    plt.title('Evolución del Error' if not model else f'Evolución del Error [{model}]')
    plt.xlabel('Iteraciones (Épocas / Generaciones)')
    plt.ylabel('Loss' if not loss_type else f'Loss ({loss_type})')
    
    # Añadido rigor visual para analizar convergencia
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    if filename:
        # bbox_inches asegura que no se recorten las etiquetas al guardar
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        
    plt.show()


def run_qlattice(df, target_col, dataset_name):
    random_state = 42
    kind = 'regression'
    output_name = target_col
    n_epochs = 10 
    threads = 4 
    criterion = 'bic' 
    max_complexity = 7 

    feyn.validate_data(df, kind, output_name)
    
    # División nativa directa en 60/20/20
    train, val, test = feyn.tools.split(df, ratio=[0.6, 0.2, 0.2], random_state=random_state)

    ql = feyn.QLattice(random_seed=random_state)

    train_losses = []
    val_losses = []
    models = []

    for epoch in range(1, n_epochs + 1):
        models += ql.sample_models(train, output_name, kind, max_complexity=max_complexity)
        models = feyn.fit_models(models, train, threads=threads, loss_function='squared_error', criterion=criterion)
        models = feyn.prune_models(models)

        train_mse = np.mean(squared_error(train[output_name], models[0].predict(train)))
        val_mse = np.mean(squared_error(val[output_name], models[0].predict(val)))
        
        train_losses.append(train_mse)
        val_losses.append(val_mse)

        feyn.show_model(
            models[0],
            label=f"Epoch {epoch}/{n_epochs}.",
            update_display=True,
        )

        ql.update(models)

    models = feyn.get_diverse_models(models)
    
    out_plot = os.path.join(RES_QLATTICE, f"{dataset_name}_qlattice_loss.png")
    plot_losses(train_losses, val_losses, n_epochs, loss_type='MSE', filename=out_plot, model ='QLattice')
    
    best_model = models[0]
    best_model.show(update_display=True)
    best_model.plot(train, filename=os.path.join(RES_QLATTICE, f"{dataset_name}_plot.html"))

    best_expr = str(best_model.sympify(signif=4))
    
    # Evaluación ciega en el test set
    test_mse = np.mean(squared_error(test[output_name], best_model.predict(test)))
    
    out_path = os.path.join(RES_QLATTICE, f"{dataset_name}_result.txt")
    with open(out_path, "w") as f:
        f.write("QLattice expr:\n")
        f.write(f"{best_expr}\n")
        f.write(f"MSE Train: {train_losses[-1]:.6e}\n")
        f.write(f"MSE Val: {val_losses[-1]:.6e}\n")
        f.write(f"MSE Test: {test_mse:.6e}\n")
    
    print(f"[QLattice] {dataset_name}: y = {best_expr} | MSE_test = {test_mse:.4e}")



def run_gplearn(X, y, dataset_name):
    # Separación en dos fases: primero 20% test, luego 25% del resto (que es 20% del total) para val
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    model = SymbolicRegressor(
        population_size=2000,
        generations=1,
        warm_start=True,
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

    out_plot = os.path.join(RES_GPLEARN, f"{dataset_name}_gplearn_loss.png")
    plot_losses(train_losses, val_losses, n_generations, loss_type='MSE', filename=out_plot, model='GPLearn')

    best_expr = str(model._program)
    
    # Evaluación ciega final
    test_mse = mean_squared_error(y_test, model.predict(X_test))

    with open(os.path.join(RES_GPLEARN, f"{dataset_name}_result.txt"), "w") as f:
        f.write("gplearn expr:\n")
        f.write(f"{best_expr}\n")
        f.write(f"MSE Train: {train_losses[-1]:.6e}\n")
        f.write(f"MSE Val: {val_losses[-1]:.6e}\n")
        f.write(f"MSE Test: {test_mse:.6e}\n")
        
    print(f"[gplearn] {dataset_name}: {best_expr} | MSE_test = {test_mse:.4e}")


def run_pysindy(X, y, dataset_name, feature_names):
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

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
        # Entrenamiento estricto sobre el 60%
        model.fit(X_train, t=1.0, x_dot=y_train, feature_names=feature_names)
        
        eqs = model.equations()
        best_expr = eqs[0] if eqs else "0"
        
        # Extracción de errores analíticos
        train_mse = mean_squared_error(y_train, model.predict(X_train))
        val_mse = mean_squared_error(y_val, model.predict(X_val))
        test_mse = mean_squared_error(y_test, model.predict(X_test))
        
        output_path = os.path.join(RES_PYSINDY, f"{dataset_name}_result.txt")
        with open(output_path, "w") as f:
            f.write("PySINDy expr:\n")
            f.write(f"{best_expr}\n")
            f.write(f"MSE Train: {train_mse:.6e}\n")
            f.write(f"MSE Val: {val_mse:.6e}\n")
            f.write(f"MSE Test: {test_mse:.6e}\n")
            
        print(f"[PySINDy] {dataset_name}: y = {best_expr} | MSE_test = {test_mse:.4e}")
        
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

        
        if gplearn:
            run_gplearn(X, y, dataset_name)
        
        if sindy:
            run_pysindy(X, y, dataset_name, feature_cols)
        

if __name__ == "__main__":
    main()