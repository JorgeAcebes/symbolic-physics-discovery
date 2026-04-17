import optuna
import optuna.visualization as vis
import optuna.importance as imp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pysr import PySRRegressor

# We take X_train and y_train from this example:
df_example = pd.read_csv(r"C:\Users\loren\Downloads\Cuarto año uni\IA\symbolic-physics-discovery\data\coulomb_no_noise.csv") #cambiar a conveniencia
print(df_example.head())
X = df_example.iloc[:, :-1]
y = df_example.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#X_train = X.iloc[:800, :]
#X_val = X.iloc[801:, :]
#y_train = y.iloc[:800]
#y_val = y.iloc[801:]

# --- 1. Define your Monte Carlo Objective Function ---
def objective(trial):
    """
    This function is called by Optuna for every trial. 
    It defines the hyperparameters, runs the Monte Carlo simulation, 
    and returns the final score.
    """
    
    populations = trial.suggest_int("populations", 10, 50)
    niterations = trial.suggest_int("niterations", 20, 100)
    population_size = trial.suggest_int("population_size", 20, 100)
    
    # We use log=True for parsimony because we want to search across magnitudes
    parsimony = trial.suggest_float("parsimony", 1e-5, 1e-1, log=True)
    
    # Set up the Monte Carlo Simulation
    num_seeds = 7
    losses = []
    
    for seed in range(num_seeds):
        model = PySRRegressor(
             niterations=niterations,
             populations=populations,
             parsimony=parsimony,
             random_state=seed, # Crucial for Monte Carlo
             binary_operators=["*", "/", "^", "+", "-"],
             constraints={'^': (-1, 1)},
             population_size=population_size,
         )
        
        model.fit(X_train, y_train)

        best_loss = model.equations_.iloc[model.equations_["score"].idxmax()]["loss"]
        loss = best_loss
        losses.append(loss)
        
        # Optional: Optuna Pruning (Early Stopping)
        # If the first seed is a total disaster, tell Optuna to abort this trial
        # trial.report(loss, step=seed)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()

    # C. Return the average loss across all seeds
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)

    trial.set_user_attr("loss_std", std_loss)

    return mean_loss, std_loss

# --- 2. Create the Study and Run it ---
if __name__ == "__main__":
    print("Starting Optuna Study...")
    
    # We want to MINIMIZE the error/loss and std
    study = optuna.create_study(directions=["minimize", "minimize"])
    
    # Run 20 trials (increase this to 50 or 100 for your actual project)
    study.optimize(objective, n_trials=20)
    
   # --- Retrieving Your Results ---
    print("\n--- Pareto Front (Best Trade-offs) ---")
    
    # best_trials returns a list of the optimal trade-off configurations
    pareto_front = study.best_trials
    
    for i, trial in enumerate(pareto_front):
        print(f"\nOptimal Trial #{i+1} (Trial Number: {trial.number})")
        # trial.values[0] is mean_loss, trial.values[1] is std_loss
        print(f"  Mean Loss: {trial.values[0]:.4f}")
        print(f"  Standard Deviation: {trial.values[1]:.4f}")
        print("  Hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    # ---------------------------------------------------------
    # Analyze Importances for Mean Loss (First Objective: values[0])
    # ---------------------------------------------------------
    print("\n--- Hyperparameter Importances for MEAN LOSS ---")
    importances_mean = imp.get_param_importances(study, target=lambda t: t.values[0])
    
    for param, importance_value in importances_mean.items():
        percentage = importance_value * 100
        print(f"  {param}: {percentage:.2f}%")

    # ---------------------------------------------------------
    # Analyze Importances for Standard Deviation (Second Objective: values[1])
    # ---------------------------------------------------------
    print("\n--- Hyperparameter Importances for STANDARD DEVIATION ---")
    try:
        importances_std = imp.get_param_importances(study, target=lambda t: t.values[1])
        for param, importance_value in importances_std.items():
            percentage = importance_value * 100
            print(f"  {param}: {percentage:.2f}%")
            
    except RuntimeError as e:
        if "zero total variance" in str(e):
            print("  Could not calculate importances: Standard deviation was identical across all trials.")
        else:
            # Re-raise the error if it's something else
            raise e
