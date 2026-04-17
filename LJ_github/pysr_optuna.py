import optuna
import optuna.importance as imp
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from pysr import PySRRegressor

# 1. Define all the datasets you want to test
experiment_files = [
    r"C:\Users\loren\Downloads\Cuarto año uni\IA\symbolic-physics-discovery\data\coulomb_no_noise.csv",
    # Add your other datasets here! 
    # r"C:\Users\loren\Downloads\...\data\newton_gravity.csv",
    # r"C:\Users\loren\Downloads\...\data\hookes_law.csv"
]

# 2. Open a text file to record the results ("w" creates/overwrites, "a" appends)
output_log_path = r"C:\Users\loren\Downloads\Cuarto año uni\IA\symbolic-physics-discovery\LJ"

with open(output_log_path, "w", encoding="utf-8") as log_file:
    
    # 3. Loop through each dataset
    for filepath in experiment_files:
        
        # Extract just the filename to make the logs pretty
        experiment_name = filepath.split("\\")[-1]
        
        print(f"\n{'='*50}", file=log_file)
        print(f"STARTING EXPERIMENT: {experiment_name}", file=log_file)
        print(f"{'='*50}\n", file=log_file)

        # Load the specific dataset for this loop
        df_example = pd.read_csv(filepath) 
        
        X = df_example.iloc[:, :-1]
        y = df_example.iloc[:, -1]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Define the objective function INSIDE the loop 
        # This is crucial so it uses the correct X_train/y_train for THIS experiment
        def objective(trial):
            populations = trial.suggest_int("populations", 10, 50)
            niterations = trial.suggest_int("niterations", 20, 100)
            population_size = trial.suggest_int("population_size", 20, 100)
            parsimony = trial.suggest_float("parsimony", 1e-5, 1e-1, log=True)
            
            num_seeds = 1
            losses = []
            start_sim_time = time.time()

            best_overall_loss = float('inf')
            best_overall_equation = ""

            for seed in range(num_seeds):
                model = PySRRegressor(
                     niterations=niterations,
                     populations=populations,
                     parsimony=parsimony,
                     random_state=seed, 
                     deterministic=True,
                     parallelism='serial',
                     binary_operators=["*", "/", "^", "+", "-"],
                     constraints={'^': (-1, 1)},
                     population_size=population_size,
                 )
                
                model.fit(X_train, y_train)

                best_row = model.equations_.iloc[model.equations_["score"].idxmax()]
                best_loss = best_row["loss"]

                best_eq_string = best_row["equation"]

                losses.append(best_loss)

                if best_loss < best_overall_loss:
                    best_overall_loss = best_loss
                    best_overall_equation = best_eq_string

            end_sim_time = time.time()
            trial.set_user_attr("simulation_time", end_sim_time - start_sim_time)

            trial.set_user_attr("best_equation", best_overall_equation)

            trial.set_user_attr("best_equation_loss", best_overall_loss)

            return np.mean(losses), np.std(losses)

        # --- Run Optuna Study ---
        print(f"Starting Optuna Study for {experiment_name}...", file=log_file)
        my_sampler = optuna.samplers.NSGAIISampler(seed=42)
        study = optuna.create_study(directions=["minimize", "minimize"], sampler=my_sampler)
        
        start_optuna_time = time.time()
        study.optimize(objective, n_trials=2) # Remember to increase n_trials!
        end_optuna_time = time.time()
        
        # --- Record Results to File ---
        print("\n--- Time Report ---", file=log_file)
        print(f"Total Optuna Search Time: {end_optuna_time - start_optuna_time:.2f} seconds", file=log_file)

        print("\n--- Pareto Front (Best Trade-offs) ---", file=log_file)
        pareto_front = study.best_trials
        
        for i, trial in enumerate(pareto_front):
            print(f"\nOptimal Trial #{i+1} (Trial Number: {trial.number})", file=log_file)
            print(f"  Mean Loss: {trial.values[0]:.4f}", file=log_file)
            print(f"  Standard Deviation: {trial.values[1]:.4f}", file=log_file)
            print(f"  Simulation Time: {trial.user_attrs.get('simulation_time'):.2f} seconds", file=log_file)

            best_eq = trial.user_attrs.get('best_equation', "No equation found")
            print(f"  Best Equation Found: {best_eq}", file=log_file)

            best_eq_loss = trial.user_attrs.get('best_equation_loss', float('inf'))
            print(f"  Loss of Best Equation: {best_eq_loss:.6e}", file=log_file)

            print("  Hyperparameters:", file=log_file)
            for key, value in trial.params.items():
                print(f"    {key}: {value}", file=log_file)

        # --- Record Importances to File ---
        print("\n--- Hyperparameter Importances for MEAN LOSS ---", file=log_file)
        importances_mean = imp.get_param_importances(study, target=lambda t: t.values[0])
        for param, importance_value in importances_mean.items():
            print(f"  {param}: {importance_value * 100:.2f}%", file=log_file)

        print("\n--- Hyperparameter Importances for STANDARD DEVIATION ---", file=log_file)
        try:
            importances_std = imp.get_param_importances(study, target=lambda t: t.values[1])
            for param, importance_value in importances_std.items():
                print(f"  {param}: {importance_value * 100:.2f}%", file=log_file)
        except RuntimeError as e:
            if "zero total variance" in str(e):
                print("  Could not calculate importances: Standard deviation was identical across all trials.", file=log_file)
            else:
                raise e

print("\nAll experiments complete! Check the 'all_experiments_results.txt' file.")