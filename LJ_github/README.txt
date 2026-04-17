# PySR Hyperparameter Optimization with Optuna

## Overview
This script automates the discovery of symbolic physics equations by combining **PySR** (Symbolic Regression) with **Optuna** (Hyperparameter Optimization). It evaluates multiple physical datasets, searches for the most accurate and parsimonious mathematical expressions, and logs the best hyperparameters and resulting equations. 

## Dependencies
Ensure you have the following Python libraries installed before running the script:
* [cite_start]`optuna` [cite: 1]
* [cite_start]`numpy` [cite: 1]
* [cite_start]`pandas` [cite: 1]
* [cite_start]`scikit-learn` (specifically `train_test_split`) [cite: 1]
* [cite_start]`pysr` (`PySRRegressor`) [cite: 1]
* [cite_start]`time` (built-in) [cite: 1]

## How the Script Works

### 1. Data Pipeline
* [cite_start]The script iterates through a predefined list of CSV datasets provided in `experiment_files` (e.g., `coulomb_no_noise.csv`)[cite: 1, 2].
* [cite_start]For each dataset, the data is split into features (`X`) and a target variable (`y`)[cite: 3].
* [cite_start]The data is then divided into training and validation sets using an 80/20 split (`test_size=0.2`) with a fixed random state for reproducibility[cite: 3].

### 2. Hyperparameter Search Space
An objective function is defined inside the loop to run a trial using **Optuna**, searching within the following hyperparameter bounds:
* [cite_start]**`populations`**: Integer between 10 and 50[cite: 4].
* [cite_start]**`niterations`**: Integer between 20 and 100[cite: 4].
* [cite_start]**`population_size`**: Integer between 20 and 100[cite: 4].
* **`parsimony`**: Float between 1e-5 and 1e-1 (sampled on a log scale)[cite: 4, 5].

### 3. Symbolic Regression Model
* [cite_start]The script configures `PySRRegressor` with the suggested hyperparameters and a standard set of binary operators: `*`, `/`, `^`, `+`, `-`[cite: 6, 7].
* [cite_start]Constraints are applied to specific operators (e.g., limiting the power operator `^` to a range of -1 to 1)[cite: 7].
* It runs the model deterministically in serial parallelism[cite: 7].
* [cite_start]After fitting the model to the training data, it isolates the best equation based on the highest score[cite: 8].

### 4. Multi-Objective Optimization
* [cite_start]The Optuna study uses the `NSGAIISampler` (with a fixed seed) to perform multi-objective optimization[cite: 11]. 
* It optimizes for two distinct directions: minimizing the mean loss and minimizing the standard deviation across runs[cite: 11]. *(Note: Be sure to increase `n_trials` in the code for a full production run!)* [cite: 11]

### 5. Logging and Outputs
[cite_start]All results are appended and formatted cleanly into a text file specified by `output_log_path`[cite: 2]. The generated log includes:
* **Run Times**: Total time taken for the Optuna search and individual simulation times[cite: 10, 12].
* [cite_start]**Pareto Front Trade-offs**: Details of the best trials, including mean loss, standard deviation, and the best hyperparameter combinations[cite: 12, 13, 14].
* [cite_start]**Equations**: The exact string representation of the best equation found and its associated loss[cite: 13, 14].
* **Feature Importance**: An analysis of which hyperparameters most heavily influenced the mean loss and standard deviation (includes error handling if the standard deviation variance is zero)[cite: 14, 15, 16].