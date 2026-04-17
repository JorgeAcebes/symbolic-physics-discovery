# Optuna + PySR Batch Experiment Runner

This script runs a batch of symbolic regression experiments using **PySR** and tunes selected PySR hyperparameters with **Optuna**. It is designed to test one or more datasets, search for good hyperparameter combinations, and log the best-performing symbolic equations and optimization results to a text file.

---

## What this script does

At a high level, the script:

1. Defines a list of CSV datasets to test.
2. Loops through each dataset one by one.
3. Splits each dataset into training and validation sets.
4. Uses **Optuna** to tune several **PySRRegressor** hyperparameters.
5. For each Optuna trial, trains a PySR model and extracts the best equation found.
6. Treats the optimization as a **multi-objective problem**:
   - Minimize the **mean loss**
   - Minimize the **standard deviation of the loss**
7. Writes the results for each dataset to a log file, including:
   - total optimization time
   - Pareto-optimal trials
   - best equation found
   - best equation loss
   - chosen hyperparameters
   - hyperparameter importance analysis

---

## Main purpose

The purpose of this script is to systematically compare symbolic regression settings across datasets and identify good trade-offs between:

- **accuracy** (low loss)
- **stability** (low variation in loss across runs)

This is useful when PySR performance depends heavily on hyperparameter choices.

---

## Workflow

### 1. Dataset definition

The script starts by defining a list called `experiment_files`:

```python
experiment_files = [
    r"...\coulomb_no_noise.csv",
]
