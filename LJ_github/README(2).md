# Symbolic Regression Hyperparameter Search with PySR + Optuna

## Overview

This script performs **multi-objective hyperparameter optimization** for **symbolic regression** experiments using [PySR](https://github.com/MilesCranmer/PySR) and [Optuna](https://optuna.org/).

Its main goal is to evaluate how different PySR hyperparameter configurations behave across one or more datasets, while considering two desirable properties at the same time:

1. **Low predictive error** on a validation set.
2. **Low variability across random seeds**, so that a configuration is not only good once, but reasonably stable.

To do this, the script:

- loops through a list of CSV datasets,
- splits each dataset into training and validation subsets,
- runs an Optuna study,
- evaluates each trial across **10 different random seeds**,
- computes the **mean** and **standard deviation** of validation MSE across those seeds,
- stores information about the best-performing equation found within each trial,
- logs Pareto-optimal trial results and hyperparameter importances to a text file.

The script is designed as an experimental framework for studying the robustness of PySR hyperparameters on symbolic-physics-style datasets.

---

## What the code does step by step

### 1. Imports the required libraries

The script uses:

- **Optuna** for hyperparameter optimization,
- **NumPy** and **pandas** for numerical and tabular handling,
- **time** for runtime measurements,
- **scikit-learn** for train/validation splitting and MSE,
- **PySRRegressor** for symbolic regression.

---

### 2. Defines the datasets to test

A list called `experiment_files` stores the CSV files that will be used as independent experiments.

Each dataset is expected to follow a simple convention:

- **all columns except the last one** are input features,
- **the last column** is the target variable.

This allows the same pipeline to be reused across multiple symbolic regression datasets.

---

### 3. Opens a log file for all experiment results

The script writes all outputs into a text file:

`all_experiments_results.txt`

This file stores:

- experiment headers,
- Optuna runtime summaries,
- Pareto-front trials,
- best equations,
- associated losses,
- seed runtime information,
- hyperparameter importance rankings.

---

### 4. Loops through each dataset

For every file in `experiment_files`, the script:

- extracts a cleaner experiment name from the path,
- loads the CSV with `pandas.read_csv`,
- separates features `X` and target `y`,
- creates a train/validation split with `test_size=0.2` and `random_state=42`.

This means every dataset gets its own independent optimization process.

---

### 5. Defines the Optuna objective function

Inside each dataset loop, the script defines `objective(trial)`.
This is important because the objective needs access to the **current dataset's** `X_train`, `X_val`, `y_train`, and `y_val`.

For each Optuna trial, it samples four PySR hyperparameters:

- `populations` from 10 to 50,
- `niterations` from 20 to 100,
- `population_size` from 20 to 100,
- `parsimony` from `1e-5` to `1e-1` on a logarithmic scale.

These control the search dynamics and the simplicity penalty in PySR.

---

### 6. Repeats each trial across 10 random seeds

For every sampled hyperparameter configuration, the script runs **10 PySR fits**, one per seed:

```python
for seed in range(num_seeds):
```

with:

```python
num_seeds = 10
```

This is a major strength of the current version of the code, because PySR is stochastic and a configuration that performs well in one run may perform worse in another.

By using several seeds, the script estimates both:

- **average performance**, and
- **stability of that performance**.

---

### 7. Trains a PySR model for each seed

For each seed, the script creates a `PySRRegressor` with the sampled hyperparameters and some fixed settings:

- `random_state=seed`
- `deterministic=True`
- `parallelism='serial'`
- binary operators: `["*", "/", "^", "+", "-"]`
- exponent constraints: `{'^': (-1, 1)}`
- tuned `population_size`, `niterations`, `populations`, and `parsimony`

Then it fits the model on the training set:

```python
model.fit(X_train, y_train)
```

---

### 8. Selects one candidate equation from PySR and evaluates it on validation data

After fitting, the script looks at `model.equations_` and selects the row with the **highest internal PySR score**:

```python
best_idx = model.equations_["score"].idxmax()
```

It stores the corresponding equation string and then predicts using exactly that equation:

```python
y_val_pred = model.predict(X_val, index=best_idx)
```

Then it computes the validation loss:

```python
val_loss = mean_squared_error(y_val, y_val_pred)
```

This is a solid improvement over the previous version because the validation error is now computed for the same equation that was selected from `equations_`.

---

### 9. Tracks the best seed within each trial

For each Optuna trial, the script keeps track of:

- the lowest validation loss found among the 10 seeds,
- the corresponding equation,
- the runtime of that specific best seed.

This allows the final report to show not only the aggregate performance of a hyperparameter configuration, but also the single best equation obtained under that configuration.

---

### 10. Returns two optimization objectives

At the end of the trial, the objective function returns:

- `np.mean(losses)`
- `np.std(losses)`

So Optuna performs **multi-objective optimization**, searching for configurations that balance:

- low average validation error,
- low variability across repeated stochastic runs.

This is useful because in symbolic regression, the most accurate configuration is not always the most reliable one.

---

### 11. Runs an Optuna NSGA-II study

The script creates a multi-objective study with:

- **NSGA-II** as sampler,
- directions `["minimize", "minimize"]`,
- `n_trials=40`.

This means Optuna searches for a **Pareto front** of good trade-offs between the two objectives, rather than forcing everything into one scalar score.

---

### 12. Writes the Pareto front to the log file

After optimization, the script retrieves:

```python
pareto_front = study.best_trials
```

and writes, for each Pareto-optimal trial:

- mean loss,
- standard deviation of loss,
- total simulation time,
- best equation found,
- loss of the best equation,
- runtime of the best seed,
- hyperparameter values.

This gives a readable summary of the best trade-off solutions discovered by the search.

---

### 13. Computes hyperparameter importances

The script also computes parameter importances separately for:

- **mean loss**,
- **standard deviation of loss**.

It includes a safeguard in case the standard deviation objective has zero variance across trials, in which case Optuna cannot compute meaningful importances.

---

## Main idea behind the methodology

The core methodological idea of this script is:

> Do not judge a PySR hyperparameter configuration from a single run.

Instead, for each hyperparameter setting, the code asks two questions:

1. **How good is this configuration on average?**
2. **How sensitive is it to random initialization?**

That makes the framework more scientifically meaningful than a single-run benchmark, especially for symbolic regression, where stochastic search can produce substantial run-to-run variation.

---

## Strengths of the current version

Compared with a simpler one-run tuning loop, this version has several strong points:

### 1. It measures robustness explicitly
Using 10 seeds makes the second objective meaningful and turns the search into a stability-aware tuning process.

### 2. It uses genuine multi-objective optimization
The code does not collapse accuracy and robustness into one arbitrary scalar. Instead, it keeps them separate through a Pareto-front analysis.

### 3. It correctly evaluates the selected PySR equation
The script now predicts with `index=best_idx`, so the reported validation loss corresponds to the chosen equation from `equations_`.

### 4. It records useful metadata
It stores runtime, best equation, best-equation loss, and best-seed time as Optuna trial attributes, improving interpretability of results.

### 5. It supports multiple datasets
The framework is already structured so that several symbolic discovery problems can be tested sequentially.

---

## Current limitations

Even though this version is much better, it still has several methodological and practical limitations.

### 1. The dataset paths are hard-coded
Both the input dataset paths and the output log path are absolute Windows paths. This makes the script hard to reuse on another machine or in a shared repository without manual editing.

**Why this matters:**
Anyone cloning the repository will have to modify the paths before the code can run.

---

### 2. There is no independent test set
The script splits the data into only:

- training set
- validation set

The validation set is used repeatedly during hyperparameter selection, which means the final chosen configuration may become indirectly tuned to that validation split.

**Why this matters:**
The reported validation performance may be optimistic. A final untouched test set would provide a more honest estimate of generalization.

---

### 3. The evaluation depends on a single train/validation split
The code always uses one split with `random_state=42`.

**Why this matters:**
A configuration may look good or bad partly because of the specific split. This is especially relevant for small datasets, noisy datasets, or datasets with uneven coverage.

---

### 4. The “best equation” inside each seed is chosen by PySR score, not directly by validation loss among all candidate equations
Within each fitted PySR model, the script selects the row with maximum `score` from `model.equations_`, and only then evaluates that equation on the validation set.

**Why this matters:**
The equation with the best internal PySR score is not necessarily the equation with the best validation MSE among all equations in the hall of fame. In some cases, a different row in `equations_` might generalize better.

---

### 5. Search space is still relatively narrow
The script tunes only four hyperparameters:

- `populations`
- `niterations`
- `population_size`
- `parsimony`

**Why this matters:**
PySR behavior is also affected by many other settings, such as mutation strategies, unary operators, complexity controls, batching options, warm starts, loss functions, denoising-related settings, and stopping criteria. The current script explores only a limited subset of the possible design space.

---

### 6. `n_trials=40` may still be too small for expensive stochastic optimization
Forty trials is far better than two, but the search is still modest given that each trial contains **10 full PySR runs**.

**Why this matters:**
The Pareto front may still be unstable, incomplete, or highly dependent on the current search budget.

---

### 7. The operator set is restrictive
The model only uses binary operators:

```python
["*", "/", "^", "+", "-"]
```

with exponent constraints:

```python
{'^': (-1, 1)}
```

**Why this matters:**
This is appropriate for some physics laws, but it may be too restrictive for datasets requiring functions such as `sin`, `cos`, `exp`, `log`, or more flexible exponent structures.

---

### 8. The script assumes the CSV format is always correct
It assumes:

- the file exists,
- the CSV loads correctly,
- the last column is the target,
- all values are numeric,
- there are no missing values,
- the dataset is large enough for splitting.

**Why this matters:**
Any malformed dataset can crash the experiment.

---

### 9. No experiment persistence or resume mechanism
The study is created in memory and is not connected to an Optuna storage backend.

**Why this matters:**
If the process is interrupted, all optimization progress is lost.

---

### 10. The runs are forced to serial execution
The code uses:

```python
parallelism='serial'
```

**Why this matters:**
This may improve determinism and simplify debugging, but it can make large experiments very slow.

---

## Possible improvements

### 1. Replace hard-coded paths with configurable paths
A good improvement would be to use:

- `pathlib`,
- relative project paths,
- command-line arguments,
- or a config file (`.json`, `.yaml`, `.toml`).

This would make the script much easier to reproduce and share.

---

### 2. Add a true test set
A better evaluation pipeline would be:

- train set for fitting,
- validation set for Optuna selection,
- test set for final unbiased reporting.

This would separate model selection from final performance estimation.

---

### 3. Use repeated splits or cross-validation
Instead of relying on one train/validation split, the script could use:

- repeated holdout,
- K-fold cross-validation,
- or nested cross-validation for smaller datasets.

This would reduce dependence on one fortunate or unlucky split.

---

### 4. Evaluate all equations in `model.equations_` on the validation set
Instead of selecting the max-`score` row first, the script could compute validation MSE for several candidate equations in the PySR hall of fame and keep the true validation-best equation.

This would align the internal selection step more closely with the final evaluation metric.

---

### 5. Expand the hyperparameter search space
Depending on the problem, it may be useful to tune additional PySR settings such as:

- unary operators,
- complexity settings,
- batching strategy,
- population migration behavior,
- loss definitions,
- stopping criteria,
- denoising or noise-robust settings.

This would make the analysis more complete.

---

### 6. Save Optuna studies to persistent storage
Using SQLite or another Optuna storage backend would allow:

- resuming interrupted searches,
- later analysis,
- reproducible experiment tracking.

---

### 7. Save structured results in addition to plain text logs
Right now, the results are written as formatted text. A useful extension would be to also save:

- CSV summaries,
- JSON trial metadata,
- or pandas DataFrames.

That would make downstream analysis and plotting easier.

---

### 8. Add error handling and dataset validation
Before fitting, the script could check for:

- missing files,
- empty datasets,
- non-numeric columns,
- NaNs or infinities,
- too-small sample sizes.

This would make the framework safer for batch experiments.

---

### 9. Include final plots
A very useful extension would be automatic visualization of:

- Pareto fronts,
- trial loss distributions,
- hyperparameter importance bars,
- runtime vs performance trade-offs.

This would make interpretation much easier.

---

### 10. Consider adaptive seed budgeting
Running 10 seeds per trial is scientifically useful but expensive.
A more advanced strategy could be:

- start with fewer seeds,
- identify promising trials,
- allocate more seeds only to the most competitive configurations.

This could preserve robustness while reducing total compute.

---

## Expected input format

Each CSV dataset should look like this conceptually:

| feature_1 | feature_2 | ... | target |
|----------|----------|-----|--------|
| x11      | x12      | ... | y1     |
| x21      | x22      | ... | y2     |
| ...      | ...      | ... | ...    |

The script interprets:

- all columns except the last as `X`,
- the last column as `y`.

---

## Expected output

The main output is a text log file containing:

- the name of each experiment,
- total Optuna runtime,
- all Pareto-optimal trials,
- best equation found for each Pareto trial,
- best equation loss,
- runtime of the best seed,
- hyperparameter importances for mean loss and standard deviation.

At the end, the script prints:

```python
All experiments complete! Check the 'all_experiments_results.txt' file.
```

---

## In one sentence

This script is a **robustness-aware symbolic regression tuning framework** that uses **Optuna + PySR** to search for hyperparameter configurations that produce equations which are both **accurate** and **stable across random seeds**.
