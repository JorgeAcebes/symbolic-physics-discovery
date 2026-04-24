# Source Code — Experiment Pipeline (`src/`)

This directory contains the full experiment pipeline for symbolic physics discovery: data ingestion, model training, evaluation, OOD testing, and hyperparameter optimisation.

---

## Directory Structure

```
src/
├── main.py                  ← Orchestrator: runs all models across all datasets
├── run_optuna.py            ← Optuna hyperparameter search
├── evaluate_model_ood.py    ← Out-of-distribution evaluation
├── aggregate_hyperparams.py ← Aggregates and summarises Optuna outputs
│
├── data/
│   ├── loader.py            ← PhysicalDataset — single source of truth for data
│   └── data_generator.py    ← Synthetic dataset generator (10 000 pts × 3 noise levels)
│
├── models/
│   ├── base.py              ← Abstract PhysicalModel contract
│   ├── mlp.py               ← MLP wrappers: Standard, Sparse, MC-Dropout
│   ├── pysr_sr.py           ← PySR symbolic regression wrapper
│   ├── gplearn_sr.py        ← GPLearn genetic programming wrapper
│   ├── pysindy_sr.py        ← PySINDy sparse dynamics wrapper
│   ├── qlattice_sr.py       ← QLattice / Feyn wrapper
│   └── polynomial.py        ← Polynomial regression wrapper
│
└── utils/
    ├── metrics.py           ← MSE and MAE in original physical space
    ├── io.py                ← Residual plots, loss curves, result export
    ├── utils.py             ← Plot style configuration (LaTeX, Latin Modern Roman)
    └── weights_dir.py       ← Model serialisation helpers
```

---

## Module Descriptions

### `data/loader.py` — `PhysicalDataset`

The **single source of truth** for all data operations. Responsibilities:

- Reads any CSV whose last column is the target variable.
- Applies a **z-score standardisation** (`StandardScaler`) to both features and target independently.
- Exposes four views of the data:
  - `get_latent_arrays()` → normalised train/val/test numpy arrays (used by MLP, Polynomial).
  - `get_physical_arrays()` → original physical-space arrays (used by symbolic regressors).
  - `get_dataloaders()` → PyTorch `DataLoader` objects for mini-batch training.
- Extracts `feature_names` dynamically from the CSV header.

The train / validation / test split is **fixed** (seed 42) to avoid data leakage between `main.py` and `run_optuna.py`.

### `data/data_generator.py`

Generates all 27 synthetic CSVs and 9 OOD `.npz` arrays from closed-form physical expressions. Configured with:

- `N_SAMPLES = 10000` points per dataset.
- `NOISE_LEVELS = {"no_noise": 0.0, "low_noise": 0.01, "high_noise": 0.1}` (relative Gaussian noise).
- `numpy.random.seed(1)` for reproducibility.

### `models/base.py` — `PhysicalModel`

Abstract base class that enforces a uniform interface across all models:

```python
class PhysicalModel(ABC):
    equation: str          # Discovered expression (string) — set in fit()
    history: dict          # {"train_loss": [...], "val_loss": [...]}

    @abstractmethod
    def fit(self, *args, **kwargs): ...

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...  # shape [N, 1]
```

### `models/mlp.py` — MLP Wrappers

Three architectures sharing the `MLPWrapper` interface:

| Key | Architecture | Activation | Regularisation |
|---|---|---|---|
| `MLP_Standard` | 64 → 64 → 1 | SiLU | — |
| `MLP_Sparse` | 16 → 16 → 1 | Tanh | L1 (Lasso) |
| `MLP_Dropout` | 32 → 32 → 1 | ReLU | MC-Dropout p=0.2 |

All MLPs operate in the **standardised latent space** and are trained with Adam (lr ≈ 9.7 × 10⁻⁴, 245 epochs by default).

### `utils/metrics.py`

Computes MSE and MAE **after** inverse-transforming predictions back to the original physical domain, ensuring metrics are comparable across models regardless of input scale.

### `utils/io.py`

Handles all output to disk:

- `save_experiment_results()` — writes metrics and equation to a JSON / text log under `results/<ModelName>/`.
- `plot_residual_analysis()` — residual scatter plot at 300 DPI with LaTeX typesetting.
- `report_all_models()` — generates a cross-model summary gallery under `results/all_models/`.
- `save_model_weights()` — serialises model state for OOD evaluation.

---

## Running the Experiments

All scripts must be executed from inside `src/`:

```bash
cd src

# Full benchmark across all datasets and selected models
python main.py

# Hyperparameter optimisation
python run_optuna.py

# OOD evaluation (requires saved weights)
python evaluate_model_ood.py

# Aggregate Optuna outputs into a summary table
python aggregate_hyperparams.py
```

To restrict the run to a subset of datasets, set `datasets_manuales = 1` in `main.py` and edit the manual list:

```python
datasets_info = [
    {"file": "oscillator_no_noise.csv", "target": "F"},
    {"file": "kepler_no_noise.csv",     "target": "T"},
]
```

---

## Adding a New Model

Follow these four steps to integrate any new regression algorithm while preserving the comparative framework.

### Step 1 — Create the wrapper

```python
# src/models/my_model.py
import numpy as np
from models.base import PhysicalModel
from my_library import MyAlgorithm

class MyModelWrapper(PhysicalModel):
    def __init__(self, feature_names=None, my_param=10):
        super().__init__()
        self.feature_names = feature_names
        self.model = MyAlgorithm(param=my_param)
```

### Step 2 — Implement `fit()`

Assign the discovered symbolic expression (as a string) to `self.equation` before returning.

```python
def fit(self, X_train, y_train):
    self.model.fit(X_train, y_train)
    self.equation = self.model.get_expression()   # adapt to your library
    # Optional: self.history["train_loss"] = self.model.loss_curve_
    return self
```

### Step 3 — Implement `predict()`

Return a numpy array of shape `[N, 1]`.

```python
def predict(self, X):
    return np.array(self.model.predict(X)).reshape(-1, 1)
```

### Step 4 — Register in `main.py`

```python
from models.my_model import MyModelWrapper

models = {
    ...
    "MyModel": MyModelWrapper(feature_names=dataset.feature_names),
}
```

Add `"MyModel"` to `models_to_run` to include it in the next run.

---

## Reproducibility

Global seed is set at the top of `main.py` and `evaluate_model_ood.py`:

```python
set_seed(42)   # random, numpy, torch, cuda
```

PySR uses `deterministic=True` and `random_state=seed` per trial.
