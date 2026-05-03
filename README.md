# Symbolic Physics Discovery

> **Benchmarking symbolic regression against neural networks for the automated recovery of fundamental physical laws from noisy data.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![PySR](https://img.shields.io/badge/PySR-1.5-green)](https://github.com/MilesCranmer/PySR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Physical Laws](#physical-laws)
- [Models](#models)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Run the full experiment pipeline](#1-run-the-full-experiment-pipeline)
  - [2. Hyperparameter optimisation with Optuna](#2-hyperparameter-optimisation-with-optuna)
  - [3. Out-of-distribution evaluation](#3-out-of-distribution-evaluation)
  - [4. Web interface](#4-web-interface)
- [Results](#results)
- [Adding a New Model](#adding-a-new-model)
- [Team](#team)

---

## Overview

This project investigates whether **symbolic regression** can automatically rediscover exact analytical expressions for physical laws from synthetic, noisy datasets — and how it compares to **universal approximators** (MLPs) in terms of predictive accuracy and extrapolation capability.

The core pipeline:

1. **Generate** synthetic datasets for 9 physical laws under 3 noise levels (no noise, low noise σ = 1 %, high noise σ = 10 %).
2. **Train** 8 models — symbolic regressors and neural networks — on each dataset.
3. **Evaluate** every model in the original physical space using MSE and MAE.
4. **Test generalisation** by probing each model on out-of-distribution (OOD) inputs outside the training domain.
5. **Visualise** residuals, convergence curves, and discovered equations in a web dashboard or via static plots.

All experiments are **fully reproducible** (seed 42 throughout) and metrics are computed in the **original physical space** after inverse-transforming standardised predictions.

---

## Physical Laws

| Dataset file prefix | Law | Target variable |
|---|---|---|
| `coulomb` | Coulomb's Law | Force $F$ |
| `oscillator` | Harmonic Oscillator | Force $F$ |
| `kepler` | Kepler's Third Law | Period *T* |
| `ideal_gas` | Ideal Gas Law | Pressure $P$ |
| `projectile_range` | Projectile Range | Range $R$ |
| `time_dilation` | Time Dilation | Dilated time $t'$ |
| `radioactive_decay` | Radioactive Decay | Number of Particles $N$ |
| `newton_cooling` | Newton's Law of Cooling | Rate d*T*/d*t* |
| `boltzmann_entropy` | Boltzmann Entropy | Entropy $S$ |


Each law is provided at **three noise levels**: `no_noise`, `low_noise`, `high_noise` (27 CSV files total).

---

## Models

| Model key | Type | Library |
|---|---|---|
| `MLP_Standard` | Dense MLP (64→64→1, SiLU) | PyTorch |
| `MLP_Sparse` | Narrow MLP (16→16→1, Tanh) + L1 | PyTorch |
| `MLP_Dropout` | MC-Dropout MLP (32→32→1, ReLU, p=0.2) | PyTorch |
| `Polynomial` | Polynomial regression (latent space) | scikit-learn |
| `PySR` | Genetic symbolic regression (Julia engine) | PySR |
| `GPLearn` | Genetic programming | gplearn |
| `PySINDy` | Sparse identification of nonlinear dynamics | PySINDy |
| `QLattice` | Feyn graph-based SR | Feyn |

---

## Repository Structure

```
symbolic-physics-discovery/
│
├── data/                        # Synthetic CSV datasets (27 files + OOD .npz)
│   ├── data_ood/                # Out-of-distribution test arrays (.npz)
│   └── README.md
|
├── docs/                        # Project report and presentation .tex files and compiled pdf
│   ├── paper/                   # Technical report and project documentation.
│   ├── presentation/            # Project presentation
│   └── README.md
│
├── src/                         # Core experiment pipeline
│   ├── main.py                  # Orchestrator — runs all models on all datasets
│   ├── run_optuna.py            # Optuna hyperparameter search
│   ├── evaluate_model_ood.py    # OOD evaluation script (same for evaluate_model_ood_graphs.py)
│   ├── table_equations_ood.py   # Table generator with all equations + ood MSE
│   ├── aggregate_hyperparams.py # Aggregation of Optuna results
│   ├── data/
│   │   ├── loader.py            # PhysicalDataset — data ingestion & preprocessing
│   │   └── data_generator.py    # Synthetic dataset generator
│   ├── models/
│   │   ├── base.py              # Abstract PhysicalModel interface
│   │   ├── mlp.py               # MLP wrappers (Standard, Sparse, Dropout)
│   │   ├── pysr_sr.py           # PySR wrapper
│   │   ├── gplearn_sr.py        # GPLearn wrapper
│   │   ├── pysindy_sr.py        # PySINDy wrapper
│   │   ├── qlattice_sr.py       # QLattice / Feyn wrapper
│   │   └── polynomial.py        # Polynomial regression wrapper
│   └── utils/
│       ├── metrics.py           # MSE/MAE in physical space
│       ├── io.py                # Result export, residual plots, loss curves
│       ├── utils.py             # Plot style helpers
│       └── weights_dir.py       # Model weight persistence
│
├── results/                     # Experiment outputs (plots, metrics, weights)
│   ├── MLP_Standard/
│   ├── MLP_Sparse/
│   ├── MLP_Dropout/
│   ├── Polynomial/
│   ├── PySR/
│   ├── GPLearn/
│   ├── PySINDy/
│   ├── QLattice/
│   ├── all_models/              # Cross-model comparison reports
│   ├── datasets_plots/          # Dataset visualisations
│   ├── optuna_hyperparams/      # Optuna study outputs
│   ├── results_ood/             # OOD evaluation outputs
│   ├── weights/                 # Serialised model weights
│   └── README.md
│
├── web/                         # Flask dashboard for interactive experiments
│   ├── app.py                   # Flask server (API + HTML)
│   ├── runner.py                # Adapter between web UI and src/
│   ├── templates/index.html     # Main UI
│   ├── static/                  # CSS + JS assets
│   └── README.md
│
├── docs/                        # LaTeX source for the technical report
│   └── README.md
│
├── requirements.txt
└── LICENSE
```

---

## Prerequisites

### System — LaTeX distribution

Required for mathematical typesetting in `matplotlib` figures. Install **one** of the following:

**Option A — TeX Live 2025 (recommended)**

```powershell
# Windows (PowerShell)
winget install TeXLive.TeXLive --version 2025
```

**Option B — MiKTeX**

```powershell
# Windows (PowerShell)
winget install MiKTeX.MiKTeX
```

> **Note for MiKTeX users:** Enable *"Always install missing packages on-the-fly"* in the MiKTeX Console to avoid compilation interruptions.

**Linux / macOS**

```bash
# TeX Live (recommended)
sudo apt install texlive-full        # Debian / Ubuntu
brew install --cask mactex           # macOS
```

### Julia (required by PySR)

PySR uses a Julia backend managed automatically by `juliacall`. Julia is downloaded on first use; no manual installation is needed. Ensure you have a working internet connection on the first run.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/JorgeAcebes/symbolic-physics-discovery.git
cd symbolic-physics-discovery

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
# Windows
.\venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt
```

> **GPU support:** PyTorch will use CUDA automatically if a compatible GPU is available. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

---

## Usage

All scripts must be run from inside the `src/` directory so that relative imports resolve correctly.

```bash
cd src
```

### 1. Run the full experiment pipeline

Edit the `models_to_run` list in `src/main.py` to select which models to train:

```python
models_to_run = [
    "MLP_Standard",
    "MLP_Sparse",
    "MLP_Dropout",
    "Polynomial",
    "PySR",
    "GPLearn",
    "PySINDy",
    "QLattice",
]
```

Then run:

```bash
python main.py
```

Results — metrics, residual plots, convergence curves, and discovered equations — are written to `results/<ModelName>/`.

A cross-model summary report is generated automatically at the end under `results/all_models/`.

### 2. Hyperparameter optimisation with Optuna

```bash
python run_optuna.py
```

Runs an Optuna study for the models specified inside `run_optuna.py`. Results (best hyperparameters, MSE distributions) are written to `results/optuna_hyperparams/`.

### 3. Out-of-distribution evaluation

```bash
python evaluate_model_ood.py
```

Loads saved model weights from `results/weights/` and evaluates each model on the OOD test arrays in `data/data_ood/`. Outputs are written to `results/results_ood/`.

### 4. Web interface

```bash
cd ../web
python app.py
```

Open [http://localhost:5050](http://localhost:5050) in your browser.

The dashboard lets you upload a custom CSV, choose models, adjust hyperparameters, run experiments, and inspect results — all without touching source code.

---

## Results

Pre-computed results for all 27 datasets × 8 models are included in the `results/` directory:

- **Residual plots** — predicted vs. true values in physical space.
- **Convergence curves** — training / validation loss per epoch (MLP models).
- **Discovered equations** — symbolic expressions recovered by PySR, GPLearn, PySINDy, and QLattice.
- **OOD performance** — extrapolation metrics outside the training domain.
- **Cross-model gallery** — `gallery_residuals.png` and `gallery_loss.png` with all models side by side.

---

## Adding a New Model

All models share the `PhysicalModel` abstract interface. To plug in a new algorithm:

1. Create `src/models/my_model.py` — subclass `PhysicalModel`, implement `fit()` and `predict()`.
2. Set `self.equation` to the discovered expression string inside `fit()`.
3. Return `predict()` as a `numpy` array of shape `[N, 1]`.
4. Import and register the wrapper in the `models` dict inside `src/main.py`.

Full instructions with code templates are in [`src/README.md`](./src/README.md).

---

## Team

| Name | GitHub |
|---|---|
| Jorge Acebes Hernández | [@JorgeAcebes](https://github.com/JorgeAcebes) |
| Andrés López Serna | [@an-coder38](https://github.com/an-coder38) |
| Lorenzo Ji | [@Lorsimu](https://github.com/Lorsimu) |

---

*Licensed under the [MIT License](./LICENSE).*
