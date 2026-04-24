# Data — Synthetic Physics Datasets

This directory contains all synthetic datasets used to train and evaluate the models, as well as the out-of-distribution (OOD) test arrays.

---

## Dataset Overview

Each physical law is sampled at **10 000 points** under three independent Gaussian noise levels:

| Noise level | Relative σ | File suffix |
|---|---|---|
| No noise | 0 % | `_no_noise.csv` |
| Low noise | 1 % | `_low_noise.csv` |
| High noise | 10 % | `_high_noise.csv` |

This produces **27 CSV files** covering 9 laws × 3 noise levels.

---

## Datasets

| File prefix | Physical law | Target column | Input columns |
|---|---|---|---|
| `kepler` | Kepler's Third Law — *T² ∝ a³* | `T` | `a`|
| `coulomb` | Coulomb's Law — *F = kq₁q₂/r²* | `F` | `q1`, `q2`, `r` |
| `oscillator` | Harmonic Oscillator — *F = −kx* | `F` | `k`, `x` |
| `boltzmann_entropy` | Boltzmann Entropy — *S = k_B ln Ω* | `S` | `Omega` |
| `ideal_gas` | Ideal Gas Law — *PV = nRT* | `P` | `n`, `R`, `T`, `V` |
| `newton_cooling` | Newton's Law of Cooling | `dT_dt` | `T`, `T_env`, `k` |
| `projectile_range` | Projectile Range — *R = v²sin(2θ)/g* | `R` | `v`, `theta`, `g` |
| `radioactive_decay` | Radioactive Decay — *A = λN* | `A` | `lambda`, `N` |
| `time_dilation` | Special Relativity — Time Dilation | `t_prime` | `t`, `v`, `c` |

---

## File Format

Each CSV contains a header row. All columns except the last are input features; the last column is always the target variable.

```
feature_1,feature_2,...,target
x11,x12,...,y1
x21,x22,...,y2
...
```

The pipeline in `src/data/loader.py` reads any CSV that follows this convention automatically — no manual configuration required.

---

## `data_ood/` — Out-of-Distribution Arrays

The `data_ood/` subdirectory contains `.npz` files with test inputs **outside the training domain** for each physical law. These are used by `src/evaluate_model_ood.py` to assess model extrapolation.

| File | Contents |
|---|---|
| `kepler_ood_data.npz` | `X_ood`, `y_ood` arrays in physical space |
| `coulomb_ood_data.npz` | — |
| `oscillator_ood_data.npz` | — |
| `boltzmann_entropy_ood_data.npz` | — |
| `ideal_gas_ood_data.npz` | — |
| `newton_cooling_ood_data.npz` | — |
| `projectile_range_ood_data.npz` | — |
| `radioactive_decay_ood_data.npz` | — |
| `time_dilation_ood_data.npz` | — |

Load with:

```python
import numpy as np
data = np.load("data/data_ood/kepler_ood_data.npz")
X_ood, y_ood = data["X_ood"], data["y_ood"]
```
---

## Naming Convention

```
<law>_<noise_level>.csv
```

Examples: `kepler_no_noise.csv`, `coulomb_high_noise.csv`, `ideal_gas_low_noise.csv`.
