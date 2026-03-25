# Symbolic Regression in Physics: PySR vs MLP

Research project for the discovery of fundamental physical laws through **Symbolic Regression** (using the `PySR` Julia engine) and its comparison with universal approximators (**Multilayer Perceptron**).

## Objectives
1. Generate synthetic datasets of physical laws (Kepler, Coulomb, Harmonic Oscillator) with different levels of Gaussian noise.
2. Recover the exact analytical expression using genetic algorithms (`PySR`).
3. Evaluate the generalization capability of a Neural Network (MLP) versus the symbolic solution in extrapolation ranges.

## Requirements
- **Julia v1.10+**: Symbolic computation engine.
- **Python 3.9+**: User interface and neural networks.
- **PySR**: Symbolic regression library.

## Repository Structure
- `src/`: Data generation scripts and model execution.
- `notebooks/`: Exploratory analysis and Pareto Frontier visualization.
- `data/`: Datasets generated for the experiments.

## Team Members
- Jorge Acebes Hernández
- Andrés López Serna
- Lorenzo Ji
