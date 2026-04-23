# Symbolic Regression in Physics: PySR vs MLP

Research project for the discovery of fundamental physical laws through **Symbolic Regression** (using the `PySR` Julia engine) and its comparison with universal approximators (**Multilayer Perceptron**).

## Objectives
1. Generate synthetic datasets of physical laws (Kepler, Coulomb, Harmonic Oscillator) with different levels of Gaussian noise.
2. Recover the exact analytical expression using genetic algorithms (`PySR`).
3. Evaluate the generalization capability of a Neural Network (MLP) versus the symbolic solution in extrapolation ranges.


## Prerequisites


### 1. LaTeX Distribution (System-wide)

This project requires a **LaTeX** distribution for mathematical typesetting in plots (`matplotlib`) and document generation. 
To ensure font consistency and package compatibility, an up-to-date distribution is required. You may choose between **TeX Live 2025** or **MiKTeX**.

### **Option A: TeX Live 2025 (Recommended)**
* **Fast Installation (PowerShell):**
    ```powershell
    winget install TeXLive.TeXLive --version 2025
    ```
* **Manual Installation:** Download from [TUG.org](https://tug.org/texlive/).

### **Option B: MiKTeX**
* **Fast Installation (PowerShell):**
    ```powershell
    winget install MiKTeX.MiKTeX
    ```
* **Manual Installation:** Download from [miktex.org](https://miktex.org/download).

---

> [!IMPORTANT]
> **Technical Note:** If you choose MiKTeX, ensure that "Always install missing packages on-the-fly" is enabled in the **MiKTeX Console**. This prevents compilation interruptions when handling documents with extensive dependencies.

### 2. Python Environment
Install the core libraries for physical computing and data analysis using the provided requirements file:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```  

## Repository Structure
- `src/`: Data generation scripts and model execution.
- `notebooks/`: Exploratory analysis and Pareto Frontier visualization.
- `data/`: Datasets generated for the experiments.

## Team Members
- [Jorge Acebes Hernández](https://github.com/JorgeAcebes)
- [Andrés López Serna](https://github.com/an-coder38)
- [Lorenzo Ji](https://github.com/Lorsimu)
