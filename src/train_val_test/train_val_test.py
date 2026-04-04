# The idea is that in the future this would be our main script, in which we will be doing the training-validation-test.
# Right now, I'll be dropping some sparse code that will be useful in the future:

import os
import glob
import pandas as pd
import numpy as np

# Regresores Simbólicos
import feyn # QLattice
from gplearn.genetic import SymbolicRegressor
import pysindy as ps
from gplearn.functions import make_function


# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../..", "data"))
RESULTS_BASE = os.path.abspath(os.path.join(BASE_DIR, "../../results/all_models"))

RES_QLATTICE = os.path.join(RESULTS_BASE, "qlattice")
RES_GPLEARN = os.path.join(RESULTS_BASE, "gplearn")
RES_PYSINDY = os.path.join(RESULTS_BASE, "pysindy")


