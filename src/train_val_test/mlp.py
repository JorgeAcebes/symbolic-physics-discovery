"""
MLP con PyTorch para ajustar leyes físicas
usando datasets generados en /data

Incluye:
- Split train / val / test
- Normalización sin data leakage
- DataLoader
- Evaluación correcta
"""

# =========================
# IMPORTS
# =========================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
#añadir para polyfit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Para reproducibilidad completa (puede ser más lento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# RUTAS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../..", "data"))

# =========================
# DATASET
# =========================

def load_dataset(filename, input_cols, target_col):

    path = os.path.join(DATA_DIR, filename)
    print(f"Cargando: {path}")

    df = pd.read_csv(path)

    X = df[input_cols].values
    y = df[target_col].values.reshape(-1, 1)

    return X, y


# =========================
# MODELO
# =========================

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(), # Imponemos continuidad en las derivadas superiores, a diferencia de ReLU
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# MÉTRICAS
# =========================

def mae_torch(preds, targets):
    return torch.mean(torch.abs(preds - targets))

def evaluate_model(model, data_loader, loss_fn, scaler_y=None):
    model.eval()
    total_loss, total_mae, total_count = 0.0, 0.0, 0

    with torch.no_grad():
        for x, y in data_loader:
            x_dev = x.to(device)
            preds = model(x_dev).cpu().numpy()
            y_true = y.numpy()

            # Recuperar unidades físicas reales
            if scaler_y is not None:
                preds = scaler_y.inverse_transform(preds)
                y_true = scaler_y.inverse_transform(y_true)

            preds_t = torch.tensor(preds, dtype=torch.float32)
            y_t = torch.tensor(y_true, dtype=torch.float32)

            loss = loss_fn(preds_t, y_t)
            mae = torch.mean(torch.abs(preds_t - y_t))

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae.item() * batch_size
            total_count += batch_size

    return total_loss / total_count, total_mae / total_count


# =========================
# ENTRENAMIENTO
# =========================

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, scaler_y=None):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"loss": [], "val_loss": []}

    best_val_loss = float("inf")
    best_state = None

    # Variables iniciales de early stopping
    patience = 5
    epochs_no_improve = 0

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        total_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            preds = model(x)
            loss = loss_fn(preds, y)

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            total_count += batch_size

        train_loss = running_loss / total_count
        val_loss, _ = evaluate_model(model, val_loader, loss_fn, scaler_y=scaler_y)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}: train={train_loss:.10f} val={val_loss:.10f}")

        # early stopping básico
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping activado en la época {epoch+1}")
                break
        model.load_state_dict(best_state)

    return model, history


# =========================
# PIPELINE DE DATOS
# =========================

def prepare_data(X, y, batch_size=32):

    # SPLIT 60/20/20
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 0.2 para test
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=1) # 0.25*0.8 para validation

    # SCALING (solo train → NO DATA LEAKAGE)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    # TENSORES
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # DATALOADERS
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    print("Train size:", len(train_loader.dataset))
    print("Val size:", len(val_loader.dataset))
    print("Test size:", len(test_loader.dataset))

    return train_loader, val_loader, test_loader, scaler_y

# =========================
# EXPERIMENTO
# =========================

def run_experiment(filename, input_cols, target_col, name):

    X, y = load_dataset(filename, input_cols, target_col)

    train_loader, val_loader, test_loader, scaler_y = prepare_data(X, y)

    model = MLP(input_dim=X.shape[1])

    model, history = train_model(model, train_loader, val_loader, scaler_y=scaler_y)

    # evaluación final
    test_loss, test_mae = evaluate_model(model, test_loader, nn.MSELoss(), scaler_y=scaler_y)

    print("\n==============================")
    print(f"{name}")
    print("==============================")
    print(f"Test Loss: {test_loss:.10f}")
    print(f"Test MAE: {test_mae:.10f}")

    # =========================
    # POLYNOMIAL REGRESSION (MULTIVARIABLE)
    # =========================
    run_polynomial_regression_experiment(
        X, y,
        degree=3,
        name=name + " PolyReg"
    )

    # gráfica
    plt.figure()
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.legend()
    plt.title(name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    # plt.show()

#Función que hace un ajuste polinómico de los datos, para comparar con MLP
def run_polynomial_regression_experiment(X, y, degree=2, name="PolyReg", plot=False):

    y = y.flatten()

    # =========================
    # SPLIT
    # =========================
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=1
    )

    # =========================
    # PIPELINE
    # =========================
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg", LinearRegression())
    ])

    # FIT SOLO EN TRAIN
    model.fit(X_train, y_train)

    # =========================
    # EVALUACIÓN
    # =========================
    def evaluate(X_split, y_split):
        y_pred = model.predict(X_split)
        mse = mean_squared_error(y_split, y_pred)
        mae = mean_absolute_error(y_split, y_pred)
        r2 = r2_score(y_split, y_pred)
        return mse, mae, r2

    train_metrics = evaluate(X_train, y_train)
    val_metrics = evaluate(X_val, y_val)
    test_metrics = evaluate(X_test, y_test)

    print("\n==============================")
    print(f"{name} (grado {degree})")
    print("==============================")

    print("Train  -> MSE: {:.3e} | MAE: {:.3e} | R2: {:.4f}".format(*train_metrics))
    print("Val    -> MSE: {:.3e} | MAE: {:.3e} | R2: {:.4f}".format(*val_metrics))
    print("Test   -> MSE: {:.3e} | MAE: {:.3e} | R2: {:.4f}".format(*test_metrics))

    # =========================
    # MOSTRAR ECUACIÓN
    # =========================
    poly = model.named_steps["poly"]
    reg = model.named_steps["reg"]

    feature_names = poly.get_feature_names_out()

    print("\nEcuación aproximada:")
    terms = []
    for coef, name_feat in zip(reg.coef_, feature_names):
        terms.append(f"{coef:.3e}*{name_feat}")

    equation = " + ".join(terms)
    equation += f" + {reg.intercept_:.3e}"

    print("y =", equation)

    # =========================
    # PLOT SOLO SI 1D
    # =========================
    if plot and X.shape[1] == 1:
        x_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
        y_plot = model.predict(x_plot)

        plt.figure()
        plt.scatter(X_train, y_train, alpha=0.4, label="Train")
        plt.scatter(X_val, y_val, alpha=0.4, label="Val")
        plt.scatter(X_test, y_test, alpha=0.4, label="Test")
        plt.plot(x_plot, y_plot, label=f"Poly grado {degree}", linewidth=2)

        plt.title(name)
        plt.legend()
        plt.grid()
        plt.show()

    return model, {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics
    }

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    set_seed(42)

    run_experiment("oscillator_no_noise.csv", ["x"], "F", "Oscilador")
    run_experiment("kepler_no_noise.csv", ["r"], "T", "Kepler")
    run_experiment("coulomb_no_noise.csv", ["q1", "q2", "r"], "F", "Coulomb")
    run_experiment("ideal_gas_no_noise.csv", ["n", "T", "V"], "P", "Gas Ideal")