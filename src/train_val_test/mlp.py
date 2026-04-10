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
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# MÉTRICAS
# =========================

def mae_torch(preds, targets):
    return torch.mean(torch.abs(preds - targets))


def evaluate_model(model, data_loader, loss_fn):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)
            loss = loss_fn(preds, y)
            mae = mae_torch(preds, y)

            batch_size = x.size(0)

            total_loss += loss.item() * batch_size
            total_mae += mae.item() * batch_size
            total_count += batch_size

    return total_loss / total_count, total_mae / total_count


# =========================
# ENTRENAMIENTO
# =========================

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"loss": [], "val_loss": []}

    best_val_loss = float("inf")
    best_state = None

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
        val_loss, _ = evaluate_model(model, val_loader, loss_fn)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")

        # early stopping básico
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    return model, history


# =========================
# PIPELINE DE DATOS
# =========================

def prepare_data(X, y, batch_size=32):

    # SPLIT 60/20/20
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=1)

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

    return train_loader, val_loader, test_loader


# =========================
# EXPERIMENTO
# =========================

def run_experiment(filename, input_cols, target_col, name):

    X, y = load_dataset(filename, input_cols, target_col)

    train_loader, val_loader, test_loader = prepare_data(X, y)

    model = MLP(input_dim=X.shape[1])

    model, history = train_model(model, train_loader, val_loader)

    # evaluación final
    test_loss, test_mae = evaluate_model(model, test_loader, nn.MSELoss())

    print("\n==============================")
    print(f"{name}")
    print("==============================")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    # gráfica
    plt.figure()
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.legend()
    plt.title(name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    run_experiment("oscillator_no_noise.csv", ["x"], "F", "Oscilador")
    run_experiment("kepler_no_noise.csv", ["r"], "T", "Kepler")
    run_experiment("coulomb_no_noise.csv", ["q1", "q2", "r"], "F", "Coulomb")
    run_experiment("ideal_gas_no_noise.csv", ["n", "T", "V"], "P", "Gas Ideal")