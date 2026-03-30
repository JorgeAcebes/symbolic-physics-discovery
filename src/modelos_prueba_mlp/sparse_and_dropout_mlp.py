import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de directorios
DATA_DIR = "data"
RES_SPARSE = "results/testing_mlp/sparse_mlp"
RES_DROPOUT = "results/testing_mlp/dropout_mlp"
os.makedirs(RES_SPARSE, exist_ok=True)
os.makedirs(RES_DROPOUT, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Arquitecturas Topológicas
# -------------------------------------------------------------------------

class SparseMLP(nn.Module):
    def __init__(self, input_dim):
        super(SparseMLP, self).__init__()
        # Mantenemos una arquitectura minimalista para forzar la interpretabilidad
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

class MCDropoutMLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(MCDropoutMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(32, 32)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(32, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.act(self.fc2(x)))
        return self.fc3(x)

# -------------------------------------------------------------------------
# 2. Rutinas de Entrenamiento Riguroso
# -------------------------------------------------------------------------

def train_sparse_mlp(X, y, dataset_name, epochs=1000, lr=1e-3, alpha=1e-3):
    model = SparseMLP(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        mse_loss = criterion(preds, y)
        
        # Penalización L1 estricta sobre la primera capa (Feature Selection empírico)
        l1_penalty = torch.sum(torch.abs(model.fc1.weight))
        loss = mse_loss + alpha * l1_penalty
        
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    # Análisis de pesos sinápticos post-entrenamiento
    weights = model.fc1.weight.detach().numpy()
    importance = np.sum(np.abs(weights), axis=0)
    
    plt.figure(figsize=(6, 4))
    plt.bar(range(X.shape[1]), importance, color='black')
    plt.title(f"Importancia L1 - {dataset_name}")
    plt.xlabel("Índice de Característica de Entrada")
    plt.ylabel(r"Magnitud del Peso $\sum |W_{1,j}|$")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_SPARSE, f"{dataset_name}_weights.png"), dpi=300)
    plt.close()
    
    return model

def train_mc_dropout(X, y, dataset_name, epochs=1000, lr=1e-3):
    model = MCDropoutMLP(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train() # Asegura que el dropout está activo
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
    # Inferencia Bayesiana: Muestreo de Monte Carlo
    model.train() # Mantenemos Dropout en inferencia para cuantificar varianza
    num_samples = 100
    with torch.no_grad():
        predictions = torch.stack([model(X) for _ in range(num_samples)])
    
    mean_preds = predictions.mean(dim=0).squeeze().numpy()
    std_preds = predictions.std(dim=0).squeeze().numpy()
    
    # Análisis de dispersión
    plt.figure(figsize=(6, 4))
    plt.scatter(y.numpy(), mean_preds, s=5, c='black', alpha=0.5)
    plt.errorbar(y.numpy(), mean_preds, yerr=std_preds, fmt='none', ecolor='red', alpha=0.2)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.title(f"Inferencia MC-Dropout - {dataset_name}")
    plt.xlabel(r"Objetivo Real $y$")
    plt.ylabel(r"Predicción $\mathbb{E}[\hat{y}] \pm \sigma$")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DROPOUT, f"{dataset_name}_uncertainty.png"), dpi=300)
    plt.close()

# -------------------------------------------------------------------------
# 3. Ejecución Principal
# -------------------------------------------------------------------------

def main():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("Fallo de I/O: No se encontraron archivos CSV en el directorio.")
        return

    for file in csv_files:
        dataset_name = os.path.basename(file).replace('.csv', '')
        
        # Extracción tensorial
        df = pd.read_csv(file)
        # Asumimos formulación estándar: columnas [0, N-1] features, columna N target
        X_tensor = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        y_tensor = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)
        
        # Normalización Z-score (crítico para la convergencia de la norma L1 y gradientes)
        X_mean, X_std = X_tensor.mean(dim=0), X_tensor.std(dim=0)
        y_mean, y_std = y_tensor.mean(), y_tensor.std()
        
        X_norm = (X_tensor - X_mean) / (X_std + 1e-8)
        y_norm = (y_tensor - y_mean) / (y_std + 1e-8)
        
        print(f"Procesando {dataset_name} | Topología de entrada: {X_norm.shape}")
        
        # Entrenamiento y evaluación
        train_sparse_mlp(X_norm, y_norm, dataset_name)
        train_mc_dropout(X_norm, y_norm, dataset_name)

if __name__ == "__main__":
    main()