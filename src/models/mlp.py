# Imports para el procesador de tensores de Pytorch + el modelo base
import torch
import torch.nn as nn
from models.base import PhysicalModel

# Definimos device como el cerebro donde se procesarán los datos. Útil cuando se tienen GPUs disponibles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# IMPLEMENTACIÓN DE 3 CLASES DE MLPs: Standard, Sparse y Dropout

# Definimos una clase para una red neuronal densa: 
# Input_dims -> 64 -> 64 -> 1
# Se emplea activación SiLU(x) = x * sigma(x). Garantiza derivadas funcionales...
# ...continuas (i.e. previene singularidades en los gradientes)
class StandardMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.SiLU(), 
            nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)


# Definimos una clase para una red neuronal con una arquitectura de capas más estrecha
# Input dims -> 16 -> 16 -> 1
# Se emplea la activación tanh(x)
# Se le aplicará un regularizador L1 (Lasso) que introduce muchos ceros en los tensores de peso
class SparseMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.act = nn.Tanh()
    def forward(self, x):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


# Definimos una clase para una red neuronal con dropout monte carlo del 20% de las neuronas
# Input dim -> 32 (25 en dropout) -> 32 (25 en dropout) -> 1
# Se emplea activación ReLU
class MCDropoutMLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(32, 32)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(32, 1)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.fc3(self.drop2(self.act(self.fc2(self.drop1(self.act(self.fc1(x)))))))
    
    
# Wrapper (envoltorio) donde se definen los hiperparámetros comunes
class MLPWrapper(PhysicalModel):
    def __init__(self, input_dim, model_type='standard', epochs=245, lr=0.00097118, l1_alpha=2.3222e-05, mc_samples=94, **kwargs):
        super().__init__()
        self.model_type = model_type # nombre del MLP
        self.epochs = epochs 
        self.lr = lr
        self.l1_alpha = l1_alpha
        self.mc_samples = mc_samples # Samples para monte carlo
        
        if model_type == 'sparse': self.model = SparseMLP(input_dim).to(device)
        elif model_type == 'dropout': self.model = MCDropoutMLP(input_dim).to(device)
        else: self.model = StandardMLP(input_dim).to(device)

    # Definimos la función fit. Emplearemos el optimizador adam 
    def fit(self, train_loader, val_loader=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss() # Emplearemos MSE como función de pérdida para optimmizar
        
        for epoch in range(self.epochs):
            self.model.train() # Iniciamos el entrenemiento
            epoch_train_loss = 0.0
            
            for x, y in train_loader: # Ciclo por lotes 
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad() # Limpieza de gradiente
                
                preds = self.model(x)
                mse_loss = loss_fn(preds, y) # Métrica física estricta
                loss = mse_loss
                
                # Regularización L_1 global para SPARSE (todas las matrices sinápticas)
                if self.model_type == 'sparse':
                    l1_reg = sum(p.abs().sum() for p in self.model.parameters() if p.requires_grad)
                    loss += self.l1_alpha * l1_reg
                    
                loss.backward() # Cálculo de los gradientes 
                optimizer.step() # Actualización de los pesos.
                
                # Registro libre de entropía topológica (escalamos el loss SIN L1 y multiplicamos por el tamaño del batch)
                epoch_train_loss += mse_loss.item() * x.size(0) # Suma total del loss para este batch
            
            self.history["train_loss"].append(epoch_train_loss / len(train_loader.dataset)) # Guardamos el loss medio en esta época (ya con todos los batches) 
            
            if val_loader: # Si tenemos para validación
                self.model.eval() # Lo ponemos en modo evaluación para que no se actualice
                epoch_val_loss = 0.0
                with torch.no_grad(): # No queremos que compute gradientes
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        epoch_val_loss += loss_fn(self.model(x), y).item() * x.size(0)
                self.history["val_loss"].append(epoch_val_loss / len(val_loader.dataset))
                
        self.equation = f"Red Neuronal ({self.model_type})" 
        return self

    def predict(self, X, return_std=False):
        if not isinstance(X, torch.Tensor): X = torch.tensor(X, dtype=torch.float32)
        X = X.to(device)
        
        if self.model_type == 'dropout':
            self.model.train() # Todavía mantenemos dropout, aunqué esté en modo validación. 
            with torch.no_grad(): # Pese a activar dropout, no queremos cómputo de gradiente.
                preds = torch.stack([self.model(X) for _ in range(self.mc_samples)]) # Ejecuta el modelo mc_samples veces
                # Apila el resultado en un tensor de predicciones M x N x 1 (nº muestras MC x nº de samples x 1)
            mean_pred = preds.mean(dim=0).cpu().numpy() # Realizamos la media sobre las muestras de MC --> (nº samples x 1, i.e. predicción de y para cada entrada)
            # En main está diseñado para realizar flatten, no hay problemas de dimensiones.
            if return_std: return mean_pred, preds.std(dim=0).cpu().numpy()
            return mean_pred # Devolvemos la media
        else:
            self.model.eval() # En el resto de casos sí que lo ponemos en modo evaluación, y no actualizamos gradientes
            with torch.no_grad(): return self.model(X).cpu().numpy()

        # Siempre lo pasamos a la CPU y lo transformamos en un ndarray

    def get_weights(self):
        """Devuelve el state_dict serializable capa a capa."""
        return {
            name: tensor.cpu().numpy()
            for name, tensor in self.model.state_dict().items()
        }