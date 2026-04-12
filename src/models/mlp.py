import torch
import torch.nn as nn
from models.base import PhysicalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StandardMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.SiLU(), 
            nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

class SparseMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.act = nn.Tanh()
    def forward(self, x):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))

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

class MLPWrapper(PhysicalModel):
    def __init__(self, input_dim, model_type='standard', epochs=50, lr=1e-3, l1_alpha=1e-3):
        super().__init__()
        self.model_type = model_type
        self.epochs = epochs
        self.lr = lr
        self.l1_alpha = l1_alpha
        
        if model_type == 'sparse': self.model = SparseMLP(input_dim).to(device)
        elif model_type == 'dropout': self.model = MCDropoutMLP(input_dim).to(device)
        else: self.model = StandardMLP(input_dim).to(device)

    def fit(self, train_loader, val_loader=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                preds = self.model(x)
                loss = loss_fn(preds, y)
                
                # Penalización L1 para forzar dispersión en pesos sinápticos
                if self.model_type == 'sparse':
                    loss += self.l1_alpha * torch.sum(torch.abs(self.model.fc1.weight))
                    
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * x.size(0)
            
            self.history["train_loss"].append(epoch_train_loss / len(train_loader.dataset))
            
            if val_loader:
                self.model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        epoch_val_loss += loss_fn(self.model(x), y).item() * x.size(0)
                self.history["val_loss"].append(epoch_val_loss / len(val_loader.dataset))
                
        self.equation = f"Red Neuronal ({self.model_type})" 
        return self

    def predict(self, X, return_std=False, mc_samples=100):
        if not isinstance(X, torch.Tensor): X = torch.tensor(X, dtype=torch.float32)
        X = X.to(device)
        
        if self.model_type == 'dropout':
            self.model.train() # Mantenemos dropout activo para inferencia bayesiana
            with torch.no_grad():
                preds = torch.stack([self.model(X) for _ in range(mc_samples)])
            mean_pred = preds.mean(dim=0).cpu().numpy()
            if return_std: return mean_pred, preds.std(dim=0).cpu().numpy()
            return mean_pred
        else:
            self.model.eval()
            with torch.no_grad(): return self.model(X).cpu().numpy()