import torch
import torch.nn as nn
from models.base import PhysicalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(), 
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class MLPWrapper(PhysicalModel):
    def __init__(self, input_dim, epochs=50, lr=1e-3):
        super().__init__()
        self.model = MLP(input_dim).to(device)
        self.epochs = epochs
        self.lr = lr

    def fit(self, train_loader, val_loader=None):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            
            for epoch in range(self.epochs):
                self.model.train()
                epoch_train_loss = 0.0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = loss_fn(self.model(x), y)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item() * x.size(0)
                
                epoch_train_loss /= len(train_loader.dataset)
                self.history["train_loss"].append(epoch_train_loss)
                
                if val_loader:
                    self.model.eval()
                    epoch_val_loss = 0.0
                    with torch.no_grad():
                        for x, y in val_loader:
                            x, y = x.to(device), y.to(device)
                            val_loss = loss_fn(self.model(x), y)
                            epoch_val_loss += val_loss.item() * x.size(0)
                    epoch_val_loss /= len(val_loader.dataset)
                    self.history["val_loss"].append(epoch_val_loss)
                    
            self.equation = "Red Neuronal (Caja Negra)" 
            return self

    def predict(self, X):
        self.model.eval()
        # Permite recibir tensores o NumPy arrays indistintamente
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        with torch.no_grad():
            preds = self.model(X.to(device))
            
        return preds.cpu().numpy()