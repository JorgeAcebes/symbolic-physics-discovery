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
        self.model = MLP(input_dim).to(device)
        self.epochs = epochs
        self.lr = lr

    def fit(self, train_loader, val_loader=None):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            
            self.model.train()
            for epoch in range(self.epochs):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = loss_fn(self.model(x), y)
                    loss.backward()
                    optimizer.step()
                    
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