import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch




# Clase para la validación del Dataset y partición del train validation test
class PhysicalDataset:
    def __init__(self, filepath, target_col, scale=True):
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        
        valid_mask = ~df.isna().any(axis=1) & ~df.isin([float('inf'), float('-inf')]).any(axis=1)
        df = df[valid_mask]

        self.target_name = target_col
        self.feature_names = [col for col in df.columns if col != target_col]

        X = df[self.feature_names].values
        y = df[self.target_name].values.reshape(-1, 1)
        
        # 1. Preservación estricta del espacio físico
        X_temp, self.X_test_phys, y_temp, self.y_test_phys = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train_phys, self.X_val_phys, self.y_train_phys, self.y_val_phys = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

        # 2. Inicialización del espacio latente
        self.X_train, self.X_val, self.X_test = self.X_train_phys, self.X_val_phys, self.X_test_phys
        self.y_train, self.y_val, self.y_test = self.y_train_phys, self.y_val_phys, self.y_test_phys

        self.scaler_X = None
        self.scaler_y = None
        
        # 3. Proyección al hipercubo normalizado si es requerido
        if scale:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
            self.X_train = self.scaler_X.fit_transform(self.X_train_phys)
            self.X_val = self.scaler_X.transform(self.X_val_phys)
            self.X_test = self.scaler_X.transform(self.X_test_phys)
            
            self.y_train = self.scaler_y.fit_transform(self.y_train_phys)
            self.y_val = self.scaler_y.transform(self.y_val_phys)
            self.y_test = self.scaler_y.transform(self.y_test_phys)

    def get_latent_arrays(self):
        """Espacio estandarizado para optimización continua (MLP, Polynomial)."""
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def get_physical_arrays(self):
        """Espacio crudo asumiendo la métrica original para inferencia discreta (PySR, SINDy)."""
        return self.X_train_phys, self.X_val_phys, self.X_test_phys, self.y_train_phys, self.y_val_phys, self.y_test_phys

    def get_dataloaders(self, batch_size=32):
        # El Dataloader opera siempre en el espacio latente
        def to_loader(X, y, shuffle):
            ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                                torch.tensor(y, dtype=torch.float32))
            return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
            
        return to_loader(self.X_train, self.y_train, True), \
               to_loader(self.X_val, self.y_val, False), \
               to_loader(self.X_test, self.y_test, False)