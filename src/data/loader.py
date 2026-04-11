import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch




# Clase para la validación del Dataset y partición del train validation test
class PhysicalDataset:
    def __init__(self, filepath, target_col, scale=True):
        df = pd.read_csv(filepath)
        valid_mask = ~df.isna().any(axis=1) & ~df.isin([float('inf'), float('-inf')]).any(axis=1)
        df = df[valid_mask]

        # Extracción generalizada de nomenclaturas físicas
        self.target_name = target_col
        self.feature_names = [col for col in df.columns if col != target_col]

        X = df[self.feature_names].values
        y = df[self.target_name].values.reshape(-1, 1)
        
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

        self.scaler_y = None
        if scale:
            scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
            self.X_train = scaler_X.fit_transform(self.X_train)
            self.X_val = scaler_X.transform(self.X_val)
            self.X_test = scaler_X.transform(self.X_test)
            
            self.y_train = self.scaler_y.fit_transform(self.y_train)
            self.y_val = self.scaler_y.transform(self.y_val)
            self.y_test = self.scaler_y.transform(self.y_test)

    def get_arrays(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def get_dataloaders(self, batch_size=32):
        # Conversión a tensores y DataLoader exclusiva para PyTorch
        def to_loader(X, y, shuffle):
            ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                                torch.tensor(y, dtype=torch.float32))
            return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
            
        return to_loader(self.X_train, self.y_train, True), \
               to_loader(self.X_val, self.y_val, False), \
               to_loader(self.X_test, self.y_test, False)