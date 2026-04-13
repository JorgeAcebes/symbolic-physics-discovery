# Importación del instrumental matemático.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

# Clase para la validación del Dataset y partición del train validation test
class PhysicalDataset:
    def __init__(self, filepath, target_col, scale=True): 
        '''
        Constructor del inyector de datos. Recibe la dirección de memoria, el escalar target y
        y un booleano para decidir si se proyectan los datos al espacio de z-score.
        '''

        df = pd.read_csv(filepath) # Carga de datos a un dataframe 
        df.columns = df.columns.str.strip() # Purga de caracteres como espacios en la cabecera del dataframe
        
        # Purga de singularidades e infinitos en los datos (saneamiento)
        valid_mask = ~df.isna().any(axis=1) & ~df.isin([float('inf'), float('-inf')]).any(axis=1)
        df = df[valid_mask]


        self.target_name = target_col # Aislamos el nombre magnitud objetivo
        self.feature_names = [col for col in df.columns if col != target_col] # Agrupamos el resto de nombres de componentes
        # en el vector de los nombres de las características

        # Extracción de la matriz de características (m x n) y del vector de magnitud objetivo (m x 1)
        X = df[self.feature_names].values.astype(np.float32)
        y = df[self.target_name].values.reshape(-1, 1).astype(np.float32)
        # Aseguramos que sean tensores float32 asegurando compatibilidad con los tensores de PyTorch.
        
        # Partición de los datos (60/20/20).
        X_temp, self.X_test_phys, y_temp, self.y_test_phys = train_test_split(X, y, test_size=0.2, random_state=42) # Aislamos 20% para test
        self.X_train_phys, self.X_val_phys, self.y_train_phys, self.y_val_phys = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42) # Del 80% restante
        # se asocia un 25% para validation: 0.8 * 0.25 = 0.20

        # Inicialización del espacio transformado
        self.X_train, self.X_val, self.X_test = self.X_train_phys.copy(), self.X_val_phys.copy(), self.X_test_phys.copy()
        self.y_train, self.y_val, self.y_test = self.y_train_phys.copy(), self.y_val_phys.copy(), self.y_test_phys.copy()

        # Inicialización de los operadores de transformación. Por defecto, el operador mapea del espacio físico al espacio transformado como si fuese la identidad.
        self.scaler_X = None
        self.scaler_y = None
        
        # Proyección al hipercubo normalizado si es requerido
        if scale:
            # Instanciamos los operadores que realizarán la transformación al espacio de z-score: (x - mu)/sigma
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
            # Transformación al espacio z-score de train, val y test (EMPLEANDO ÚNICAMENTE TRAIN, EVITAMOS DATA LEAKAGE)
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
        """Espacio crudo asumiendo la métrica original para inferencia discreta (Regresadores Simbólicos)."""
        return self.X_train_phys, self.X_val_phys, self.X_test_phys, self.y_train_phys, self.y_val_phys, self.y_test_phys

    def get_dataloaders(self, batch_size=32):
        '''
        Creación de generadores de batches para PyTorch.
        Para train desordenamso los datos para mayor generalización.
        Para validation y test no barajamos para evaluación reproducible.
        '''
        def to_loader(X, y, shuffle):
            # Juntamos X e y en un dataset
            ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), # Convierte X a PyTorch
                                                torch.tensor(y, dtype=torch.float32)) # Convierte y a PyTorch
            
            return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
            
        return to_loader(self.X_train, self.y_train, True), \
               to_loader(self.X_val, self.y_val, False), \
               to_loader(self.X_test, self.y_test, False)