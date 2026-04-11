from abc import ABC, abstractmethod

# Garantiza que cualquier modelo deba implementar los métodos fit y predict.
class PhysicalModel(ABC):
    def __init__(self):
        self.equation = "N/A"
        self.history = {"train_loss": [], "val_loss": []}

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass