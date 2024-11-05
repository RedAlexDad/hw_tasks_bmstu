import torch
import torch.nn as nn
import torch.optim as optim

from PolynomialDataset import PolynomialDataset
from DefaultSimpleMLP import DefaultSimpleMLP


class CreateModel():
    def __init__(self, df, M, K, model_type, batch_size, hidden_layers, learning_rate=0.001, epochs=10, patience=2, factor=0.9, dropout_rate=0.5, device=None):
        self.dataset = PolynomialDataset(df, M, K, model_type, batch_size)
        self.model = DefaultSimpleMLP(input_size=self.dataset.X.shape[1] + 1, hidden_layers=hidden_layers, dropout_rate=dropout_rate, learning_rate=learning_rate)

        self.device = self.get_device(device)
        print(f'Using drive: {self.device}')
        self.model.to(self.device)
                
        self.epochs = epochs
        self.history = {"epoch": [], "total_rmse": [], "rmse_real": [], "rmse_imag": []} # История обучения
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=patience, factor=factor)
        
    @staticmethod
    def get_device(select=None):
        return torch.device('cuda' if (select in [None, 'cuda'] and torch.cuda.is_available()) else 'cpu')
    