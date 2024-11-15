import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ComplexDataset import ComplexDataset
from RVTDNNModel import *


class CreateModel():
    def __init__(self, input_file, time_delay, hidden_layers, batch_size, dropout_rate=0.1, device=None, epochs=10, learning_rate=0.001, patience=3, factor=0.1):
        self.dataset = ComplexDataset(input_file, time_delay)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        self.device = self.get_device()
        print('Using device: ', self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.history = {"epoch": [], "total_rmse": [], "rmse_real": [], "rmse_imag": []} # История обучения
        
        self.model = RVTDNNModel_0(time_delay, hidden_layers[0]).to(self.device)
        # self.model = RVTDNNModel_2(time_delay, hidden_layers, dropout_rate).to(self.device)
        # self.model = RVTDNNModel_3(time_delay, hidden_layers, dropout_rate).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=patience, factor=factor)
        
    @staticmethod
    def get_device(select=None):
        return torch.device('cuda' if (select in [None, 'cuda'] and torch.cuda.is_available()) else 'cpu')
    