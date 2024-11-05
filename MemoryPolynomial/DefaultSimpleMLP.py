import torch.nn as nn

# 1. Класс для нейросети
class DefaultSimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size=2, dropout_rate=0.5, learning_rate=0.001):
            super().__init__()
            self.input_size = input_size
            self.hidden_layers = hidden_layers
            self.output_size = output_size
            self.dropout_rate = dropout_rate
            self.learning_rate = learning_rate
 
            self.layers = []
            self.activations = []  # Список для хранения активаций
            self.hooks = []  # Список для хранения hook'ов
    
            # Входной слой
            layer = nn.Linear(input_size, hidden_layers[0])
            self.layers.append(layer)
            self.add_activation_logging(layer)  # Добавить логирование активаций
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_layers[0]))  # Batch Normalization
            self.layers.append(nn.Dropout(dropout_rate))
    
            # Скрытые слои
            for i in range(1, len(hidden_layers)):
                layer = nn.Linear(hidden_layers[i - 1], hidden_layers[i])
                self.layers.append(layer)
                self.add_activation_logging(layer)  # Добавить логирование активаций
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm1d(hidden_layers[i]))  # Batch Normalization
                self.layers.append(nn.Dropout(dropout_rate))
    
            # Выходной слой
            layer = nn.Linear(hidden_layers[-1], output_size)
            self.layers.append(layer)
    
            self.model = nn.Sequential(*self.layers)
    
        def add_activation_logging(self, layer):
            self.hooks.append(layer.register_forward_hook(self.hook))
    
        def hook(self, _, __, output):
            self.activations.append(output.detach().cpu())
            
        def clear_activations(self):
            """Очистка сохранённых активаций."""
            self.activations = []
    
        def forward(self, x):
            return self.model(x)
    
        def close_hooks(self):
            """Закрытие hook'ов после окончания обучения."""
            for hook in self.hooks:
                hook.remove()
    