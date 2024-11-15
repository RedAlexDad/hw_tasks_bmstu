import torch
import torch.nn as nn


class RVTDNNModel_0(nn.Module):
    def __init__(self, time_delay, hidden_neurons):
        super(RVTDNNModel_0, self).__init__()
        self.time_delay = time_delay
        input_dim = 2 * (time_delay + 1)
        self.fc1 = nn.Linear(input_dim, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, 2)
        self.activation = nn.Tanh()

    def forward(self, I_in, Q_in):
        x = torch.cat((I_in, Q_in), dim=1)
        x = x.view(-1, 2 * (self.time_delay + 1))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
    

class RVTDNNModel_1(nn.Module):
    def __init__(self, time_delay, hidden_layers, dropout_rate=0.5):
        super(RVTDNNModel, self).__init__()
        self.activations = []
        self.hooks = []

        self.time_delay = time_delay
        input_dim = 2 * (time_delay + 1)
        
        # Строим последовательные слои
        self.layers = []

        # Входной слой
        layer = nn.Linear(input_dim, hidden_layers[0])
        self.layers.append(layer)
        self.add_activation_logging(layer)
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_layers[0]))
        self.layers.append(nn.Dropout(dropout_rate))

        # Скрытые слои
        for i in range(1, len(hidden_layers)):
            layer = nn.Linear(hidden_layers[i - 1], hidden_layers[i])
            self.layers.append(layer)
            self.add_activation_logging(layer)
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
            self.layers.append(nn.Dropout(dropout_rate))

        # Выходной слой
        layer = nn.Linear(hidden_layers[-1], 2)  # два выходных нейрона для I_out и Q_out
        self.layers.append(layer)

        # Создаем последовательную модель
        self.model = nn.Sequential(*self.layers)

    def add_activation_logging(self, layer):
        self.hooks.append(layer.register_forward_hook(self.hook))

    def hook(self, _, __, output):
        self.activations.append(output.detach().cpu())

    def clear_activations(self):
        """Очистка сохранённых активаций."""
        self.activations = []

    def forward(self, I_in, Q_in):
        x = torch.cat((I_in, Q_in), dim=1)
        x = x.view(-1, 2 * (self.time_delay + 1))
        return self.model(x)

    def close_hooks(self):
        """Закрытие hook'ов после окончания обучения."""
        for hook in self.hooks:
            hook.remove()
              
    
class RVTDNNModel_2(nn.Module):
    def __init__(self, time_delay, hidden_layers, dropout_rate=0.5):
        super(RVTDNNModel_2, self).__init__()
        self.time_delay = time_delay
        input_dim = 2 * (time_delay + 1)

        # Создание последовательности из линейных слоев
        layers = []
        prev_dim = input_dim

        # Добавление скрытых слоев
        for hidden_neurons in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_neurons))
            layers.append(nn.Tanh())
            # layers.append(nn.BatchNorm1d(hidden_neurons))
            # layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_neurons

        # Добавление выходного слоя
        layers.append(nn.Linear(prev_dim, 2))  # два выходных нейрона для I_out и Q_out

        # Создание последовательности из слоев
        self.model = nn.Sequential(*layers)

    def forward(self, I_in, Q_in):
        x = torch.cat((I_in, Q_in), dim=1)
        x = x.view(-1, 2 * (self.time_delay + 1))
        return self.model(x)
    
class RVTDNNModel_3(nn.Module):
    def __init__(self, time_delay, hidden_layers, dropout_rate=0.5):
        super(RVTDNNModel_3, self).__init__()
        self.time_delay = time_delay
        input_dim = 2 * (time_delay + 1)

        # Создание последовательности из линейных слоев
        layers = []
        prev_dim = input_dim

        # Добавление скрытых слоев
        for hidden_neurons in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_neurons))
            layers.append(nn.Tanh())
            layers.append(nn.BatchNorm1d(hidden_neurons))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_neurons

        # Добавление выходного слоя
        layers.append(nn.Linear(prev_dim, 2))  # два выходных нейрона для I_out и Q_out

        # Создание последовательности из слоев
        self.model = nn.Sequential(*layers)

    def forward(self, I_in, Q_in):
        x = torch.cat((I_in, Q_in), dim=1)
        x = x.view(-1, 2 * (self.time_delay + 1))
        return self.model(x)