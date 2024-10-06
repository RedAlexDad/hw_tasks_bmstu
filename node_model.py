import timeit
import time
import sys
import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display, HTML
from datetime import datetime
import matplotlib.pyplot as plt

# Метрики
from sklearn.metrics import mean_squared_error

# Нейросети
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torchdiffeq import odeint  # Импортируем ODE-решатель
from TorchDiffEqPack.odesolver import odesolve  # Заменяем пакет решателей ODE

# Класс для обучения и оценки модели Neural ODE
class NeuralODETrainer:
    def __init__(self, df, batch_size=64, learning_rate=0.001, epochs=10, hidden_size=128, method='dopri5'):
        # Преобразуем данные
        self.df = self.prepare_data(df)
        # self.df = df
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.method = method
        self.history = {"epoch": [], "rmse": []}  # История обучения

        # Подготовка датасета и DataLoader
        self.dataset = self.TimeSeriesDataset(self.df)
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        
        # Инициализация модели
        self.model = self.NeuralODEModel(hidden_size=self.hidden_size).to(self.get_device())
        self.criterion = nn.MSELoss()  # Используем MSE как функцию потерь
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # Метод для преобразования данных в нужный формат
    @staticmethod
    def prepare_data(df):
        df['Input'] = df['Input'].apply(lambda x: complex(x))  # Преобразуем строки в комплексные числа
        df['Output'] = df['Output'].apply(lambda x: complex(x))
        df['input_real'] = df['Input'].apply(lambda x: x.real)
        df['input_imag'] = df['Input'].apply(lambda x: x.imag)
        df['output_real'] = df['Output'].apply(lambda x: x.real)
        df['output_imag'] = df['Output'].apply(lambda x: x.imag)
        df = df.drop(['Input', 'Output'], axis=1)
        df = df.set_index('Time')  # Устанавливаем временной ряд как индекс
        return df
        
    # Класс для датасета временных рядов
    class TimeSeriesDataset(Dataset):
        def __init__(self, df):
            self.inputs = df[['input_real', 'input_imag']].values
            self.outputs = df[['output_real', 'output_imag']].values
            self.time = df.index.values.astype('float32')

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            x = torch.tensor(self.inputs[idx], dtype=torch.float32)
            y = torch.tensor(self.outputs[idx], dtype=torch.float32)
            t = torch.tensor(self.time[idx], dtype=torch.float32)
            return t, x, y

    # Класс ODE функции
    class ODEFunc(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)
            self.relu = nn.ReLU()

        def forward(self, t, x):
            return self.relu(self.linear(x))

    # Класс модели Neural ODE
    class NeuralODEModel(nn.Module):
        def __init__(self, input_size=2, hidden_size=128, output_size=2, num_hidden_layers=2):
            super().__init__()
            self.input_layer = nn.Linear(input_size, hidden_size)
            self.hidden_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)
            ])
            self.ode_func = NeuralODETrainer.ODEFunc(hidden_size)
            self.output_layer = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x, t):
            x = self.input_layer(x)
            for layer in self.hidden_layers:
                residual = x
                x = torch.relu(layer(x))  # Использование ReLU активации
                x = self.dropout(x)
                x += residual  # Остаточное соединение
                
            options = {
                'method': 'dopri5',  # метод решения (адаптивный шаг)
                't0': t[0].item(),  # начальное время
                't1': t[-1].item(),  # конечное время
                'rtol': 1e-9,  # относительная точность
                'atol': 1e-10,  # абсолютная точность
                'print_neval': False,  # не выводить количество итераций
                'neval_max': 1e7,  # максимальное число итераций
                't_eval': None,  # оценивать только конечное значение
                'regenerate_graph': False  # не перегенерировать граф для обратного прохода
            }
            
            x = odesolve(self.ode_func, x, options=options)
            x = self.output_layer(x)
            return x

    # Метод для получения устройства (CPU или GPU)
    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        device = self.get_device()
        self.model.train()
    
        for epoch in range(self.epochs):
            running_loss = 0.0
            rmse_total = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")
    
            for batch_idx, (t, inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(device), targets.to(device)
                self.optimizer.zero_grad()
    
                outputs = self.model(inputs, t)
                loss = self.criterion(outputs, targets)
                rmse = torch.sqrt(loss)  # RMSE
                rmse.backward()
                self.optimizer.step()
    
                # Рассчитываем RMSE
                rmse_total += rmse.item()
    
                progress_bar.set_postfix(rmse=f"{rmse:.6f}")
    
            avg_rmse = rmse_total / len(self.train_loader)
            
            # Сохраняем метрики в историю
            self.history["epoch"].append(epoch + 1)
            self.history["rmse"].append(avg_rmse)
    
            print(f"Epoch {epoch+1}/{self.epochs} completed with RMSE: {avg_rmse:.6f}")
    
        print("Training completed!")
        self.save_training_history(self.history)

    # Метод для оценки модели
    def evaluate(self):
        self.model.eval()
        true_values = []
        predicted_values = []
        times = []
    
        with torch.no_grad():
            for t, inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.get_device()), targets.to(self.get_device())
                outputs = self.model(inputs, t)
                true_values.append(targets.cpu().numpy())
                predicted_values.append(outputs.cpu().numpy())
                times.append(t.cpu().numpy())
    
        # Конкатенация всех предсказанных значений, истинных значений и временных меток
        self.true_values = np.concatenate(true_values, axis=0)
        self.predicted_values = np.concatenate(predicted_values, axis=0)
        self.times = np.concatenate(times, axis=0)
    
        # Сортировка данных по временам
        sort_indices = np.argsort(self.times)
        self.times = self.times[sort_indices]
        self.true_values = self.true_values[sort_indices]
        self.predicted_values = self.predicted_values[sort_indices]
    
        # Сохраним предсказанные значения
        self.save_prediction(self.predicted_values)
        # Сохраним модель
        self.save_model(self.model)
        
        # Вычисление RMSE
        rmse_real = mean_squared_error(self.true_values[:, 0], self.predicted_values[:, 0], squared=False)
        rmse_imag = mean_squared_error(self.true_values[:, 1], self.predicted_values[:, 1], squared=False)

        print(f"Evaluation RMSE (Real): {rmse_real:.6f}")
        print(f"Evaluation RMSE (Imaginary): {rmse_imag:.6f}")
    
        return rmse_real, rmse_imag

    @staticmethod
    def save_model(model, filename_prefix='node', save_dir='models'):
        # Создаем папку, если ее нет
        os.makedirs(save_dir, exist_ok=True)

        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pth"

        # Полный путь к файлу
        filepath = os.path.join(save_dir, filename)

        # Сохраняем ВСЮ модель
        torch.save(model.state_dict(), filepath)

    @staticmethod
    def save_prediction(predictions, filename_prefix="predictions", save_dir='history'):
        # Создаем папку, если ее нет
        os.makedirs(save_dir, exist_ok=True)

        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"

        # Полный путь к файлу
        filepath = os.path.join(save_dir, filename)
        
        df_predictions = pd.DataFrame({'real': predictions[:, 0], 'imag': predictions[:, 1],})
        df_predictions.to_csv(filepath, index=False)
        print(f"Training history saved to {filepath}")

    @staticmethod
    def save_training_history(history, filename_prefix="training_history", save_dir='history'):
        # Создаем папку, если ее нет
        os.makedirs(save_dir, exist_ok=True)

        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"

        # Полный путь к файлу
        filepath = os.path.join(save_dir, filename)
        
        df_history = pd.DataFrame(history)
        df_history.to_csv(filepath, index=False)
        print(f"Training history saved to {filepath}")

    def plot_training_history(self):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
        # Преобразуем список эпох для оси X
        epochs = self.history["epoch"]
    
        # Первый график: Полная история
        axs[0].plot(epochs, self.history["rmse"], marker='o', linestyle='-', color='b', markersize=5, label='RMSE')
        # Второй график: RMSE
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Average Loss')
        axs[0].set_title('Loss Function (Full History)')
        axs[0].grid(True)
        axs[0].legend()

        # Второй график: Половина истории
        mid_index = len(epochs) // 2
        axs[1].plot(epochs[mid_index:], self.history["rmse"][mid_index:], marker='o', linestyle='-', color='b', markersize=5, label='RMSE')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Average RMSE')
        axs[1].set_title('Loss Function (Second Half of Training)')
        axs[1].grid(True)
        axs[1].legend()
    
        plt.tight_layout()
        plt.show()

    def plot_predictions(self, time_start=0, time_end=1.005e2):
        # Убедимся, что данные были вычислены в evaluate()
        if not hasattr(self, 'true_values') or not hasattr(self, 'predicted_values') or not hasattr(self, 'times'):
            raise ValueError("You must run evaluate() before plotting predictions.")
    
        # Фильтрация данных по времени
        mask = (self.times >= time_start) & (self.times <= time_end)
        filtered_times = self.times[mask]
        filtered_true_values = self.true_values[mask]
        filtered_predicted_values = self.predicted_values[mask]
    
        # Построение графика для реальных значений
        plt.figure(figsize=(20, 5))
    
        plt.subplot(1, 2, 1)  # 1 ряд, 2 колонки, 1-й график
        plt.plot(filtered_times, filtered_true_values[:, 0], label="True Real", linestyle='-', color='red')
        plt.plot(filtered_times, filtered_predicted_values[:, 0], label="Predicted Real", linestyle='-', color='blue')
        plt.legend()
        plt.title("True vs Predicted Real Values")
        plt.xlabel("Time")
        plt.ylabel("Real Value")
    
        # Построение графика для мнимых значений
        plt.subplot(1, 2, 2)  # 1 ряд, 2 колонки, 2-й график
        plt.plot(filtered_times, filtered_true_values[:, 1], label="True Imag", linestyle='-', color='red')
        plt.plot(filtered_times, filtered_predicted_values[:, 1], label="Predicted Imag", linestyle='-', color='blue')
        plt.legend()
        plt.title("True vs Predicted Imaginary Values")
        plt.xlabel("Time")
        plt.ylabel("Imaginary Value")
    
        plt.tight_layout()
        plt.show()

    def cross_validate(self, param_grid, cv=5):
        # Разделение данных на входы и выходы
        X = self.dataset.inputs
        y = self.dataset.outputs
    
        # Определение модели для кросс-валидации
        model = self.NeuralODEModel(hidden_size=self.hidden_size)
    
        # Использование GridSearchCV для поиска лучших гиперпараметров
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
        grid_search.fit(X, y)
    
        print("Best parameters found: ", grid_search.best_params_)
        print("Best score (RMSE): ", -grid_search.best_score_)