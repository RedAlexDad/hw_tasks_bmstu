import os
import sys
import uuid
import numpy as np
import pandas as pd
import seaborn as sb
from tqdm.auto import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from IPython.display import display, HTML, Markdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MemoryPolynomialNNTrainer:
    def __init__(self, df, M, K, batch_size=64, learning_rate=0.001, epochs=10, hidden_layers=[64, 128], device=None):
        """
        Инициализация класса для тренировки модели Memory Polynomial с использованием нейронных сетей.

        Args:
            df (pd.DataFrame): Входные данные.
            M (int): Глубина памяти.
            K (int): Степень полинома.
            batch_size (int, optional): Размер батча для обучения. По умолчанию 64.
            learning_rate (float, optional): Скорость обучения для оптимизатора. По умолчанию 0.001.
            epochs (int, optional): Количество эпох. По умолчанию 10.
            hidden_layers (list, optional): Конфигурация скрытых слоев. По умолчанию [64, 128].
            device (str, optional): Устройство для выполнения вычислений ('cpu' или 'cuda'). По умолчанию None.
        """
        # Подготовка данных
        self.df = self.prepare_data(df)
        self.device = self.get_device(device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.M = M
        self.K = K
        self.hidden_layers = hidden_layers

        # История обучения
        self.history = {"epoch": [], "rmse": []}

        # Генерация уникального ID
        self.model_id = str(uuid.uuid4())

        # Подготовка данных
        X, y, self.times = self.create_dataset(self.df, M, K)
        self.dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(self.times, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Инициализация модели
        self.model = self.SimpleMLP(input_size=X.shape[1] + 1, hidden_layers=hidden_layers, output_size=2).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size=2):
            """
            Простая классическая нейросеть MLP для регрессии реальной и мнимой частей сигнала.
    
            Args:
                input_size (int): Размерность входного слоя (количество признаков).
                hidden_layers (list): Список размеров скрытых слоев.
                output_size (int, optional): Размерность выходного слоя. По умолчанию 2 (для реальной и мнимой части).
            """
            super().__init__()
            
            layers = []
            # Входной слой
            layers.append(nn.Linear(input_size, hidden_layers[0]))
            layers.append(nn.ReLU())
            
            # Скрытые слои
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(nn.ReLU())
            
            # Выходной слой
            layers.append(nn.Linear(hidden_layers[-1], output_size))
            
            self.model = nn.Sequential(*layers)
    
        def forward(self, x):
            """
            Прямой проход через MLP.
            Args:
                x (torch.Tensor): Входные данные.
            Returns:
                torch.Tensor: Прогнозы (реальная и мнимая части).
            """
            return self.model(x)
                
    @staticmethod
    def prepare_data(df):
        """
        Преобразование входных данных: разделение на реальные и мнимые части.

        Args:
            df (pd.DataFrame): Входные данные.

        Returns:
            pd.DataFrame: Обработанные данные.
        """
        df.columns = df.columns.str.lower()
        df['input'] = df['input'].apply(lambda x: complex(x))
        df['output'] = df['output'].apply(lambda x: complex(x))
        df['input_real'] = df['input'].apply(lambda x: x.real)
        df['input_imag'] = df['input'].apply(lambda x: x.imag)
        df['output_real'] = df['output'].apply(lambda x: x.real)
        df['output_imag'] = df['output'].apply(lambda x: x.imag)
        df = df.drop(['input', 'output'], axis=1)
        df = df.set_index('time')  # 'time' колонка теперь индекс
        return df

    @staticmethod
    def get_device(select=None):
        """
        Определение устройства для вычислений (CPU или GPU).

        Args:
            select (str, optional): Выбор устройства ('cpu', 'cuda'). По умолчанию None.

        Returns:
            torch.device: Устройство для вычислений.
        """
        if select is None or select == 'cuda':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device('cpu')

    @staticmethod
    def create_dataset(df, M, K):
        """
        Создание обучающих данных на основе полиномиальной модели с памятью и добавлением времени.

        Args:
            df (pd.DataFrame): Обработанные данные.

        Returns:
            tuple: Матрица признаков X, объединенные целевые значения y и временные метки times.
        """
        x_real = df['input_real'].values
        x_imag = df['input_imag'].values
        y_real = df['output_real'].values[M:]
        y_imag = df['output_imag'].values[M:]
        times = df.index.values[M:]

        # Объединение реальных и мнимых частей в один целевой вектор
        y = np.stack([y_real, y_imag], axis=1)

        N = len(x_real)
        X = np.zeros((N, (M + 1) * K * 2), dtype=np.float64)
        for n in range(M, N):
            index = 0
            for m in range(M + 1):
                for k in range(1, K + 1):
                    X[n, index] = np.abs(x_real[n - m])**(k-1) * x_real[n - m]
                    X[n, index + 1] = np.abs(x_imag[n - m])**(k-1) * x_imag[n - m]
                    index += 2
        return X[M:], y, times

    def train(self):
        """
        Обучение модели на объединенных целевых данных (реальная и мнимая части).
        """
        self.model.train()

        for epoch in range(self.epochs):
            total_rmse = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")

            for batch_idx, (X_batch, times_batch, y_batch) in enumerate(progress_bar):
                X_batch, times_batch, y_batch = X_batch.to(self.device), times_batch.to(self.device), y_batch.to(self.device)

                # Объединение временных меток с входными признаками
                X_with_times = torch.cat((X_batch, times_batch.unsqueeze(1)), dim=1)

                self.optimizer.zero_grad()
                pred = self.model(X_with_times)
                loss = self.criterion(pred, y_batch)
                rmse = torch.sqrt(loss)  # RMSE
                rmse.backward()
                self.optimizer.step()

                total_rmse += rmse.item()
                progress_bar.set_postfix(rmse=f"{rmse:.6f}")

            avg_rmse = total_rmse / len(self.train_loader)
            self.history["epoch"].append(epoch + 1)
            self.history["rmse"].append(avg_rmse)

            print(f"Epoch {epoch+1}/{self.epochs}, RMSE: {avg_rmse:.6f}")

    def evaluate(self):
        """
        Оценка модели после обучения.
        
        Returns:
            float: Значение RMSE.
        """
        self.model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for X_batch, times_batch, y_batch in self.train_loader:
                X_batch, times_batch, y_batch = X_batch.to(self.device), times_batch.to(self.device), y_batch.to(self.device)
                
                # Объединение временных меток с входными признаками
                X_with_times = torch.cat((X_batch, times_batch.unsqueeze(1)), dim=1)

                pred = self.model(X_with_times)
                
                all_preds.append(pred.cpu().numpy())
                all_true.append(y_batch.cpu().numpy())

        # Конкатенация всех предсказаний и истинных значений
        self.pred = np.concatenate(all_preds)
        self.true = np.concatenate(all_true)
        
        # Вычисление RMSE
        rmse = np.sqrt(mean_squared_error(self.true, self.pred))

        print(f"Evaluation RMSE: {rmse:.6f}")
        return rmse

    def save_model(self, path="models"):
        """
        Сохранение модели в файл.

        Args:
            path (str, optional): Путь для сохранения модели. По умолчанию 'models'.
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, f"model_{self.model_id}.pt"))
        print(f"Model saved in {path}")

    def plot_training_history(self):
        """
        Строит графики истории обучения модели, отображая RMSE на каждой эпохе.
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
        # Преобразуем список эпох для оси X
        epochs = self.history["epoch"]
    
        # Первый график: Полная история
        axs[0].plot(epochs, self.history["rmse"], marker='o', linestyle='-', color='b', markersize=5, label='RMSE')
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

    def plot_predictions(self, time_start=0, time_end=1.01e-4):
        """
        Построение графиков предсказанных и фактических значений в заданном временном диапазоне.

        Args:
            time_start (float, optional): Начальное время для отображения. По умолчанию 0.
            time_end (float, optional): Конечное время для отображения. По умолчанию 1.01e-4.
        """
        self.model.eval()
        pred_real, pred_imag = self.pred[:, 0], self.pred[:, 1]
        true_real, true_imag = self.true[:, 0], self.true[:, 1]

        # Фильтрация данных по указанному временному диапазону
        time_mask = (self.times >= time_start) & (self.times <= time_end)

        selected_times = self.times[time_mask]
        
        # Фильтрация предсказаний и фактических значений по временному диапазону
        pred_real = pred_real[time_mask]
        pred_imag = pred_imag[time_mask]
        true_real = true_real[time_mask]
        true_imag = true_imag[time_mask]
    
        # Проверка, чтобы убедиться, что данные не пустые
        if len(selected_times) == 0:
            print(f"No data points found between {time_start} and {time_end}.")
            return
        
        # Построение графиков
        fig, axs = plt.subplots(2, 1, figsize=(15, 8))

        # Реальная часть
        axs[0].plot(selected_times, true_real, label='True Real', linestyle='-', color='red')
        axs[0].plot(selected_times, pred_real, label='Predicted Real', linestyle='-', color='blue')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Real Part')
        axs[0].legend()
        axs[0].grid(True)

        # Мнимая часть
        axs[1].plot(selected_times, true_imag, label='True Imag', linestyle='-', color='red')
        axs[1].plot(selected_times, pred_imag, label='Predicted Imag', linestyle='-', color='blue')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Imaginary Part')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    def draw_plot_signal(self, signal_type, time_start=0, time_end=1e-6):
        """
        Построение графика сигнала в указанном временном диапазоне.
        
        Args:
            signal_type (str): Тип сигнала ('input' или 'output').
            time_start (float): Начальное время.
            time_end (float): Конечное время.
        """
        # Фильтрация данных по временной отметке
        filtered_data = self.df[(self.df.index >= time_start) & (self.df.index <= time_end)]
        time = filtered_data.index

        # Построение графика реальной и мнимой частей сигнала
        plt.figure(figsize=(10, 6))
        plt.plot(time, filtered_data[f'{signal_type}_real'], label=f'{signal_type} Real Part', color='blue', linestyle='-')
        plt.plot(time, filtered_data[f'{signal_type}_imag'], label=f'{signal_type} Imaginary Part', color='red', linestyle='-')
        
        plt.title(f'{signal_type.capitalize()} Signal from {time_start} to {time_end} seconds')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()
        

if __name__ == '__main__':
    try:
        df = pd.read_csv('Amp_C_train.txt')
    except:
        for dirname, _, filenames in os.walk('/kaggle/input'):
            for filename in filenames:
                print(os.path.join(dirname, filename))
                
        df = pd.read_csv(os.path.join(dirname, filename))
        
        
    M = 2  # Глубина памяти
    K = 1  # Степень полинома
    batch_size=1024*1
    learning_rate=1e-4
    epochs=2
    hidden_layers=[2**6, 2**7, 2**7, 2**6]

    # Создание экземпляра класса с настройкой гиперпараметров
    model_nn = MemoryPolynomialNNTrainer(
        df, 
        M, K, 
        batch_size,
        learning_rate, 
        epochs, 
        hidden_layers, 
        device=None
    )
    
    model_nn.train()
    
    model_nn.evaluate()
    
    