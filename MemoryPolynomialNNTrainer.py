import os
import sys
import uuid
import time
import signal
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
from torch.utils.tensorboard import SummaryWriter

class MemoryPolynomialNNTrainer:
    def __init__(
        self, df, M, K, model_type,
        batch_size=64, learning_rate=0.001, epochs=10, hidden_layers=[64, 128], 
        dropout_rate=0.5, patience=2, factor=0.9, edit_model=None, device=None
        ):
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
            experiment_name(str, optional): Название эксперимента. По умолчанию None.
            model_type (str): Тип полиномиальной модели ('memory_polynomial', 'sparse_delay_memory_polynomial', 'non_uniform_memory_polynomial').
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
        self.dropout_rate = dropout_rate
        self.history = {"epoch": [], "total_rmse": [], "rmse_real": [], "rmse_imag": []} # История обучения
        self.model_id = str(uuid.uuid4()) # Генерация уникального ID
        self.last_rmse_real = None
        self.last_rmse_imag = None
        self.model_type = model_type
        
        # Создание данных в зависимости от выбранного типа модели
        self.prepare_dataset(model_type, M, K)
        
        self.dataset = TensorDataset(
            torch.tensor(self.X, dtype=torch.float32),
            torch.tensor(self.times, dtype=torch.float32),
            torch.tensor(self.y, dtype=torch.float32)
        )
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Инициализация модели
        if edit_model:
            self.model = edit_model(input_size=X.shape[1] + 1, hidden_layers=hidden_layers, output_size=2, dropout_rate=dropout_rate).to(self.device)
        else:
            self.model = self.DefaultSimpleMLP(input_size=self.X.shape[1] + 1, hidden_layers=hidden_layers, output_size=2, dropout_rate=dropout_rate).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=patience, factor=factor)
        
        # Установка обработчика сигналов
        signal.signal(signal.SIGINT, self.signal_handler)
    
    class DefaultSimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size=2, dropout_rate=0.5):
            super().__init__()
    
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

    def prepare_dataset(self, model_type, M, K):
        """Подготавливает данные на основе типа модели для оценки."""
        if model_type == 'memory_polynomial':
            self.X, self.y, self.times = self.create_dataset_memory_polynominal(self.df, M, K)
        elif model_type == 'sparse_delay_memory_polynomial':
            delays = range(self.M + 1)
            self.X, self.y, self.times = self.create_dataset_sparse_delay_memory_polynominal(self.df, M, K, delays)
        elif model_type == 'non_uniform_memory_polynomial':
            K_list = [self.K] * (self.M + 1)
            self.X, self.y, self.times = self.create_dataset_non_uniform_memory_polynominal(self.df, M, K_list)
        elif model_type == 'envelope_memory_polynomial':
            self.X, self.y, self.times = self.create_dataset_envelope_memory_polynomial(self.df, M, K)
        else:
            raise ValueError(f"Unknown model type")
  
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
        Определяет устройство для вычислений (CPU или GPU) и выводит информацию об использовании.

        Args:
            select (str, optional): Выбор устройства ('cpu', 'cuda'). По умолчанию None.

        Returns:
            torch.device: Устройство для вычислений.
        """
        device = torch.device('cuda' if torch.cuda.is_available() and (select is None or select == 'cuda') else 'cpu')
        
        if device.type == 'cuda':
            print("CUDA is available. Using GPU.")
        else:
            print("CUDA not available or not selected. Using CPU.")
        
        return device


    @staticmethod
    def prepare_data_for_create_dataset(df, M):
        """Подготовка данных для преобразования.

        Args:
            df (pd.DataFrame): Обработанные данные.
            M (int): Параметр памяти.

        Returns:
            tuple: Содержит следующие компоненты:
                - x_real (np.ndarray): Вектор реальных частей входных данных.
                - x_imag (np.ndarray): Вектор мнимых частей входных данных.
                - y_real (np.ndarray): Вектор реальных частей выходных данных, начиная с M-го элемента.
                - y_imag (np.ndarray): Вектор мнимых частей выходных данных, начиная с M-го элемента.
                - times (np.ndarray): Вектор временных меток начиная с M-го элемента.
        """
        x_real = df['input_real'].values
        x_imag = df['input_imag'].values
        y_real = df['output_real'].values[M:]
        y_imag = df['output_imag'].values[M:]
        times = df.index.values[M:]
        return x_real, x_imag, y_real, y_imag, times
    
    def create_dataset_memory_polynominal(self, df, M, K):
        """
        Создание обучающих данных на основе полиномиальной модели с памятью и добавлением времени.

        Args:
            df (pd.DataFrame): Обработанные данные.
            M (int): Параметр памяти.
            K (int): Максимальная степень полинома.

        Returns:
            tuple: Матрица признаков X, объединенные целевые значения y и временные метки times.
        """
        x_real, x_imag, y_real, y_imag, times = self.prepare_data_for_create_dataset(df, M)
        
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
    
    def create_dataset_sparse_delay_memory_polynominal(self, df, M, K, delays):
        """
        Создание обучающих данных на основе Sparse-Delay Memory Polynomial (SDMP) модели.

        Args:
            df (pd.DataFrame): Обработанные данные.
            M (int): Параметр памяти (не используется для SDMP, но включен для совместимости).
            K (int): Максимальная степень полинома.

        Returns:
            tuple: Матрица признаков X, объединенные целевые значения y и временные метки times.
        """
        x_real, x_imag, y_real, y_imag, times = self.prepare_data_for_create_dataset(df, M)

        # Объединение реальных и мнимых частей в один целевой вектор
        y = np.stack([y_real[M:], y_imag[M:]], axis=1)

        N = len(x_real)
        X = np.zeros((N - M, len(delays) * K * 2), dtype=np.float64)
        for n in range(M, N):
            index = 0
            for m in delays:
                if n - m >= 0:
                    for k in range(1, K + 1):
                        X[n - M, index] = np.abs(x_real[n - m])**(k - 1) * x_real[n - m]
                        X[n - M, index + 1] = np.abs(x_imag[n - m])**(k - 1) * x_imag[n - m]
                    index += 2

        # Обрезка до необходимых размеров
        X = X[:len(y)]  # Срез в зависимости от наименьшего размера
        times = times[M:M + len(y)]  # Приводим ко всем одинаковым размерам

        return X, y, times
    
    def create_dataset_non_uniform_memory_polynominal(self, df, M, K_list):
        """
        Создание обучающих данных на основе Non-Uniform Memory Polynomial (NUMP) модели.

        Args:
            df (pd.DataFrame): Обработанные данные.
            M (int): Параметр памяти (на этот раз может использоваться для NUMP).
            K_list (list of int): Перечень степеней полиномов для каждой задержки.

        Returns:
            tuple: Матрица признаков X, объединенные целевые значения y и временные метки times.
        """
        x_real, x_imag, y_real, y_imag, times = self.prepare_data_for_create_dataset(df, M)

        # Объединение реальных и мнимых частей в один целевой вектор
        y = np.stack([y_real, y_imag], axis=1)

        N = len(x_real)
        X = np.zeros((N, sum(K_list) * 2), dtype=np.float64)
        index = 0
        for m in range(self.M + 1):
            for k in range(1, K_list[m] + 1):
                for n in range(self.M, N):
                    X[n, index] = np.abs(x_real[n - m])**(k-1) * x_real[n - m]
                    X[n, index + 1] = np.abs(x_imag[n - m])**(k-1) * x_imag[n - m]
                index += 2
        return X[M:], y, times
    
    def create_dataset_envelope_memory_polynomial(self, df, M, K):
        """
        Создание обучающих данных на основе Envelope Memory Polynomial (EMP) модели.

        Args:
            df (pd.DataFrame): Обработанные данные.
            M (int): Параметр памяти.
            K (int): Максимальная степень полинома.

        Returns:
            tuple: Матрица признаков X, объединенные целевые значения y и временные метки times.
        """
        x_real, x_imag, y_real, y_imag, times = self.prepare_data_for_create_dataset(df, M)

        # Объединение реальных и мнимых частей в один целевой вектор
        y = np.stack([y_real, y_imag], axis=1)

        N = len(x_real)
        amplitude = np.sqrt(x_real**2 + x_imag**2)  # Амплитуда сигнала

        X = np.zeros((N - M, (M + 1) * K), dtype=np.float64)
        for n in range(M, N):
            index = 0
            for m in range(M + 1):
                for k in range(1, K + 1):
                    X[n - M, index] = amplitude[n] * amplitude[n - m]**(k-1)
                    index += 1

        # Обрезка X до соответствующего размера
        X = X[:len(y[M:])]
        return X, y[M:], times[M:]
    
    def train(self, max_early_stopping_counter=5):
        """
        Обучение модели на объединенных целевых данных (реальная и мнимая части).
        """
        self.model.train()
        early_stopping_counter = 0
        best_rmse = float('inf')

        for epoch in range(self.epochs):
            total_rmse = 0
            total_rmse_real = 0
            total_rmse_imag = 0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")

            for batch_idx, (X_batch, times_batch, y_batch) in enumerate(progress_bar):
                X_batch, times_batch, y_batch = X_batch.to(self.device), times_batch.to(self.device), y_batch.to(self.device)

                # Объединение временных меток с входными признаками
                X_with_times = torch.cat((X_batch, times_batch.unsqueeze(1)), dim=1)

                self.optimizer.zero_grad()
                pred = self.model(X_with_times)
                loss = self.criterion(pred, y_batch)
                rmse = torch.sqrt(loss)  # RMSE
                rmse_real = torch.sqrt(self.criterion(pred[:, 0], y_batch[:, 0]))
                rmse_imag = torch.sqrt(self.criterion(pred[:, 1], y_batch[:, 1]))
                
                rmse.backward()
                self.optimizer.step()

                total_rmse += rmse.item()
                total_rmse_real += rmse_real.item()
                total_rmse_imag += rmse_imag.item()

                # Логирование в TensorBoard
                self.writer.add_scalar('Training/RMSE', rmse.item(), epoch * len(self.dataloader) + batch_idx)
                self.writer.add_scalar('Training/RMSE_Real', rmse_real.item(), epoch * len(self.dataloader) + batch_idx)
                self.writer.add_scalar('Training/RMSE_Imag', rmse_imag.item(), epoch * len(self.dataloader) + batch_idx)

                # Получение текущего значения learning rate
                current_learning_rate = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(rmse=f"{rmse:.10f}", lr=f"{current_learning_rate:.6f}")

            avg_rmse = total_rmse / len(self.dataloader)
            avg_rmse_real = total_rmse_real / len(self.dataloader)
            avg_rmse_imag = total_rmse_imag / len(self.dataloader)
            self.history["epoch"].append(epoch + 1)
            self.history["total_rmse"].append(avg_rmse)
            self.history["rmse_real"].append(avg_rmse_real)
            self.history["rmse_imag"].append(avg_rmse_imag)

            # Логирование средней RMSE в TensorBoard
            self.writer.add_scalar('Training/Average_RMSE', avg_rmse, epoch)
            self.writer.add_scalar('Training/Average_RMSE_Real', avg_rmse_real, epoch)
            self.writer.add_scalar('Training/Average_RMSE_Imag', avg_rmse_imag, epoch)
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Логирование распределения весов для каждого слоя
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f'Weights/{name}', param, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

            # Обновление learning rate scheduler
            self.scheduler.step(avg_rmse)

            # Early Stopping
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= max_early_stopping_counter:  # Параметр patience для Early Stopping
                print("Early stopping activated.")
                break

            print(f"Epoch {epoch+1}/{self.epochs};\t AVG RMSE: {avg_rmse:.10f};\t Real RMSE: {avg_rmse_real:.3f};\t Imag RMSE: {avg_rmse_imag:.3f}")
                
            # Освобождение памяти GPU в конце каждой эпохи
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
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
            for X_batch, times_batch, y_batch in self.dataloader:
                X_batch, times_batch, y_batch = X_batch.to(self.device), times_batch.to(self.device), y_batch.to(self.device)
                
                # Объединение временных меток с входными признаками
                X_with_times = torch.cat((X_batch, times_batch.unsqueeze(1)), dim=1)

                pred = self.model(X_with_times)
                
                all_preds.append(pred.cpu().numpy())
                all_true.append(y_batch.cpu().numpy())

        # Конкатенация всех предсказаний и истинных значений
        self.pred = np.concatenate(all_preds)
        self.true = np.concatenate(all_true)
        
        # Извлечение реальной и мнимой частей
        self.pred_real = self.pred[:, 0]
        self.true_real = self.true[:, 0]
        self.pred_imag = self.pred[:, 1]
        self.true_imag = self.true[:, 1]
        
        # Вычисление RMSE для реальной и мнимой части
        rmse_real = np.sqrt(mean_squared_error(self.true_real, self.pred_real))
        rmse_imag = np.sqrt(mean_squared_error(self.true_imag, self.pred_imag))
    
        # Логируем RMSE в TensorBoard
        self.writer.add_scalar('Evaluation/REAL', rmse_real, len(self.history["epoch"]) - 1)
        self.writer.add_scalar('Evaluation/IMAG', rmse_imag, len(self.history["epoch"]) - 1)
            
        # Освобождение памяти GPU (при использовании CUDA)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # Сохранение последних значений RMSE
        self.last_rmse_real = rmse_real
        self.last_rmse_imag = rmse_imag

        return rmse_real, rmse_imag
        
    def save_model_pt(self, save_dir='models'):
        """
        Сохраняет всю модель PyTorch в формате .pt, используя имя эксперимента и индекс.

        Args:
            save_dir (str, optional): Директория для сохранения модели. По умолчанию 'models'.
        """
        # Извлекаем конечную часть пути, чтобы получить имя эксперимента с индексом
        experiment_name_with_index = os.path.basename(self.log_dir)

        # Генерируем имя файла с текущей датой и временем
        filename = f"{experiment_name_with_index}.pt"

        # Полный путь к файлу, без создания дополнительных подпапок
        filepath = os.path.join(save_dir, filename)

        # Убедитесь, что основная директория для сохранения моделей существует
        os.makedirs(save_dir, exist_ok=True)

        # Создаем словарь, включающий модель и тип модели
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'M': self.M,
            'K': self.K,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate
        }

        # Сохраняем модель и метаданные
        torch.save(checkpoint, filepath)
        print(f"Model saved in {filepath}")

    def plot_training_history(self, window_size=5):
        """
        Строит графики истории обучения модели, отображая RMSE на каждой эпохе и скользящее среднее.
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
        # Преобразуем список эпох для оси X
        epochs = self.history["epoch"]
    
        # Вычисляем скользящее среднее
        rmse = np.array(self.history["total_rmse"])
        moving_avg = np.convolve(rmse, np.ones(window_size)/window_size, mode='valid')
    
        # Первый график: Полная история
        axs[0].plot(epochs, rmse, marker='o', linestyle='-', color='b', markersize=5, label='RMSE')
        axs[0].plot(epochs[window_size-1:], moving_avg, color='r', label=f'Moving Average (window size={window_size})')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Average Loss')
        axs[0].set_title('Loss Function (Full History)')
        axs[0].grid(True)
        axs[0].legend()
    
        # Второй график: Половина истории
        mid_index = len(epochs) // 2
        axs[1].plot(epochs[mid_index:], rmse[mid_index:], marker='o', linestyle='-', color='b', markersize=5, label='RMSE')
        axs[1].plot(epochs[mid_index + window_size - 1:], moving_avg[mid_index:], color='r', label=f'Moving Average (window size={window_size})')
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

    def print_model_summary(self, filename_prefix="model_parameters", save_dir='history'):
        """
        Выводит информацию о модели и сохраняет её параметры и их размерности в CSV файл.

        Args:
            filename_prefix (str, optional): Префикс имени файла. По умолчанию 'model_parameters'.
            save_dir (str, optional): Директория для сохранения файла. По умолчанию 'history'.
        """
        # Создаем папку, если ее нет
        os.makedirs(save_dir, exist_ok=True)

        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}_{self.model_id}.csv"

        # Полный путь к файлу
        filepath = os.path.join(save_dir, filename)
        
        df_params = pd.DataFrame(columns=['Parameter name', 'Parameter shape', 'Parameter count'])
        
        print(f"Model architecture: {self.model}")
        print("-" * 50)

        total_params = 0
        for name, param in self.model.named_parameters():
            print(f"Parameter name: {name}")
            print(f"Parameter shape: {param.shape}")
            param_count = torch.numel(param)
            print(f"Parameter count: {param_count}")
            print("-" * 30)

            # Добавляем информацию о параметре в DataFrame
            df_params.loc[len(df_params)] = [name, param.shape, param_count] 
            
            total_params += param_count

        print(f"Total trainable parameters: {total_params}")
        print("=" * 50)
        
        # Сохраняем DataFrame в CSV файл
        # df_params.to_csv(filepath, index=False)
        
        # print(f"Print model saved in {filepath}")
       
    def save_log_for_tensorboard(self):
        # Сохранение эксперимента для Tensorboard
        self.writer = self.initialize_log_dir(self.model_type)
        # Логирование структуры модели
        self.writer.add_graph(self.model, torch.randn(1, self.X.shape[1] + 1).to(self.device))
        
    def initialize_log_dir(self, experiment_name):
        """
        Проверяет существование директории для логов и создаёт новую, если она существует.

        Args:
            experiment_name (str): Имя эксперимента для создания директории.

        Returns:
            SummaryWriter: Экземпляр SummaryWriter для TensorBoard.
        """
        self.log_dir = f'logs/{experiment_name}'  # Базовый путь для логов
        # Проверка существования директории и генерация нового имени, если необходимо
        i = 0
        while os.path.exists(self.log_dir):
            self.log_dir = f'logs/{experiment_name}_{i}'
            i += 1
        os.makedirs(self.log_dir)  # Создание директории
        print(f'The experiment with the name has been saved: {self.log_dir}')
        return SummaryWriter(log_dir=self.log_dir)  # Инициализация SummaryWriter
    
    def log_predictions_to_tensorboard(self):
        """
        Логирование всех предсказанных и фактических значений в TensorBoard.
        """
        self.model.eval()
        pred_real, pred_imag = self.pred[:, 0], self.pred[:, 1]
        true_real, true_imag = self.true[:, 0], self.true[:, 1]

        # Логирование в TensorBoard для реальных значений
        for i in range(len(self.pred)):
            self.writer.add_scalars(
                'Predictions/Real', 
                {
                    'Predicted': pred_real[i],
                    'True': true_real[i],
                },
                global_step=i
            )
    
        # Логирование в TensorBoard для мнимых значений
        for i in range(len(self.pred)):
            self.writer.add_scalars(
                'Predictions/Imag', 
                {
                    'Predicted ': pred_imag[i],
                    'True': true_imag[i]
                },
                global_step=i
            )

        print(f"All predictions logged to TensorBoard.")

    def log_hparams_and_metrics(self, rmse_real=None, rmse_imag=None):
        hparams = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'M': self.M,
            'K': self.K,
            'dropout_rate': self.dropout_rate,
            'num_epochs': self.epochs
        }
            
        # Получение текущего времени в читаемом формате
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}"  # Читаемое имя
    
        # Логирование гиперпараметров и метрик
        self.writer.add_hparams(hparams, { 
            'rmse_real': rmse_real,
            'rmse_imag': rmse_imag 
        }, run_name=run_name)
        
        self.writer.close()
        print(f"Hyperparameters have been successfully saved with the name: {os.path.basename(self.log_dir)}/{run_name}")
        
    def signal_handler(self, signum, frame):
        print("Program interruption! Saving hyperparameters...")
            
        # Освобождение памяти GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print("CUDA memory cleared.")
            
        rmse_real, rmse_imag = self.evaluate()
        print(f"Evaluation RMSE (Real): {rmse_real:.6f}")
        print(f"Evaluation RMSE (Imag): {rmse_imag:.6f}")
        self.log_hparams_and_metrics(rmse_real, rmse_imag)
        print("Hyperparameters are saved. Completion of the program.")
        
        exit(0)