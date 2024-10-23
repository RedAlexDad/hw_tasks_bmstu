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
        X, y_real, y_imag, times = self.create_dataset(df, M, K)
        # Создаем датасет, объединяя реальную и мнимую части в один тензор
        
        self.dataset = TensorDataset(
            torch.tensor(times, dtype=torch.float32),
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(np.stack([y_real, y_imag], axis=1), dtype=torch.float32) # Объединяем реальную и мнимую части
        )
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        # Заменяем предыдущую сеть на SimpleMLP
        self.model = self.SimpleMLP(input_size=X.shape[1], hidden_layers=hidden_layers, output_size=2).to(self.device)
        # Оптимизатор и функция потерь
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

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
        df = df.set_index('time')  # Устанавливаем временной ряд как индекс
        return df

    @staticmethod
    def get_device( select=None):
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
        Создание обучающих данных на основе полиномиальной модели с памятью.

        Args:
            df (pd.DataFrame): Обработанные данные.

        Returns:
            tuple: Матрица признаков X, реальные значения y_real и мнимые значения y_imag.
        """
        x_real = df['input_real'].values
        x_imag = df['input_imag'].values
        y_real = df['output_real'].values[M:]
        y_imag = df['output_imag'].values[M:]

        N = len(x_real)
        X = np.zeros((N, (M + 1) * K * 2), dtype=np.float64)
        
        for n in range(M, N):
            index = 0
            for m in range(M + 1):
                for k in range(1, K + 1):
                    X[n, index] = np.abs(x_real[n - m])**(k-1) * x_real[n - m]
                    X[n, index + 1] = np.abs(x_imag[n - m])**(k-1) * x_imag[n - m]
                    index += 2
                    
        times = np.arange(M, N)  # Генерируем временные метки для каждого наблюдения
        
        # Проверка порядка временных меток
        assert np.all(np.diff(times) >= 0), "Временные метки не отсортированы!"

        return X[M:], y_real, y_imag, times

    def train(self):
        """
        Обучение модели на реальной и мнимой частях данных.
        """
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")
                
            for batch_idx, (X_batch, y_batch, times_batch) in enumerate(progress_bar):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)  # Сравнение объединенных реальной и мнимой частей
                loss.backward()
                self.optimizer.step()
    
                total_loss += loss.item()
                
                progress_bar.set_postfix(loss=f'{loss:.10f}')

            avg_loss = total_loss / len(self.train_loader)
            
            self.history["epoch"].append(epoch + 1)
            self.history["rmse"].append(np.sqrt(avg_loss))

            print(f"Epoch {epoch+1}/{self.epochs}, Avg loss: {avg_loss:.6f}")

    def evaluate(self):
        """
        Оценка модели после обучения.
        
        Returns:
            tuple: Значения RMSE для реальной и мнимой частей.
        """
        self.model.eval()  # Используем только одну модель теперь
        all_preds = []
        all_true = []
        times = []

        with torch.no_grad():
            for X_batch, y_batch, times_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
                # Предсказание модели
                predictions = self.model(X_batch)
                
                all_preds.append(predictions.cpu().numpy())
                all_true.append(y_batch.cpu().numpy())
                times.append(times_batch.cpu().numpy())

        # Конкатенация всех предсказаний и истинных значений
        self.predictions = np.concatenate(all_preds)
        self.true_values = np.concatenate(all_true)
        self.times = np.concatenate(times)  # Сохранение временных меток

        # Разделение реальной и мнимой частей
        pred_real = self.predictions[:, 0]
        pred_imag = self.predictions[:, 1]
        true_real = self.true_values[:, 0]
        true_imag = self.true_values[:, 1]
    
        # Вычисление RMSE для реальной и мнимой частей
        rmse_real = np.sqrt(mean_squared_error(true_real, pred_real))
        rmse_imag = np.sqrt(mean_squared_error(true_imag, pred_imag))
    
        print(f"Evaluation RMSE Real: {rmse_real:.6f}, RMSE Imag: {rmse_imag:.6f}")
        return rmse_real, rmse_imag

    def save_model_pt(self, filename_prefix='node', save_dir='models'):
        """
        Сохраняет всю модель PyTorch в формате .pt.

        Args:
            filename_prefix (str, optional): Префикс имени файла. По умолчанию 'node'.
            save_dir (str, optional): Директория для сохранения модели. По умолчанию 'models'.
        """
        # Создаем папку, если ее нет
        os.makedirs(save_dir, exist_ok=True)

        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}_{self.model_id}.pt"

        # Полный путь к файлу
        filepath = os.path.join(save_dir, filename)

        os.makedirs(path, exist_ok=True)
        torch.save(self.model_real.state_dict(), os.path.join(path, f"model_real_{self.model_id}.pt"))
        torch.save(self.model_imag.state_dict(), os.path.join(path, f"model_imag_{self.model_id}.pt"))
        print(f"Models saved in {path}")
     
    def save_prediction(self, predictions, filename_prefix="predictions", save_dir='history'):
        # Создаем папку, если ее нет
        os.makedirs(save_dir, exist_ok=True)

        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}_{self.model_id}.csv"

        # Полный путь к файлу
        filepath = os.path.join(save_dir, filename)
        
        df_predictions = pd.DataFrame({'real': predictions[:, 0], 'imag': predictions[:, 1],})
        df_predictions.to_csv(filepath, index=False)
        print(f"Training history saved to {filepath}")
         
    def save_training_history(self, filename_prefix="training_history", save_dir='history'):
        # Создаем папку, если ее нет
        os.makedirs(save_dir, exist_ok=True)

        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}_{self.model_id}.csv"

        # Полный путь к файлу
        filepath = os.path.join(save_dir, filename)
        
        df_history = pd.DataFrame(self.history)
        df_history.to_csv(filepath, index=False)
        print(f"Training history saved to {filepath}")
        
    def plot_training_history(self):
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

    def plot_predictions(self, time_start=0, time_end=1.005e2):
        """
        Построение графиков предсказанных и истинных значений для реальной и мнимой частей.
    
        Args:
            time_start (float, optional): Начальное время для фильтрации данных. По умолчанию 0.
            time_end (float, optional): Конечное время для фильтрации данных. По умолчанию 1.005e2.
        """
        # Проверяем, что оценка была выполнена, и данные сохранены
        if not hasattr(self, 'times') or not hasattr(self, 'predictions') or not hasattr(self, 'true_values'):
            raise ValueError("Необходимо сначала выполнить evaluate(), чтобы сохранить предсказанные значения.")
    
        # Фильтрация данных по времени
        mask = (self.times >= time_start) & (self.times <= time_end)
        filtered_times = self.times[mask]
        # Сортировка данных по времени
        sorted_indices = np.argsort(filtered_times)
        
        # Разделение истинных и предсказанных значений для реальной и мнимой частей
        filtered_true_real = self.true_values[mask, 0][sorted_indices]
        filtered_true_imag = self.true_values[mask, 1][sorted_indices]
        filtered_pred_real = self.predictions[mask, 0][sorted_indices]
        filtered_pred_imag = self.predictions[mask, 1][sorted_indices]

        # Построение графика для реальных значений
        plt.figure(figsize=(20, 5))
    
        plt.subplot(1, 2, 1)  # 1 ряд, 2 колонки, 1-й график
        plt.plot(filtered_times, filtered_true_real, label="True Real", linestyle='-', color='red')
        plt.plot(filtered_times, filtered_pred_real, label="Predicted Real", linestyle='-', color='blue')
        plt.legend()
        plt.title("True vs Predicted Real Values")
        plt.xlabel("Time")
        plt.ylabel("Real Value")
    
        # Построение графика для мнимых значений
        plt.subplot(1, 2, 2)  # 1 ряд, 2 колонки, 2-й график
        plt.plot(filtered_times, filtered_true_imag, label="True Imag", linestyle='-', color='red')
        plt.plot(filtered_times, filtered_pred_imag, label="Predicted Imag", linestyle='-', color='blue')
        plt.legend()
        plt.title("True vs Predicted Imaginary Values")
        plt.xlabel("Time")
        plt.ylabel("Imaginary Value")
    
        plt.tight_layout()
        plt.show()
        
    def print_model_summary(self, filename_prefix="model_parameters", save_dir='history'):
        """
        Выводит информацию о модели: архитектуру, параметры и их размерности.
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
        df_params.to_csv(filepath, index=False)


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
    
    