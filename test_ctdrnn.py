import os
import sys
import uuid
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from CTDRNN import CTDRNN

class CTDRNNEvaluator:
    def __init__(self, input_file: str, model_path: str, device:str =None):
        """
        Инициализирует класс оценки CTDRNN.
        
        Args:
            input_file (str): Путь к файлу с входными данными.
            model_path (str): Путь к файлу сохраненной модели.
            device (str, optional): Устройство для загрузки модели ('cpu' или 'cuda').
                                     По умолчанию 'cpu'.
        """
        self.input_file = input_file
        self.df = self.prepare_data(pd.read_csv(input_file))
        self.model_path = model_path
        self.device = self.get_device(device)

        # Загружаем модель
        self.load_model()

        # Подготовка данных
        self.X_combined, self.Y_combined = self.create_time_delays_with_feedback(self.df, self.p, self.q, self.input_size, self.batch_size)
        # Обновление индекса
        time_index = self.df.index.values[max(self.p, self.q):len(self.X_combined) + max(self.p, self.q)]
        self.train_dataset = TensorDataset(
            torch.tensor(time_index, dtype=torch.float32), 
            torch.tensor(self.X_combined, dtype=torch.float32).unsqueeze(1), 
            torch.tensor(self.Y_combined, dtype=torch.float32)
        )
        self.data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        self.true_values = []
        self.predicted_values = []
        self.times = []
        
    def load_model(self):
        """Загружает модель с указанного пути."""
        # print(f"Using device: {self.device}")
        checkpoint = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        
        self.input_size = checkpoint['input_size']
        self.hidden_layers = checkpoint['hidden_layers']
        self.output_size = checkpoint['output_size']
        self.learning_rate = checkpoint['learning_rate']
        self.p = checkpoint['p']
        self.q = checkpoint['q']
        self.batch_size = checkpoint['batch_size']
        self.model_id = checkpoint['model_id']

        # Воссоздаем модель с теми же параметрами
        self.loaded_model = CTDRNN.CTDRNN_Model(
            input_size=self.input_size,
            hidden_layers=self.hidden_layers,
            output_size=self.output_size,
        ).to(self.device)

        self.loaded_model.load_state_dict(checkpoint['model_state_dict'])  # Choose whatever GPU device number you want
        self.loaded_model.to(self.device)
        self.loaded_model.eval()
    
    @staticmethod
    def get_device(select: str=None):
        """
        Получает устройство для вычислений (CPU или GPU).

        Args:
            select (str, optional): Выбор устройства ('cpu', 'gpu' или 'cuda'). По умолчанию None.

        Returns:
            torch.device: Устройство для выполнения вычислений.
        """
        if select is None or select == 'gpu' or select == 'cuda':
            if torch.cuda.is_available():
                # print('Using GPU (CUDA)')
                return torch.device('cuda')
            else:
                # print("CUDA not available, falling back to CPU.")
                return torch.device('cpu')
        # CPU
        else:
            # print('Using CPU')
            return torch.device('cpu')
    
    @staticmethod
    def prepare_data(df):
        """Загружает и подготавливает данные из файла."""
        # Приведение к нижнему регистру названия колонки
        df.columns = df.columns.str.lower()
        df['input'] = df['input'].apply(lambda x: complex(x))  # Преобразуем строки в комплексные числа
        df['output'] = df['output'].apply(lambda x: complex(x))
        df['input_real'] = df['input'].apply(lambda x: x.real)
        df['input_imag'] = df['input'].apply(lambda x: x.imag)
        df['output_real'] = df['output'].apply(lambda x: x.real)
        df['output_imag'] = df['output'].apply(lambda x: x.imag)
        df = df.drop(['input', 'output'], axis=1)
        df = df.set_index('time')  # Устанавливаем временной ряд как индекс
        
        return df

    @staticmethod
    def create_time_delays_with_feedback(df, p:int, q:int, input_size:int, batch_size:int):
        """
        Создает временные задержки с обратной связью с использованием apply.

        Args:
            df (pd.DataFrame): DataFrame, содержащий 'input_real', 'input_imag', 'output_real', 'output_imag'.
            p (int): Порядок задержки для входных данных.
            q (int): Порядок задержки для обратной связи.
            input_size (int): Размер входных данных (не используется в текущей реализации).
            batch_size (int): Размер пакета (не используется в текущей реализации).

        Returns:
            X_combine, Y_bicombine (np.array): Векторизованные входные и обратные задержки.
        """
        X_real, X_imag, Y_real, Y_imag = df['input_real'], df['input_imag'], df['output_real'], df['output_imag']
        
        # Используем np.array для векторизации данных
        X_real_arr = X_real.values
        X_imag_arr = X_imag.values
        Y_real_arr = Y_real.values
        Y_imag_arr = Y_imag.values

        # Входные задержки
        X_real_delays = np.lib.stride_tricks.sliding_window_view(X_real_arr, p+1)
        X_imag_delays = np.lib.stride_tricks.sliding_window_view(X_imag_arr, p+1)
        
        # Обратная связь
        Y_real_delays = np.lib.stride_tricks.sliding_window_view(Y_real_arr, q)
        Y_imag_delays = np.lib.stride_tricks.sliding_window_view(Y_imag_arr, q)

        # Приводим все массивы к одному размеру
        min_size = min(len(X_real_delays), len(Y_real_delays))
        X_real_delays = X_real_delays[-min_size:]
        X_imag_delays = X_imag_delays[-min_size:]
        Y_real_delays = Y_real_delays[-min_size:]
        Y_imag_delays = Y_imag_delays[-min_size:]

        # Комбинирование задержек
        X_combined = np.hstack([X_real_delays, X_imag_delays, Y_real_delays, Y_imag_delays])

        # Текущие значения выходных данных
        Y_combined = np.column_stack([Y_real_arr[max(p, q):], Y_imag_arr[max(p, q):]])

        return X_combined, Y_combined

    def evaluate(self):
        """Оценивает модель на подготовленных данных и выводит метрики."""
        if not self.loaded_model:
            raise RuntimeError("Model is not loaded. Call 'load_model()' first.")
        if not self.data_loader:
            raise RuntimeError("Data is not prepared. Call 'prepare_data()' first.")
        
        with torch.no_grad():
            for t, inputs, targets in self.data_loader:
                t, inputs, targets = t.to(self.device), inputs.to(self.device), targets.to(self.device)
                outputs = self.loaded_model(inputs, t)
                self.true_values.append(targets.cpu().numpy())
                self.predicted_values.append(outputs.cpu().numpy())
                self.times.append(t.cpu().numpy())

        self.true_values = np.concatenate(self.true_values, axis=0)
        self.predicted_values = np.concatenate(self.predicted_values, axis=0)
        self.times = np.concatenate(self.times, axis=0)

        # Сортировка данных по временам
        sort_indices = np.argsort(self.times)
        self.times = self.times[sort_indices]
        self.true_values = self.true_values[sort_indices]
        self.predicted_values = self.predicted_values[sort_indices]

        # Проверяем размерности после удаления лишних размерностей
        assert self.true_values.shape[1] == 2, f"True values should have shape (n_samples, 2), but got {self.true_values.shape}"
        assert self.predicted_values.shape[1] == 2, f"Predicted values should have shape (n_samples, 2), but got {self.predicted_values.shape}"

        rmse_real = mean_squared_error(self.true_values[:, 0], self.predicted_values[:, 0], squared=False)
        rmse_imag = mean_squared_error(self.true_values[:, 1], self.predicted_values[:, 1], squared=False)

        return rmse_real, rmse_imag, self.predicted_values

    def save_prediction(self, predictions:np.ndarray, filename_prefix:str="predictions_test", save_dir:str='history'):
        """
        Сохраняет предсказанные значения в формате CSV.

        Args:
            predictions_test (np.ndarray): Предсказанные значения.
            filename_prefix (str, optional): Префикс имени файла. По умолчанию 'predictions'.
            save_dir (str, optional): Директория для сохранения файла. По умолчанию 'history'.
        """
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

# Пример ввода командной строки
# python3 test_ctdrnn.py --input_file Amp_C_train.txt --model_path models/ctdrnn_20241017_013453_c470d2c7-3e4a-4617-90cb-df6f48afe721.pt --device 'cpu'

if __name__ == "__main__":
    # Инициализация парсера аргументов
    parser = argparse.ArgumentParser(description='Evaluate CTDRNN model.')
    # Добавление аргументов
    parser.add_argument(
        '--input_file', type=str, default='Amp_C_train.txt',
        help='Path to the dataset file (default: Amp_C_train.txt)'
    )
    parser.add_argument(
        '--model_path', type=str, default='models/ctdrnn_20241017_013453_c470d2c7-3e4a-4617-90cb-df6f48afe721.pt',
        help='Path to the pretrained model file (default: models/...)'
    )
    parser.add_argument(
        '--device', type=str, default='cpu', choices=['cpu', 'cuda'],
        help='Device to use (cpu or cuda, default: cpu)'
    )
    # Парсинг аргументов
    args = parser.parse_args()
    
    # Инициализируем класс оценки CTDRNN
    evaluator = CTDRNNEvaluator(args.input_file, args.model_path, args.device)
        
    # Оцениваем модель 
    rmse_real, rmse_imag, predictions = evaluator.evaluate()
    # print(f"Evaluation RMSE (Real): {rmse_real:.6f}")
    # print(f"Evaluation RMSE (Imaginary): {rmse_imag:.6f}")

    # Выводим результаты
    print(predictions)

    # Сохраняем предсказанные значения в CSV
    evaluator.save_prediction(predictions)
