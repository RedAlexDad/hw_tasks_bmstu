import sys
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from NeuralODETrainer import NeuralODETrainer


class NeuralODEEvaluator:
    def __init__(self, model_path, input_file, batch_size=1024, device=None):
        """
        Инициализирует класс оценки Neural ODE.
        
        Args:
            model_path (str): Путь к файлу сохраненной модели.
            input_file (str): Путь к файлу с входными данными.
            device (str, optional): Устройство для загрузки модели ('cpu' или 'cuda').
                                     По умолчанию 'cpu'.
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.input_file = input_file
        self.loaded_model = None
        self.data_loader = None
        self.true_values = []
        self.predicted_values = []
        self.times = []
        self.batch_size = batch_size
        
    def load_model(self):
        """Загружает модель с указанного пути."""
        print(f"Using device: {self.device}")
        self.loaded_model = torch.load(self.model_path, map_location=self.device)
        self.loaded_model.to(self.device)
        self.loaded_model.eval()
    
    def prepare_data(self):
        """Загружает и подготавливает данные из файла."""
        df = pd.read_csv(self.input_file)
        df['Input'] = df['Input'].apply(lambda x: complex(x))
        df['Output'] = df['Output'].apply(lambda x: complex(x))
        df['input_real'] = df['Input'].apply(lambda x: x.real)
        df['input_imag'] = df['Input'].apply(lambda x: x.imag)
        df['output_real'] = df['Output'].apply(lambda x: x.real)
        df['output_imag'] = df['Output'].apply(lambda x: x.imag)
        df = df.drop(['Input', 'Output'], axis=1)
        df = df.set_index('Time')
        
        # Создаем DataLoader
        dataset = TensorDataset(
            torch.tensor(df.index.values, dtype=torch.float32),
            torch.tensor(df[['input_real', 'input_imag']].values, dtype=torch.float32),
            torch.tensor(df[['output_real', 'output_imag']].values, dtype=torch.float32),
        )
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    def evaluate(self):
        """Оценивает модель на подготовленных данных и выводит метрики."""
        if not self.loaded_model:
            raise RuntimeError("Model is not loaded. Call 'load_model()' first.")
        if not self.data_loader:
            raise RuntimeError("Data is not prepared. Call 'prepare_data()' first.")
        
        with torch.no_grad():
            for t, inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                t = t.to(self.device)
                outputs = self.loaded_model(inputs, t)
                self.true_values.append(targets.cpu().numpy())
                self.predicted_values.append(outputs.cpu().numpy())
                self.times.append(t.cpu().numpy())

        self.true_values = np.concatenate(self.true_values, axis=0)
        self.predicted_values = np.concatenate(self.predicted_values, axis=0)

        rmse_real = mean_squared_error(self.true_values[:, 0], self.predicted_values[:, 0], squared=False)
        rmse_imag = mean_squared_error(self.true_values[:, 1], self.predicted_values[:, 1], squared=False)

        return rmse_real, rmse_imag, self.predicted_values


if __name__ == "__main__":
    if len(sys.argv) < 3:
        # print("Usage: python3 main.py dataset.csv path_to_pretrained_model.pth")
        input_file = "Amp_C_train.txt"
        model_path = "models/node_20241013_182741_84692ad9-3e30-41ef-bc44-3ba134a634f2.pt"
    else:        
        input_file = sys.argv[1]
        model_path = sys.argv[2]
    
    # Инициализируем класс оценки Neural ODE
    evaluator = NeuralODEEvaluator(model_path, input_file)
    
    # Загружаем модель
    evaluator.load_model()
    
    # Подготавливаем данные
    evaluator.prepare_data()

    # Оцениваем модель 
    rmse_real, rmse_imag, predictions = evaluator.evaluate()
    # print(f"Evaluation RMSE (Real): {rmse_real:.6f}")
    # print(f"Evaluation RMSE (Imaginary): {rmse_imag:.6f}")

    # Создаем DataFrame из предсказаний
    df_predictions = pd.DataFrame(predictions, columns=['pred_real', 'pred_imag'])
    # Добавляем столбец с индексом
    df_predictions['index'] = np.arange(len(predictions))
    # Сохраняем в CSV
    df_predictions.to_csv('history/predictions_test.csv', index=False)
    # Выводим результаты
    print(predictions)