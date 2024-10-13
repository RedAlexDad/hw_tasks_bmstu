import os
import pandas as pd
import numpy as np

import unittest

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from torch.utils.data import TensorDataset, DataLoader
from NeuralODETrainer import NeuralODETrainer


class TestNeuralODEEvaluator(unittest.TestCase):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.dataset_path = 'Amp_C_train.txt'
        self.df = pd.read_csv(self.dataset_path)

        self.model_path = 'models/node_20241013_182741_84692ad9-3e30-41ef-bc44-3ba134a634f2.pt'
        self.save_model_test = 'saved_node_model.pt'
        self.save_path_model_test = f'models/{self.save_model_test}'

        self.batch_size = 1024
        self.learning_rate = 1e-3
        self.epochs = 2
        self.hidden_layers = [2**5, 2**6, 2*5]

    def test_01_train_and_prediction_rmse(self):
        """
        Тестируем, что модель правильно обучена и делает предсказания.
        """
        # Создаем объект NODE с помощью загруженных параметров
        self.NODE_model = NeuralODETrainer(  
            self.df, 
            self.batch_size, 
            self.learning_rate, 
            self.epochs, 
            self.hidden_layers,
            device='cpu'
        )

        # Обучение модели
        self.NODE_model.train()

        self.assertIsNotNone(self.NODE_model, "Модель не была создана")

        # Оценка модели
        rmse_real, rmse_imag, predictions = self.NODE_model.evaluate()

        # Проверяем размер предсказанных данных
        self.assertEqual(len(predictions), len(self.NODE_model.dataset), "Размеры предсказаний и реальных данных должны совпадать.")

        # Выводим метрики RMSE
        print(f"RMSE (реальная часть): {rmse_real}")
        print(f"RMSE (мнимая часть): {rmse_imag}")
        
        try:
            # Проверяем, что RMSE не превышает допустимый порог (например, 1e-5)
            max_limit = 1e-5
            self.assertLess(rmse_real, max_limit, "RMSE для реальной части слишком высок!")
            self.assertLess(rmse_imag, max_limit, "RMSE для мнимой части слишком высок!")
        finally:
            # Сохраняем обученную модель
            torch.save(self.NODE_model, self.save_path_model_test)
            print(f"Модель сохранена в {self.save_path_model_test}")
            
        # Проверяем, что файл сохранен
        self.assertTrue(os.path.exists(self.save_path_model_test), "Файл модели не был сохранен.")


    def test_02_model_load_and_test(self):
        """
        Тестируем загрузку модели, а затем делаем предсказания.
        """
        device = 'cpu'
        # Проверяем, что файл сохранен
        self.assertTrue(os.path.exists(self.save_path_model_test), "Файл модели не был сохранен.")
        NODE_model_test = torch.load(self.save_path_model_test, map_location=device)

        # Извлекаем модель из объекта NeuralODETrainer
        NODE_model = NODE_model_test.model  # Получаем модель

        NODE_model.to(device)  # Вызываем to() для модели, а не для NeuralODETrainer
        NODE_model.eval()
        print("Модель успешно загружена.")

        df = pd.read_csv(self.dataset_path)
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

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        """Оценивает модель на подготовленных данных и выводит метрики."""
        if not NODE_model:
            raise RuntimeError("Model is not loaded. Call 'load_model()' first.")
        if not data_loader:
            raise RuntimeError("Data is not prepared. Call 'prepare_data()' first.")
        
        true_values = []
        predicted_values = []
        times = []

        with torch.no_grad():
            for t, inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                t = t.to(device)
                outputs = NODE_model(inputs, t)
                true_values.append(targets.cpu().numpy())
                predicted_values.append(outputs.cpu().numpy())
                times.append(t.cpu().numpy())

        true_values = np.concatenate(true_values, axis=0)
        predicted_values = np.concatenate(predicted_values, axis=0)

        # Проверяем размер предсказанных данных
        self.assertEqual(len(predicted_values), len(NODE_model_test.dataset), "Размеры предсказаний и реальных данных должны совпадать.")

        rmse_real = mean_squared_error(true_values[:, 0], predicted_values[:, 0], squared=False)
        rmse_imag = mean_squared_error(true_values[:, 1], predicted_values[:, 1], squared=False)

        # Выводим метрики RMSE
        print(f"RMSE (реальная часть, загруженная модель): {rmse_real}")
        print(f"RMSE (мнимая часть, загруженная модель): {rmse_imag}")
        
        try:
            # Проверяем, что RMSE приемлем
            max_limit = 1e-5
            self.assertLess(rmse_real, max_limit, "RMSE для реальной части после загрузки слишком высок!")
            self.assertLess(rmse_imag, max_limit, "RMSE для мнимой части после загрузки слишком высок!")
        finally:
            if os.path.exists(self.save_path_model_test):
                os.remove(self.save_path_model_test)


# python3 -m unittest test.py

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestNeuralODEEvaluator('test_01_train_and_prediction_rmse'))
    suite.addTest(TestNeuralODEEvaluator('test_02_model_load_and_test'))

    runner = unittest.TextTestRunner()
    runner.run(suite)