import os
import uuid
import pandas as pd
import numpy as np

import unittest

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from torch.utils.data import TensorDataset, DataLoader
from CTDRNN import CTDRNN
from test_ctdrnn import CTDRNNEvaluator          

class TestCTDRNNEvaluator(unittest.TestCase):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.input_file = 'Amp_C_train.txt'
        self.model_path = 'models/saved_ctdrnn_model.pt'

        self.batch_size = 1024
        self.learning_rate = 1e-3
        self.epochs = 2
        self.hidden_layers = [2**5, 2**6, 2*5]
        # Генерация уникального ID при создании экземпляра класса
        self.model_id = str(uuid.uuid4())

    def test_01_train_and_prediction_rmse(self):
        """
        Тестируем, что модель правильно обучена и делает предсказания.
        """
        # Создаем объект CTRNN с помощью загруженных параметров
        # Параметры задержки p
        p=25 # Входная задержка
        q=15  # Выходная задержка

        # Создадим объект CTRNN
        self.ctdrnn = CTDRNN(
            df=pd.read_csv(self.input_file),
            input_size=2 * (p + 1 + q), 
            hidden_layers=[2**4, 2**4],
            output_size=2, 
            epochs=1,
            learning_rate=1e-4,
            p=p,
            q=q,
            batch_size=1024*1
        )
        self.assertIsNotNone(self.ctdrnn, "Модель не была создана")
        # Обучение модели
        self.ctdrnn.train()
        # Оценка модели
        rmse_real, rmse_imag, predictions = self.ctdrnn.evaluate()
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
            torch.save({
                'model_state_dict': self.ctdrnn.model.state_dict(),
                'input_size': self.ctdrnn.input_size,
                'hidden_layers': self.ctdrnn.hidden_layers,
                'output_size': self.ctdrnn.output_size,
                'learning_rate': self.ctdrnn.learning_rate,
                'p': self.ctdrnn.p,
                'q': self.ctdrnn.q,
                'batch_size': self.ctdrnn.batch_size,
                'model_id': self.model_id,
            }, self.model_path)

    def test_02_model_load_and_test(self):
        """
        Тестируем загрузку модели, а затем делаем предсказания.
        """
        # Проверяем, что модель была сохранена
        self.assertTrue(os.path.exists(self.model_path), "Файл модели не был сохранен.")
                
        # Инициализируем класс оценки CTDRNN
        evaluator = CTDRNNEvaluator(self.input_file, self.model_path, device='cpu')

        if not evaluator.load_model:
            raise RuntimeError("Model is not loaded. Call 'load_model()' first.")
        if not evaluator.data_loader:
            raise RuntimeError("Data is not prepared. Call 'prepare_data()' first.")

        # Оцениваем модель 
        rmse_real, rmse_imag, predictions = evaluator.evaluate()
        
        rmse_real = mean_squared_error(evaluator.true_values[:, 0], evaluator.predicted_values[:, 0], squared=False)
        rmse_imag = mean_squared_error(evaluator.true_values[:, 1], evaluator.predicted_values[:, 1], squared=False)

        # Выводим метрики RMSE
        print(f"RMSE (реальная часть, загруженная модель): {rmse_real}")
        print(f"RMSE (мнимая часть, загруженная модель): {rmse_imag}")
        
        try:
            # Проверяем, что RMSE приемлем
            max_limit = 1e-5
            self.assertLess(rmse_real, max_limit, "RMSE для реальной части после загрузки слишком высок!")
            self.assertLess(rmse_imag, max_limit, "RMSE для мнимой части после загрузки слишком высок!")
        finally:
            if os.path.exists(self.model_path):
                os.remove(self.model_path)


# python3 -m unittest test.py

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestCTDRNNEvaluator('test_01_train_and_prediction_rmse'))
    suite.addTest(TestCTDRNNEvaluator('test_02_model_load_and_test'))

    runner = unittest.TextTestRunner()
    runner.run(suite)