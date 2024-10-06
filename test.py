import unittest
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from node_model import NeuralODETrainer
from test_model import F


class TestCTDRNN(unittest.TestCase):
    
    def setUp(self):
        """
        Этот метод будет выполнен перед каждым тестом.
        Загружаем датасет и создаем объект модели.
        """
        # Загрузка данных из файла
        dataset_path = 'Amp_C_train.txt'
        model_path = "models/node_20241006_124330_kaggle.pth"
        self.df = pd.read_csv(dataset_path)
        batch_size=1024*5
        learning_rate=1e-5
        epochs=1000
        hidden_size=512

        # Создаем объект NODE с помощью загруженных параметров
        self.NODE_model = NeuralODETrainer(self.df, batch_size, learning_rate, epochs, hidden_size)

        # Создаем экземпляр модели
        self.NODE_model.model = self.NODE_model.NeuralODEModel(hidden_size=self.NODE_model.hidden_size).to(self.NODE_model.get_device())

        # Загружаем веса модели на CPU
        self.NODE_model.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.NODE_model.model.eval()  # Вызываем eval() у модели


    def test_prediction_and_rmse(self):
        """
        Тестируем, что модель правильно делает предсказания и вычисляем метрику RMSE.
        """
        X_in = self.df['Input'].values.astype(np.complex64)
        Y_true = self.df['Output'].values.astype(np.complex64)

        # Подготавливаем и делаем предсказания
        predictions, rmse_real, rmse_imag = F(X_in, self.NODE_model.model, self.NODE_model.train_loader) 
        
        # Проверяем размер предсказанных данных
        self.assertEqual(len(predictions), len(Y_true), "Размеры предсказаний и реальных данных должны совпадать.")

        # Выводим метрики RMSE
        print(f"RMSE (реальная часть): {rmse_real}")
        print(f"RMSE (мнимая часть): {rmse_imag}")
        
        # Проверяем, что RMSE не превышает допустимый порог (например, 1e-5)
        max_limit = 1e-5
        self.assertLess(rmse_real, max_limit, "RMSE для реальной части слишком высок!")
        self.assertLess(rmse_imag, max_limit, "RMSE для мнимой части слишком высок!")

    def test_model_save_and_load(self):
        """
        Тестируем сохранение и загрузку модели, а затем делаем предсказания.
        """
        # Сохраняем обученную модель
        save_path = 'models/saved_node_model.pth'

        torch.save(self.NODE_model.model.state_dict(), save_path)
        print(f"Модель сохранена в {save_path}")
        
        # Проверяем, что файл сохранен
        self.assertTrue(os.path.exists(save_path), "Файл модели не был сохранен.")

        batch_size=1024*5
        learning_rate=1e-5
        epochs=1000
        hidden_size=512

        # Создаем объект NODE с помощью загруженных параметров
        NODE_model = NeuralODETrainer(self.df, batch_size, learning_rate, epochs, hidden_size)

        # Создаем экземпляр модели
        NODE_model.model = NODE_model.NeuralODEModel(hidden_size=NODE_model.hidden_size).to(NODE_model.get_device())

        # Загружаем веса модели на CPU
        NODE_model.model.load_state_dict(torch.load(save_path, map_location='cpu'))
        NODE_model.model.eval()  
        print("Модель успешно загружена.")

        X_in = self.df['Input'].values.astype(np.complex64)
        # Истинные значения для вычисления метрики RMSE
        Y_true = self.df['Output'].values.astype(np.complex64)

        # Подготавливаем и делаем предсказания
        predictions, rmse_real, rmse_imag = F(X_in, NODE_model.model, NODE_model.train_loader) 

        # Выводим метрики RMSE
        print(f"RMSE (реальная часть, загруженная модель): {rmse_real}")
        print(f"RMSE (мнимая часть, загруженная модель): {rmse_imag}")
        
        # Проверяем, что RMSE приемлем
        max_limit = 1e-5
        self.assertLess(rmse_real, max_limit, "RMSE для реальной части после загрузки слишком высок!")
        self.assertLess(rmse_imag, max_limit, "RMSE для мнимой части после загрузки слишком высок!")

        # Удаляем файл модели после теста
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == '__main__':
    unittest.main()