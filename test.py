import unittest
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

# Импорт тестируемых функций/классов из ваших модулей
from CTDRNN import CTDRNN, CTDRNN_Model



class TestCTDRNN(unittest.TestCase):
    
    def setUp(self):
        """
        Этот метод будет выполнен перед каждым тестом.
        Загружаем датасет и создаем объект модели.
        """
        # Загрузка данных из файла
        dataset_path = 'Amp_C_train.txt'
        df = pd.read_csv(dataset_path)
        df.columns = df.columns.str.lower()
        df.set_index('time', inplace=True)
        df['input'] = df['input'].apply(lambda x: complex(x.strip('()').replace('i', 'j')))
        df['output'] = df['output'].apply(lambda x: complex(x.strip('()').replace('i', 'j')))
        df['input_real'] = df['input'].apply(np.real)
        df['input_imag'] = df['input'].apply(np.imag)
        df['output_real'] = df['output'].apply(np.real)
        df['output_imag'] = df['output'].apply(np.imag)

        # Настройка параметров модели
        p=10
        q=5
        input_size = 2 * (p + 1 + q)
        hidden_sizes = [50, 25]
        output_size = 2  # Вещественная и мнимая части выходного сигнала
        num_layers = 2
        num_epochs = 100
        learning_rate = 0.001
        batch_size = 64

        # Создание модели
        self.model = CTDRNN(input_size, hidden_sizes, output_size, num_layers, num_epochs, learning_rate, p=10, q=5, batch_size=batch_size)
        
        # Подготовка данных
        self.X_seq, self.Y_seq = self.model.prepare_data(df)

    def test_prediction_and_rmse(self):
        """
        Тестируем, что модель правильно делает предсказания и вычисляем метрику RMSE.
        """
        # Делаем предсказания
        Y_pred = self.model.predict(self.X_seq)
        
        # Проверяем размер предсказанных данных
        self.assertEqual(Y_pred.shape, self.Y_seq.shape, "Размеры предсказаний и реальных данных должны совпадать.")
        
        # Вычисляем RMSE для вещественной и мнимой частей
        rmse_real, rmse_imag = self.model.evaluate(self.Y_seq, Y_pred)
        
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
        save_path = 'saved_ctdrnn_model.pth'
        # Сохранение модели с параметрами
        checkpoint = {
            'model_state_dict': self.model.model.state_dict(),
            'input_size': self.model.input_size,
            'hidden_sizes': self.model.hidden_sizes,
            'output_size': self.model.output_size,
            'num_layers': self.model.num_layers,
            'p': self.model.p,
            'q': self.model.q
        }

        torch.save(checkpoint, 'saved_ctdrnn_model.pth')
        print(f"Модель сохранена в {save_path}")
        
        # Проверяем, что файл сохранен
        self.assertTrue(os.path.exists(save_path), "Файл модели не был сохранен.")

        # Загрузка модели с параметрами
        checkpoint = torch.load('saved_ctdrnn_model.pth', map_location=lambda storage, loc: storage)

        # Извлечение параметров модели
        input_size = checkpoint['input_size']
        hidden_sizes = checkpoint['hidden_sizes']
        output_size = checkpoint['output_size']
        num_layers = checkpoint['num_layers']
        p = checkpoint['p']
        q = checkpoint['q']

        # Восстановление модели с извлеченными параметрами
        loaded_model = CTDRNN_Model(input_size, hidden_sizes, output_size, num_layers)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        loaded_model.eval()
        print("Модель успешно загружена.")

        # Делаем предсказания с загруженной моделью
        with torch.no_grad():
            X_tensor = torch.tensor(self.model.scaler_X.transform(self.X_seq), dtype=torch.float32).unsqueeze(1)
            Y_pred_scaled = loaded_model(X_tensor).cpu().numpy()
            Y_pred = self.model.scaler_Y.inverse_transform(Y_pred_scaled)
        
        # Проверяем, что размер предсказаний совпадает
        self.assertEqual(Y_pred.shape, self.Y_seq.shape, "Размеры предсказаний после загрузки модели должны совпадать с реальными данными.")
        
        # Вычисляем RMSE для загруженной модели
        rmse_real, rmse_imag = self.model.evaluate(self.Y_seq, Y_pred)
        
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
