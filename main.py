import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Разбиение на обучающую, валидационную и тестовую выборку
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, TimeSeriesSplit
# Метрики
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from CTDRNN import CTDRNN

if __name__ == '__main__':
    df = pd.read_csv('Amp_C_train.txt')
    # Проверка наличия CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print("CUDA is available!")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"Driver version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Running on CPU.")

    # Параметры задержки p
    p=25 # Входная задержка
    q=15  # Выходная задержка

    # Создадим объект CTRNN
    ctdrnn = CTDRNN(
        df=df,
        input_size=2 * (p + 1 + q), 
        # hidden_layers=[2**4, 2**6, 2**8, 2**6, 2**4],
        hidden_layers=[2**8, 2**10, 2**12, 2**10, 2**8],
        # hidden_layers=[2**4, 2**4],
        output_size=2, 
        epochs=10,
        learning_rate=1e-4,
        p=p,
        q=q,
        batch_size=1024*1
    )
    # Вывод параметров в виде HTML-таблицы
    ctdrnn.print_params()
    # Обучим модель
    # ctdrnn.train()
    ctdrnn.train_and_save_plots()
    # Сохраним историю обучения
    ctdrnn.save_training_history()
    # График обучения
    ctdrnn.plot_training_history()
    # Оценка модели
    rmse_real, rmse_imag, predictions = ctdrnn.evaluate()
    print(f"Evaluation RMSE (Real): {rmse_real:.6f}")
    print(f"Evaluation RMSE (Imaginary): {rmse_imag:.6f}")
    # Сохранение обученной модели
    ctdrnn.save_model_pt()
    # Сохраним предсказанные значения
    ctdrnn.save_prediction(predictions)
    # Графики
    ctdrnn.plot_predictions(time_start=3.005e-4, time_end=3.006e-4)
