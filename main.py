import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Метрики
from sklearn.metrics import mean_squared_error

from node_model import NeuralODETrainer

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

    batch_size=1024*5
    learning_rate=1e-4
    epochs=2
    layers=[64, 128, 256, 128]

    # Создание экземпляра класса с настройкой гиперпараметров
    NODE_model = NeuralODETrainer(df, batch_size, learning_rate, epochs, layers)

    # Обучение модели
    NODE_model.train()

    # NODE_model.plot_training_history()

    # Оценка модели
    NODE_model.evaluate()

    NODE_model.plot_predictions(time_start=0, time_end=0.1e-5)