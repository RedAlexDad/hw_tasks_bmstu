import os
import sys
import torch
import pandas as pd

from NeuralODETrainer import NeuralODETrainer

if __name__ == '__main__':
    df = pd.read_csv('Amp_C_train.txt')

    # Проверка наличия CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

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
    epochs=5
    layers=[2**2, 2**4, 2**2]

    # Создание экземпляра класса с настройкой гиперпараметров
    NODE_model = NeuralODETrainer(df, batch_size, learning_rate, epochs, layers)

    # Обучение модели
    NODE_model.train()
    NODE_model.save_training_history()

    NODE_model.plot_training_history()

    # Оценка модели
    rmse_real, rmse_imag, predictions = NODE_model.evaluate()
    # Сохраним предсказанные значения
    NODE_model.save_prediction(predictions)

    print(f"Evaluation RMSE (Real): {rmse_real:.6f}")
    print(f"Evaluation RMSE (Imaginary): {rmse_imag:.6f}")

    # Сохранение модели
    NODE_model.save_model_pt()

    NODE_model.plot_predictions(time_start=0, time_end=0.1e-5)