import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from CTDRNN import CTDRNN, CTDRNN_Model

def load_model(model_path):
    p = 10
    q = 5
    input_size = 2 * (p + 1 + q)

    # ctdrnn = CTDRNN(
    #     input_size=input_size, 
    #     hidden_sizes=[128, 256, 256, 128, 64],
    #     output_size=2, 
    #     num_layers=2,
    #     num_epochs=3,
    #     learning_rate=0.001,
    #     p=p,
    #     q=q,
    #     batch_size=2048
    # )

    # Загрузка параметры модели
    # ctdrnn.model.load_state_dict(torch.load(model_path))

    # Загрузка модели
    # ctdrnn = torch.load(model_path, weights_only=False)
    ctdrnn = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False) 

    # Инициализация модели
    # ctdrnn.model = CTDRNN(
    #     input_size=input_size, 
    #     hidden_sizes=[128, 256, 256, 128, 64],
    #     output_size=2, 
    #     num_layers=2,
    #     num_epochs=3,
    #     learning_rate=0.001,
    #     p=p,
    #     q=q,
    #     batch_size=2048
    # )
    ctdrnn.model = CTDRNN_Model(input_size, [128, 256, 256, 128, 64], 2, 2).to('cuda')

    # Установка модели в режим оценки
    ctdrnn.model.eval()

    return ctdrnn

def load_model1(model_path):
    # Загружаем данные модели
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    # Извлекаем параметры
    input_size = checkpoint['input_size']
    hidden_sizes = checkpoint['hidden_sizes']
    output_size = checkpoint['output_size']
    num_layers = checkpoint['num_layers']
    p = checkpoint['p']
    q = checkpoint['q']

    # Создаем объект CTDRNN
    ctdrnn = CTDRNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        num_layers=num_layers,
        num_epochs=3,
        learning_rate=0.001,
        p=p,
        q=q,
        batch_size=2048
    )

    # Загружаем параметры модели
    ctdrnn.model.load_state_dict(checkpoint['model_state_dict'])

    # Загружаем scaler_X и scaler_Y
    ctdrnn.scaler_X = checkpoint['scaler_X']
    ctdrnn.scaler_Y = checkpoint['scaler_Y']

    ctdrnn.model.eval()

    return ctdrnn

def F(X_in, load_model):   
    with torch.no_grad():
        X_in_tensor = torch.from_numpy(X_in).view(-1, load_model.input_size).to('cuda')
        X_in_tensor = torch.stack([X_in_tensor.real, X_in_tensor.imag], dim=1) 

        predicted_output = load_model.model(X_in_tensor).cpu().numpy()

    return predicted_output

def F1(X_in, loading_model):
    with torch.no_grad():
        # Очистка памяти GPU
        torch.cuda.empty_cache()

        try:
            # Разделение на вещественную и мнимую части
            X_in_real = X_in.real.reshape(-1, 1)  # Преобразуем в 2D
            X_in_imag = X_in.imag.reshape(-1, 1)  # Преобразуем в 2D

            # Масштабирование
            X_scaled_real = loading_model.scaler_X.transform(X_in_real)
            X_scaled_imag = loading_model.scaler_X.transform(X_in_imag)
            X_scaled = np.stack((X_scaled_real, X_scaled_imag), axis=1)  # Объединяем 

            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to('cuda')
            Y_pred_scaled = loading_model.model(X_tensor).cpu().numpy()
            Y_pred = loading_model.scaler_Y.inverse_transform(Y_pred_scaled)
        except RuntimeError as e:
            # Если возникла ошибка нехватки памяти, переносим модель на CPU
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory! Moving model to CPU...")
                loading_model.model.to(torch.device('cpu'))
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(torch.device('cpu'))
                Y_pred_scaled = loading_model.model(X_tensor).cpu().numpy()
                Y_pred = loading_model.scaler_Y.inverse_transform(Y_pred_scaled)
            else:
                raise e  # Если это не ошибка нехватки памяти, перенаправляем ее

    return Y_pred

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 main.py dataset.csv path_to_pretrained_model.pth")
        sys.exit(1)

    input_file = sys.argv[1]
    model_path = sys.argv[2]

    df = pd.read_csv(input_file)
    X_in = df['Input'].values.astype(np.complex64)

    # Загружаем модель
    model = load_model1(model_path)

    # Делаем предсказания
    predictions = F1(X_in, model)

    # Создаем DataFrame из предсказаний
    df_predictions = pd.DataFrame(predictions, columns=['pred_real', 'pred_imag'])

    # Добавляем столбец с индексом (если необходимо)
    df_predictions['index'] = np.arange(len(predictions))

    # Сохраняем в CSV
    df_predictions.to_csv('predictions.csv', index=False)

    print("Predicted outputs:")
    for pred in predictions:
        print(pred)