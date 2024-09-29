import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from CTDRNN import CTDRNN, CTDRNN_Model

def load_model(model_path):
    # Загружаем данные модели
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    # Извлекаем параметры
    input_size = checkpoint['input_size']
    hidden_sizes = checkpoint['hidden_sizes']
    output_size = checkpoint['output_size']
    num_layers = checkpoint['num_layers']
    p = checkpoint['p']
    q = checkpoint['q']

    # Создаем объект CTDRNN с помощью загруженных параметров
    ctdrnn = CTDRNN(
        input_size=input_size,  # Используем загруженный input_size
        hidden_sizes=hidden_sizes,  # Используем загруженный hidden_sizes
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

def F(X_in, loading_model):
    with torch.no_grad():
        # Очистка памяти GPU
        torch.cuda.empty_cache()

        # Разделение на вещественную и мнимую части
        X_in_real = X_in.real.reshape(-1, 1)  # Преобразуем в 2D
        X_in_imag = X_in.imag.reshape(-1, 1)  # Преобразуем в 2D

        # Предполагается, что для предсказания требуется несколько задержанных значений
        p = loading_model.p  # Задержка на входе
        q = loading_model.q  # Задержка на выходе

        # Создаем задержки для входных данных
        X_delayed = []

        for i in range(max(p, q), len(X_in_real)):
            # Задержки вещественной и мнимой частей (p+1 значений)
            X_real_window = X_in_real[i-p:i+1].flatten()  # Задержки вещественной части
            X_imag_window = X_in_imag[i-p:i+1].flatten()  # Задержки мнимой части

            # Задержка выхода (только q значений)
            Y_real_window = X_in_real[i-q:i].flatten()  # Задержки выхода вещественной части (q значений)
            Y_imag_window = X_in_imag[i-q:i].flatten()  # Задержки выхода мнимой части (q значений)

            # Комбинируем задержанные вещественные и мнимые части
            X_combined = np.concatenate([X_real_window, X_imag_window, Y_real_window, Y_imag_window])
            X_delayed.append(X_combined)

            # Проверяем размерность
            assert len(X_combined) == loading_model.input_size, f"Expected {loading_model.input_size} features, but got {len(X_combined)}"

        X_delayed = np.array(X_delayed)

        # print("Shape of X_in:", X_in.shape)
        # print("Shape of X_delayed:", X_delayed.shape)

        # Проверяем, соответствует ли количество признаков ожидаемому
        expected_features = loading_model.input_size
        if X_delayed.shape[1] != expected_features:
            raise ValueError(f"Количество признаков не совпадает: ожидалось {expected_features}, а получили {X_delayed.shape[1]}")

        # Масштабирование данных
        X_scaled = loading_model.scaler_X.transform(X_delayed)
        
        try:
            # Преобразуем в тензор
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to('cuda')

            # Прогон через модель
            Y_pred_scaled = loading_model.model(X_tensor).cpu().numpy()

            # Обратное преобразование предсказанных данных
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
        # sys.exit(1)
        input_file = "Amp_C_train.txt"
        model_path = "models/ctdrnn_20240929_100631.pth"
    else:        
        input_file = sys.argv[1]
        model_path = sys.argv[2]

    # Чтение тестовых данных
    df = pd.read_csv(input_file)
    X_in = df['Input'].values.astype(np.complex64)

    # Загружаем модель
    model = load_model(model_path)

    # Подготавливаем и делаем предсказания
    predictions = F(X_in, model)

    # # Создаем DataFrame из предсказаний
    # df_predictions = pd.DataFrame(predictions, columns=['pred_real', 'pred_imag'])

    # # Добавляем столбец с индексом (если необходимо)
    # df_predictions['index'] = np.arange(len(predictions))

    # # Сохраняем в CSV
    # df_predictions.to_csv('predictions_test.csv', index=False)

    # print("Predicted outputs:")
    # for pred in predictions:
        # print(pred)
