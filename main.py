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

from CTDRNN import CTDRNN, CTDRNN_Model

if __name__ == '__main__':
    df = pd.read_csv('Amp_C_train.txt')

    # Приведение к нижнему регистру названия колонки
    df.columns = df.columns.str.lower()

    # Установка колонки 'time' в качестве индекса
    df.set_index('time', inplace=True)

    df['input'] = df['input'].apply(lambda x: complex(x.strip('()').replace('i', 'j')))
    df['output'] = df['output'].apply(lambda x: complex(x.strip('()').replace('i', 'j')))

    df['input_real'] = df['input'].apply(np.real)
    df['input_imag'] = df['input'].apply(np.imag)
    df['output_real'] = df['output'].apply(np.real)
    df['output_imag'] = df['output'].apply(np.imag)

    df = df.drop(['input', 'output'], axis=1)

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
    p=30 # Входная задержка
    q=15  # Выходная задержка

    # Создадим объект CTRNN
    ctdrnn = CTDRNN(
        input_size=2 * (p + 1 + q), 
        hidden_sizes=[128, 256, 256, 128, 64],
        output_size=2, 
        num_layers=2,
        num_epochs=10,
        learning_rate=0.001,
        p=p,
        q=q,
        batch_size=1024*1
    )

    # Вывод параметров в виде HTML-таблицы
    ctdrnn.print_params()

    # Подготовка данных
    X_seq, Y_seq = ctdrnn.prepare_data(df)

    # Обучим модель
    ctdrnn.train()

    # График обучения
    ctdrnn.plot_loss()

    # Сделаем предсказания
    # X_seq - входные данные для предсказания
    Y_pred = ctdrnn.predict(X_seq)

    # Создаем DataFrame из предсказаний
    df_predictions = pd.DataFrame(Y_pred, columns=['pred_real', 'pred_imag'])

    # Добавляем столбец с индексом (если необходимо)
    df_predictions['index'] = np.arange(len(Y_pred))

    # Сохраняем в CSV
    df_predictions.to_csv('predictions_train.csv', index=False)

    # Оценка модель
    # Y_seq - реальные выходные данные
    rmse_real, rmse_imag = ctdrnn.evaluate(Y_seq, Y_pred)

    print(f'RMSE (Real part): {rmse_real}')
    print(f'RMSE (Imaginary part): {rmse_imag}')
    # Здесь будем сохранить результаты обучения
    results = pd.DataFrame()
    # А это будет счетчтиком для нумерация моеделй
    count_model = 0
    # Сохранение результатов
    results[count_model] = ctdrnn.save_results(Y_seq, Y_pred, rmse_real, rmse_imag)

    # display(results)
    count_model += 1

    # Сохраним модель
    ctdrnn.save_model('ctdrnn') 

    # Построим графики
    info_result = pd.DataFrame({
        'default_real': Y_seq[:, 0],  # Реальные значения (вещественная часть)
        'default_imag': Y_seq[:, 1],  # Реальные значения (мнимая часть)
        'pred_real': Y_pred[:, 0],    # Предсказанные значения (вещественная часть)
        'pred_imag': Y_pred[:, 1]     # Предсказанные значения (мнимая часть)
    })

    info_result.index = range(len(Y_seq))  # Пример временной шкалы: индексы от 0 до количества наблюдений
    ctdrnn.plot_predictions_vs_real(info_result, time_start=0, time_end=1.005e2)

