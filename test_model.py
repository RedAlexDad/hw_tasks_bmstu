import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from node_model import NeuralODETrainer

# Метрики
from sklearn.metrics import mean_squared_error

def F(X_in, model, train_loader):
    try:
        model.eval()
        true_values = []
        predicted_values = []
        times = []
        # Получаем устройство модели
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for t, inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, t)
                true_values.append(targets.cpu().numpy())
                predicted_values.append(outputs.cpu().numpy())
                times.append(t.cpu().numpy())

        # Конкатенация всех предсказанных значений, истинных значений и временных меток
        true_values = np.concatenate(true_values, axis=0)
        predicted_values = np.concatenate(predicted_values, axis=0)
        times = np.concatenate(times, axis=0)

        # Сортировка данных по временам
        sort_indices = np.argsort(times)
        times = times[sort_indices]
        true_values = true_values[sort_indices]
        predicted_values = predicted_values[sort_indices]

        # Вычисление RMSE
        rmse_real = mean_squared_error(true_values[:, 0], predicted_values[:, 0], squared=False)
        rmse_imag = mean_squared_error(true_values[:, 1], predicted_values[:, 1], squared=False)

        return predicted_values, rmse_real, rmse_imag
    except RuntimeError as e:
        raise e

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 main.py dataset.csv path_to_pretrained_model.pth")
        # sys.exit(1)
        input_file = "Amp_C_train.txt"
        model_path = "models/node_20241006_124330_kaggle.pth"
    else:        
        input_file = sys.argv[1]
        model_path = sys.argv[2]

    # Чтение тестовых данных
    df = pd.read_csv(input_file)
    
    batch_size=1024*5
    learning_rate=1e-5
    epochs=1000
    hidden_size=512

    # Создаем объект NODE с помощью загруженных параметров
    NODE_model = NeuralODETrainer(df, batch_size, learning_rate, epochs, hidden_size)

    # Создаем экземпляр модели
    NODE_model.model = NODE_model.NeuralODEModel(hidden_size=NODE_model.hidden_size).to(NODE_model.get_device())

    # Загружаем веса модели на CPU
    NODE_model.model.load_state_dict(torch.load(model_path, map_location='cpu'))
    NODE_model.model.eval()

    X_in = df['Input'].values.astype(np.complex64)
    # Истинные значения для вычисления метрики RMSE
    Y_true = df['Output'].values.astype(np.complex64)

    # Подготавливаем и делаем предсказания
    predictions, rmse_real, rmse_imag = F(X_in, NODE_model.model, NODE_model.train_loader) 

    # print(f"Evaluation RMSE (Real): {rmse_real:.6f}")
    # print(f"Evaluation RMSE (Imaginary): {rmse_imag:.6f}")
    
    # Создаем DataFrame из предсказаний
    df_predictions = pd.DataFrame(predictions, columns=['pred_real', 'pred_imag'])
    # Добавляем столбец с индексом
    df_predictions['index'] = np.arange(len(predictions))
    # Сохраняем в CSV
    df_predictions.to_csv('history/predictions_test.csv', index=False)

    # print("Predicted outputs:")
    # for pred in predictions:
    #     print(pred)