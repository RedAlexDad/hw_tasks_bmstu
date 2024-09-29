import time
import csv
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from IPython.display import display, HTML

# Разбиение на обучающую, валидационную и тестовую выборку
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, TimeSeriesSplit

# Масштабируемость модели
from sklearn.preprocessing import StandardScaler

# Метрики
from sklearn.metrics import mean_squared_error

# Нейросети
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

class CTDRNN:
    def __init__(self, input_size, hidden_sizes, output_size, num_layers, num_epochs=100, learning_rate=0.001, p=10, q=5, batch_size=64):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.p = p
        self.q = q
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CTDRNN_Model(input_size, hidden_sizes, output_size, num_layers).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        self.loss_values = []

    def create_time_delays_with_feedback(self, X_real, X_imag, Y_real, Y_imag):
        X_delayed = []
        Y_delayed = []
        
        for i in range(max(self.p, self.q), len(X_real)):
            # Входные задержки для X_real и X_imag
            X_real_window = X_real.iloc[i-self.p:i+1].values  # Входные задержки (x(k), x(k-1), ..., x(k-p))
            X_imag_window = X_imag.iloc[i-self.p:i+1].values
            
            # Обратная связь для Y_real и Y_imag
            Y_real_window = Y_real.iloc[i-self.q:i].values  # Обратная связь (y(k-1), y(k-2), ..., y(k-q))
            Y_imag_window = Y_imag.iloc[i-self.q:i].values
            
            # Комбинируем вещественные и мнимые части
            X_combined = np.concatenate([X_real_window, X_imag_window, Y_real_window, Y_imag_window])
            
            X_delayed.append(X_combined)
            
            # Текущие значения выходных данных
            Y_combined = np.array([Y_real.iloc[i], Y_imag.iloc[i]])
            Y_delayed.append(Y_combined)
        
        return np.array(X_delayed), np.array(Y_delayed)

    def prepare_data(self, df):
        X_seq, Y_seq = self.create_time_delays_with_feedback(
            df['input_real'],
            df['input_imag'],
            df['output_real'],
            df['output_imag']
        )   
        # Переформатируем X_seq, чтобы соответствовать размеру входных данных модели
        X_seq = X_seq.reshape(-1, self.input_size)

        X_scaled = self.scaler_X.fit_transform(X_seq)
        Y_scaled = self.scaler_Y.fit_transform(Y_seq)

        X_train_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
        Y_train_tensor = torch.tensor(Y_scaled, dtype=torch.float32).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return X_seq, Y_seq

    def train(self, print_batch=True, csv_file='training_history.csv'):
        print('Start train model')
        start_time = time.time()

        # Открываем файл для записи истории обучения
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Записываем заголовки столбцов
            writer.writerow(['Epoch', 'Batch ID', 'Batch', 'Progress (%)', 'Loss'])

            for epoch in range(self.num_epochs):
                epoch_loss = 0
                for batch_idx, (X_batch, Y_batch) in enumerate(self.train_loader):
                    # Перенос батчей на GPU
                    X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)  
                    
                    # Прямой проход
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, Y_batch)
                    
                    # Обратное распространение и оптимизация
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item() * X_batch.size(0)  # Умножаем на размер батча для подсчета средней потери

                    if print_batch:
                        # Вывод прогресса обучения
                        if batch_idx % 10 == 0:
                            print(f'Train Epoch: [{epoch}/{self.num_epochs}] [{batch_idx * len(X_batch)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                    else:       
                        print(f'Train Epoch: [{epoch}/{self.num_epochs}] \tLoss: {loss.item():.6f}')
                    
                    # Подсчет прогресса в процентах
                    progress = 100. * batch_idx / len(self.train_loader)

                    # Записываем каждую итерацию батча в CSV-файл
                    writer.writerow([epoch, batch_idx, f'[{batch_idx * len(X_batch)}/{len(self.train_loader.dataset)}]', f'{progress:.0f}%', loss.item()])
                
                # Сохраняем среднее значение функции потерь для текущей эпохи
                avg_loss = epoch_loss / len(self.train_loader.dataset)
                self.loss_values.append(avg_loss)

                # Записываем результат для завершенной эпохи в CSV
                writer.writerow([epoch, 'DONE', '100%', avg_loss])

                # Вывод потери для всей эпохи
                print(f'Train Epoch: {epoch} [DONE]\tLoss: {avg_loss:.6f}')

        end_time = time.time()
        self.elapsed = end_time - start_time


    def predict(self, X_seq):
        self.model.eval()
        with torch.no_grad():
            # Очистка памяти GPU
            torch.cuda.empty_cache()

            try:
                X_scaled = self.scaler_X.transform(X_seq)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
                Y_pred_scaled = self.model(X_tensor).cpu().numpy()
                Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
            except RuntimeError as e:
                # Если возникла ошибка нехватки памяти, переносим модель на CPU
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory! Moving model to CPU...")
                    self.model.to(torch.device('cpu'))
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(torch.device('cpu'))
                    Y_pred_scaled = self.model(X_tensor).cpu().numpy()
                    Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
                else:
                    raise e  # Если это не ошибка нехватки памяти, перенаправляем ее

        return Y_pred

    def evaluate(self, Y_seq, Y_pred):
        rmse_real = np.sqrt(mean_squared_error(Y_seq[:, 0], Y_pred[:, 0]))
        rmse_imag = np.sqrt(mean_squared_error(Y_seq[:, 1], Y_pred[:, 1]))

        return rmse_real, rmse_imag

    def plot_loss(self):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        epoch_start = 1
        axs[0].plot(range(epoch_start, self.num_epochs + 1), self.loss_values[epoch_start-1:], marker='o', linestyle='-', color='b')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Average Loss')
        axs[1].set_title(f'Loss Function (starting from Epoch {epoch_start})')
        axs[0].grid(True)

        epoch_start = self.num_epochs // 2
        axs[1].plot(range(epoch_start, self.num_epochs + 1), self.loss_values[epoch_start-1:], marker='o', linestyle='-', color='b')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Average Loss')
        axs[1].set_title(f'Loss Function (starting from Epoch {epoch_start})')
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_real(self, info_result, time_start, time_end):
        # Фильтрация данных по временной отметке
        filtered_data = info_result[(info_result.index >= time_start) & (info_result.index <= time_end)]
        time = filtered_data.index

        # Создание фигуры с двумя подграфиками
        fig, axs = plt.subplots(1, 2, figsize=(20, 5), sharex=True, sharey=True)
        
        # Общий заголовок для всей фигуры
        fig.suptitle('Comparison of Real and Predicted Values', fontsize=22)

        # Построение графиков реальных и предсказанных значений (вещественная часть)
        axs[0].plot(time, filtered_data['default_real'], alpha=0.7, color='red', lw=2.5, label='Real (default)')
        axs[0].plot(time, filtered_data['pred_real'], alpha=0.7, color='blue', lw=2.5, label='Predicted')
        axs[0].legend(fontsize='xx-large')
        axs[0].grid()
        axs[0].set_xlabel('Time', fontsize=20)
        axs[0].set_ylabel('Real Part', fontsize=20)
        axs[0].set_title('Real Part: Default vs Predicted', fontsize=20)

        # Построение графиков реальных и предсказанных значений (мнимая часть)
        axs[1].plot(time, filtered_data['default_imag'], alpha=0.7, color='red', lw=2.5, label='Real (default)')
        axs[1].plot(time, filtered_data['pred_imag'], alpha=0.7, color='blue', lw=2.5, label='Predicted')
        axs[1].legend(fontsize='xx-large')
        axs[1].grid()
        axs[1].set_xlabel('Time', fontsize=20)
        axs[1].set_ylabel('Imaginary Part', fontsize=20)
        axs[1].set_title('Imaginary Part: Default vs Predicted', fontsize=20)

        # Показываем график
        plt.tight_layout()
        plt.show()

    def save_model(self, filename_prefix, save_dir='models'):
        # Создаем папку, если ее нет
        os.makedirs(save_dir, exist_ok=True)

        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pth"

        # Полный путь к файлу
        filepath = os.path.join(save_dir, filename)

        # Сохраняем ВСЮ модель
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
            'p': self.p,
            'q': self.q,
            'scaler_X': self.scaler_X,
            'scaler_Y': self.scaler_Y,
            'epoch': self.num_epochs
        }, filepath)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_epochs = checkpoint['epoch']

    def save_results(self, Y_seq, Y_pred, rmse_real, rmse_imag):
        results = pd.Series({
            'NAME': 'CTDRNN',  # Название модели
            'RMSE VALID REAL': rmse_real,
            'RMSE VALID IMAG': rmse_imag,
            'TIME TRAINING [s]': self.elapsed,
            'PREDICTIONS': Y_pred,
            'PARAMETRS': self.model.state_dict()  # Хранение параметров модели
        })
   
        return results

    def print_params(self):
        html = """
        <table style="border-collapse: collapse; width: 50%;">
        <tr style="background-color: #f2f2f2;">
            <th style="text-align: left; padding: 8px;">Параметр</th>
            <th style="text-align: left; padding: 8px;">Значение</th>
        </tr>
        """
        html += f"<tr><td>Input size</td><td>{self.input_size}</td></tr>"
        html += f"<tr><td>Hidden sizes</td><td>{self.hidden_sizes}</td></tr>"
        html += f"<tr><td>Output size</td><td>{self.output_size}</td></tr>"
        html += f"<tr><td>Number of layers</td><td>{self.num_layers}</td></tr>"
        html += f"<tr><td>Number of epochs</td><td>{self.num_epochs}</td></tr>"
        html += f"<tr><td>Learning rate</td><td>{self.learning_rate}</td></tr>"
        html += f"<tr><td>Input delay (p)</td><td>{self.p}</td></tr>"
        html += f"<tr><td>Output delay (q)</td><td>{self.q}</td></tr>"
        html += f"<tr><td>Batch size</td><td>{self.batch_size}</td></tr>"
        html += "</table>"
        
        display(HTML(html))


class CTDRNN_Model(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_layers):
        super(CTDRNN_Model, self).__init__()
        
        # LSTM слой
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_sizes[0], num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3)

        # Динамическое создание полносвязных слоев
        self.fcs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        input_dim = hidden_sizes[0] * 2  # Учитываем bidirectional
        for i in range(1, len(hidden_sizes)):
            self.fcs.append(nn.Linear(input_dim, hidden_sizes[i]))
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[i]))  # Инициализируем LayerNorm с правильным размером
            input_dim = hidden_sizes[i]  # Для следующего слоя

        # Финальный выходной слой
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Проход через LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Берем последний временной шаг для каждого батча
        
        # Проход через динамически созданные полносвязные слои
        for fc, ln in zip(self.fcs, self.layer_norms):
            x = self.relu(fc(x))
            x = ln(x)  # Применяем заранее инициализированный LayerNorm
            x = self.dropout(x)
        
        # Финальный выход
        out = self.fc_out(x)
        return out