import os
import torch
import signal
import numpy as np
from tqdm.auto import tqdm

from CreateModel import CreateModel
from LogsTensorboard import LogsTensorboard

# Класс для управления моделью
class ModelTrainer(CreateModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Инициализация логгера
        self.logs_tensorboard = LogsTensorboard(self.model, self.dataset, self.epochs, self.device)
        self.logs_tensorboard.log_model_graph()
        
        # Установка обработчика сигналов
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("Program interruption! Saving hyperparameters...")
            
        # Освобождение памяти GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print("CUDA memory cleared.")
            
        rmse_real, rmse_imag = self.evaluate()
        print(f"Evaluation RMSE (Real): {rmse_real:.6f}")
        print(f"Evaluation RMSE (Imag): {rmse_imag:.6f}")
        self.logs_tensorboard.log_hparams_and_metrics(rmse_real, rmse_imag)
        print("Hyperparameters are saved. Completion of the program.")
        
        exit(0)
        
    def train(self, max_early_stopping_counter=10):
        self.model.train()
        early_stopping_counter = 0
        best_rmse = float('inf')

        for epoch in range(self.epochs):
            running_rmse = 0.0
            running_rmse_real = 0.0
            running_rmse_imag = 0.0
            total_samples = 0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")
            
            for batch_idx, ((I_in, Q_in), (I_out, Q_out)) in enumerate(progress_bar):
                I_in, Q_in, I_out, Q_out = I_in.to(self.device), Q_in.to(self.device), I_out.to(self.device), Q_out.to(self.device)

                self.optimizer.zero_grad()
                
                outputs = self.model(I_in, Q_in)
                target = torch.cat((I_out.view(-1, 1), Q_out.view(-1, 1)), dim=1)
                loss = self.criterion(outputs, target)
                
                # Разделим предсказания и цели на реальные и мнимые части
                pred_real, pred_imag = outputs[:, 0], outputs[:, 1]
                true_real, true_imag = target[:, 0], target[:, 1]
                
                # Вычисление RMSE для реальной и мнимой частей отдельно
                rmse_real = torch.sqrt(torch.mean((pred_real - true_real) ** 2))
                rmse_imag = torch.sqrt(torch.mean((pred_imag - true_imag) ** 2))
                
                # Используем RMSE как основную метрику
                rmse = torch.sqrt(loss)
                
                rmse.backward()
                self.optimizer.step()

                batch_size = len(I_in)
                running_rmse += rmse.item() * batch_size
                running_rmse_real += rmse_real.item() * batch_size
                running_rmse_imag += rmse_imag.item() * batch_size
                total_samples += batch_size

                # Логирование в TensorBoard
                self.logs_tensorboard.writer.add_scalar('Training/RMSE', rmse.item(), epoch * len(self.dataloader) + batch_idx)
                self.logs_tensorboard.writer.add_scalar('Training/RMSE_Real', rmse_real.item(), epoch * len(self.dataloader) + batch_idx)
                self.logs_tensorboard.writer.add_scalar('Training/RMSE_Imag', rmse_imag.item(), epoch * len(self.dataloader) + batch_idx)

                # Получение текущего значения learning rate
                current_learning_rate = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(rmse=f"{rmse:.10f}", lr=f"{current_learning_rate:.6f}")

            avg_rmse = running_rmse / total_samples
            avg_rmse_real = running_rmse_real / total_samples
            avg_rmse_imag = running_rmse_imag / total_samples
            
            self.history["epoch"].append(epoch + 1)
            self.history["total_rmse"].append(avg_rmse)
            self.history["rmse_real"].append(avg_rmse_real)
            self.history["rmse_imag"].append(avg_rmse_imag)

            # Логирование средней RMSE в TensorBoard
            self.logs_tensorboard.writer.add_scalar('Training/Average_RMSE', avg_rmse, epoch)
            self.logs_tensorboard.writer.add_scalar('Training/Average_RMSE_Real', avg_rmse_real, epoch)
            self.logs_tensorboard.writer.add_scalar('Training/Average_RMSE_Imag', avg_rmse_imag, epoch)
            self.logs_tensorboard.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Логирование распределения весов для каждого слоя
            for name, param in self.model.named_parameters():
                self.logs_tensorboard.writer.add_histogram(f'Weights/{name}', param, epoch)
                if param.grad is not None:
                    self.logs_tensorboard.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

            # Обновление learning rate scheduler
            self.scheduler.step(avg_rmse)

            # Early Stopping
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= max_early_stopping_counter:  # Параметр patience для Early Stopping
                print("Early stopping activated.")
                break
            
            print(f"Epoch {epoch+1}/{self.epochs};\t AVG RMSE: {avg_rmse:.10f};\t Real RMSE: {avg_rmse_real:.5f};\t Imag RMSE: {avg_rmse_imag:.5f}")
                
            # Освобождение памяти GPU в конце каждой эпохи
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_true = []
        
        total_rmse_real = 0.0
        total_rmse_imag = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for (I_in, Q_in), (I_out, Q_out) in self.dataloader:
                I_in, Q_in, I_out, Q_out = I_in.to(self.device), Q_in.to(self.device), I_out.to(self.device), Q_out.to(self.device)
                outputs = self.model(I_in, Q_in)
                target = torch.cat((I_out.view(-1, 1), Q_out.view(-1, 1)), dim=1)

                rmse = torch.sqrt(torch.mean((outputs - target) ** 2, dim=0))
                
                total_rmse_real += rmse[0].item() * len(I_in)
                total_rmse_imag += rmse[1].item() * len(I_in)
                total_samples += len(I_in)
                
                all_preds.append(outputs.cpu().numpy())
                all_true.append(target.cpu().numpy())

        avg_rmse_real = total_rmse_real / total_samples
        avg_rmse_imag = total_rmse_imag / total_samples

        # Объединяем все батчи в один массив
        all_preds = np.concatenate(all_preds, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        
        # Логируем RMSE в TensorBoard
        self.logs_tensorboard.writer.add_scalar('Evaluation/REAL', avg_rmse_real, len(self.history["epoch"]) - 1)
        self.logs_tensorboard.writer.add_scalar('Evaluation/IMAG', avg_rmse_imag, len(self.history["epoch"]) - 1)
            
        # Освобождение памяти GPU (при использовании CUDA)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return avg_rmse_real, avg_rmse_imag

    def save_model_pt(self, filename='model.pt'):
        path = os.path.join('models', filename)
        os.makedirs('models', exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')
            
    def save_model_pt(self, save_dir='models'):
        """
        Сохраняет всю модель PyTorch в формате .pt, используя имя эксперимента и индекс.

        Args:
            save_dir (str, optional): Директория для сохранения модели. По умолчанию 'models'.
        """
        # Извлекаем конечную часть пути, чтобы получить имя эксперимента с индексом
        experiment_name_with_index = os.path.basename(self.logs_tensorboard.log_dir)

        # Генерируем имя файла с текущей датой и временем
        filename = f"{experiment_name_with_index}.pt"

        # Полный путь к файлу, без создания дополнительных подпапок
        filepath = os.path.join(save_dir, filename)

        # Убедитесь, что основная директория для сохранения моделей существует
        os.makedirs(save_dir, exist_ok=True)

        # Создаем словарь, включающий модель и тип модели
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'time_delay': self.dataset.time_delay,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate
        }

        # Сохраняем модель и метаданные
        torch.save(checkpoint, filepath)
        print(f"Model saved in {filepath}")

    def plot_training_history(self, window_size=5):
        """
        Строит графики истории обучения модели, отображая RMSE на каждой эпохе и скользящее среднее.
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
        # Преобразуем список эпох для оси X
        epochs = self.history["epoch"]
    
        # Вычисляем скользящее среднее
        rmse = np.array(self.history["total_rmse"])
        moving_avg = np.convolve(rmse, np.ones(window_size)/window_size, mode='valid')
    
        # Первый график: Полная история
        axs[0].plot(epochs, rmse, marker='o', linestyle='-', color='b', markersize=5, label='RMSE')
        axs[0].plot(epochs[window_size-1:], moving_avg, color='r', label=f'Moving Average (window size={window_size})')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Average Loss')
        axs[0].set_title('Loss Function (Full History)')
        axs[0].grid(True)
        axs[0].legend()
    
        # Второй график: Половина истории
        mid_index = len(epochs) // 2
        axs[1].plot(epochs[mid_index:], rmse[mid_index:], marker='o', linestyle='-', color='b', markersize=5, label='RMSE')
        axs[1].plot(epochs[mid_index + window_size - 1:], moving_avg[mid_index:], color='r', label=f'Moving Average (window size={window_size})')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Average RMSE')
        axs[1].set_title('Loss Function (Second Half of Training)')
        axs[1].grid(True)
        axs[1].legend()
    
        plt.tight_layout()
        plt.show()

    def plot_predictions(self, time_start=0, time_end=1.01e-4):
        """
        Построение графиков предсказанных и фактических значений в заданном временном диапазоне.

        Args:
            time_start (float, optional): Начальное время для отображения. По умолчанию 0.
            time_end (float, optional): Конечное время для отображения. По умолчанию 1.01e-4.
        """
        self.model.eval()
        pred_real, pred_imag = self.pred[:, 0], self.pred[:, 1]
        true_real, true_imag = self.true[:, 0], self.true[:, 1]

        # Фильтрация данных по указанному временному диапазону
        time_mask = (self.times >= time_start) & (self.times <= time_end)

        selected_times = self.times[time_mask]
        
        # Фильтрация предсказаний и фактических значений по временному диапазону
        pred_real = pred_real[time_mask]
        pred_imag = pred_imag[time_mask]
        true_real = true_real[time_mask]
        true_imag = true_imag[time_mask]
    
        # Проверка, чтобы убедиться, что данные не пустые
        if len(selected_times) == 0:
            print(f"No data points found between {time_start} and {time_end}.")
            return
        
        # Построение графиков
        fig, axs = plt.subplots(2, 1, figsize=(15, 8))

        # Реальная часть
        axs[0].plot(selected_times, true_real, label='True Real', linestyle='-', color='red')
        axs[0].plot(selected_times, pred_real, label='Predicted Real', linestyle='-', color='blue')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Real Part')
        axs[0].legend()
        axs[0].grid(True)

        # Мнимая часть
        axs[1].plot(selected_times, true_imag, label='True Imag', linestyle='-', color='red')
        axs[1].plot(selected_times, pred_imag, label='Predicted Imag', linestyle='-', color='blue')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Imaginary Part')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    def draw_plot_signal(self, signal_type, time_start=0, time_end=1e-6):
        """
        Построение графика сигнала в указанном временном диапазоне.
        
        Args:
            signal_type (str): Тип сигнала ('input' или 'output').
            time_start (float): Начальное время.
            time_end (float): Конечное время.
        """
        # Фильтрация данных по временной отметке
        filtered_data = self.df[(self.df.index >= time_start) & (self.df.index <= time_end)]
        time = filtered_data.index

        # Построение графика реальной и мнимой частей сигнала
        plt.figure(figsize=(10, 6))
        plt.plot(time, filtered_data[f'{signal_type}_real'], label=f'{signal_type} Real Part', color='blue', linestyle='-')
        plt.plot(time, filtered_data[f'{signal_type}_imag'], label=f'{signal_type} Imaginary Part', color='red', linestyle='-')
        
        plt.title(f'{signal_type.capitalize()} Signal from {time_start} to {time_end} seconds')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    def print_model_summary(self, filename_prefix="model_parameters", save_dir='history'):
        """
        Выводит информацию о модели и сохраняет её параметры и их размерности в CSV файл.

        Args:
            filename_prefix (str, optional): Префикс имени файла. По умолчанию 'model_parameters'.
            save_dir (str, optional): Директория для сохранения файла. По умолчанию 'history'.
        """
        # Создаем папку, если ее нет
        os.makedirs(save_dir, exist_ok=True)

        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}_{self.model_id}.csv"

        # Полный путь к файлу
        filepath = os.path.join(save_dir, filename)
        
        df_params = pd.DataFrame(columns=['Parameter name', 'Parameter shape', 'Parameter count'])
        
        print(f"Model architecture: {self.model}")
        print("-" * 50)

        total_params = 0
        for name, param in self.model.named_parameters():
            print(f"Parameter name: {name}")
            print(f"Parameter shape: {param.shape}")
            param_count = torch.numel(param)
            print(f"Parameter count: {param_count}")
            print("-" * 30)

            # Добавляем информацию о параметре в DataFrame
            df_params.loc[len(df_params)] = [name, param.shape, param_count] 
            
            total_params += param_count

        print(f"Total trainable parameters: {total_params}")
        print("=" * 50)
        
        # Сохраняем DataFrame в CSV файл
        # df_params.to_csv(filepath, index=False)
        
        # print(f"Print model saved in {filepath}")
        