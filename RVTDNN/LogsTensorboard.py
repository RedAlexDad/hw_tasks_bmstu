import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

class LogsTensorboard():
    def __init__(self, model, dataset, epochs, device):
        self.model = model
        self.model_type = model.__class__.__name__
        self.dataset = dataset
        self.epochs = epochs
        self.device = device
        
        self.writer = self.initialize_log_dir()
        
    def initialize_log_dir(self):
        self.log_dir = f'logs/{self.model_type}'
        i = 0
        while os.path.exists(self.log_dir):
            self.log_dir = f'logs/{self.model_type}_{i}'
            i += 1
        os.makedirs(self.log_dir)
        print(f'The experiment with the name has been saved: {self.log_dir}')
        return SummaryWriter(log_dir=self.log_dir)
    
    def log_model_graph(self):
        time_delay = self.model.time_delay
        # Размер инпута должен быть в зависимости от time_delay и быть двойным
        input_size = 2 * (time_delay + 1)
        
        # Создаем фиктивные входы с типом float
        dummy_I_in = torch.randn(1, time_delay + 1, dtype=torch.float32).to(self.device)
        dummy_Q_in = torch.randn(1, time_delay + 1, dtype=torch.float32).to(self.device)
        
        # Логирование модели. Передаем кортеж из двух тензоров
        self.writer.add_graph(self.model, (dummy_I_in, dummy_Q_in))
        
    def log_predictions_to_tensorboard(self):
        """
        Логирование всех предсказанных и фактических значений в TensorBoard.
        """
        self.model.eval()
        pred_real, pred_imag = self.pred[:, 0], self.pred[:, 1]
        true_real, true_imag = self.true[:, 0], self.true[:, 1]

        # Логирование в TensorBoard для реальных значений
        for i in range(len(self.pred)):
            self.writer.add_scalars(
                'Predictions/Real', 
                {
                    'Predicted': pred_real[i],
                    'True': true_real[i],
                },
                global_step=i
            )
    
        # Логирование в TensorBoard для мнимых значений
        for i in range(len(self.pred)):
            self.writer.add_scalars(
                'Predictions/Imag', 
                {
                    'Predicted ': pred_imag[i],
                    'True': true_imag[i]
                },
                global_step=i
            )

        print(f"All predictions logged to TensorBoard.")

    def log_hparams_and_metrics(self, learning_rate, dropout_rate, batch_size, time_delay, epochs, rmse_real=None, rmse_imag=None):
        hparams = {
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'time_delay': time_delay,
            'epochs': epochs
        }
            
        # Получение текущего времени в читаемом формате
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}"  # Читаемое имя
    
        # Логирование гиперпараметров и метрик
        self.writer.add_hparams(hparams, { 
            'rmse_real': rmse_real,
            'rmse_imag': rmse_imag 
        }, run_name=run_name)
        
        self.writer.close()
        print(f"Hyperparameters have been successfully saved with the name: {os.path.basename(self.log_dir)}/{run_name}")
          