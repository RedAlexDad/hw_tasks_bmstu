import torch
import pandas as pd

from CreateModel import CreateModel

class ModelEvalutor(CreateModel):
    def __init__(self, input_file, model_path, device=None):
        """
        Инициализация класса для оценки полиномиальных моделей.

        Args:
            input_file (str): Путь к файлу с входными данными.
            model_path (str): Путь к файлу сохраненной модели.
            device (str, optional): Устройство для вычислений ('cpu' или 'cuda').
        """
        # Создание DataFrame из файла
        df = pd.read_csv(input_file)
        
        # Загрузка параметров модели из сохранённого состояния
        time_delay, batch_size, learning_rate, epochs, hidden_layers, dropout_rate, model_state_dict = self.load_model_with_info(model_path, device)
            
        # Инициализация базового класса
        super().__init__(
            input_file=input_file,
            time_delay=time_delay,
            hidden_layers=hidden_layers,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            device=device
        )
        
        # Загрузка состояния модели
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
    def load_model_with_info(self, model_path, device):
        """Загружает параметры модели и конфигурацию из чекпоинта."""
        checkpoint = torch.load(model_path, map_location=device)
        model_state_dict = checkpoint.get('model_state_dict', None)
    
        if model_state_dict is None:
            raise ValueError(f"The checkpoint file at {model_path} is missing required fields 'model_state_dict'")
        
        time_delay = checkpoint.get('time_delay', None)
        batch_size = checkpoint.get('batch_size', None)
        learning_rate = checkpoint.get('learning_rate', None)
        epochs = checkpoint.get('epochs', None)
        hidden_layers = checkpoint.get('hidden_layers', None)
        dropout_rate = checkpoint.get('dropout_rate', None)
                
        return time_delay, batch_size, learning_rate, epochs, hidden_layers, dropout_rate, model_state_dict
    
    def evaluate(self):
        """Оценивает модель и выводит RMSE с использованием PyTorch."""
        self.model.eval()
        
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
                
        avg_rmse_real = total_rmse_real / total_samples
        avg_rmse_imag = total_rmse_imag / total_samples

        # Освобождение памяти GPU (при использовании CUDA)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return avg_rmse_real, avg_rmse_imag
