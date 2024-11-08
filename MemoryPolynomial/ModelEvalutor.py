import torch
import pandas as pd

from MemoryPolynomial.CreateModel import CreateModel

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
        memory_depth, polynomial_degree, model_type, batch_size, learning_rate, epochs, hidden_layers, dropout_rate, model_state_dict = self.load_model_with_info(model_path, device)
        
        # Инициализация базового класса
        super().__init__(
            df=df,
            M=memory_depth,
            K=polynomial_degree,
            model_type=model_type,
            batch_size=batch_size,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            epochs=epochs,
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
        
        memory_depth = checkpoint.get('M', None)
        polynomial_degree = checkpoint.get('K', None)
        model_type = checkpoint.get('model_type', None)
        batch_size = checkpoint.get('batch_size', None)
        learning_rate = checkpoint.get('learning_rate', None)
        epochs = checkpoint.get('epochs', None)
        hidden_layers = checkpoint.get('hidden_layers', None)
        dropout_rate = checkpoint.get('dropout_rate', None)
                
        return memory_depth, polynomial_degree, model_type, batch_size, learning_rate, epochs, hidden_layers, dropout_rate, model_state_dict
    
    def evaluate(self):
        """Оценивает модель и выводит RMSE с использованием PyTorch."""
        self.model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for X_batch, y_batch, times_batch in self.dataset.dataloader:
                X_batch, y_batch, times_batch = X_batch.to(self.device), y_batch.to(self.device), times_batch.to(self.device)
                
                # Объединение временных меток с входными признаками
                if times_batch.dim() == 1:
                    times_batch = times_batch.unsqueeze(1)
                X_with_times = torch.cat((X_batch, times_batch), dim=1)

                pred = self.model(X_with_times)
                
                all_preds.append(pred)
                all_true.append(y_batch)

        # Конкатенация всех предсказаний и истинных значений
        all_preds = torch.cat(all_preds, dim=0)
        all_true = torch.cat(all_true, dim=0)
        
        # Извлечение реальной и мнимой частей
        pred_real = all_preds[:, 0]
        true_real = all_true[:, 0]
        pred_imag = all_preds[:, 1]
        true_imag = all_true[:, 1]
        
        # Вычисление RMSE для реальной и мнимой части с использованием PyTorch
        mse_real = torch.mean((pred_real - true_real) ** 2)
        rmse_real = torch.sqrt(mse_real)
        
        mse_imag = torch.mean((pred_imag - true_imag) ** 2)
        rmse_imag = torch.sqrt(mse_imag)
                
        # Освобождение памяти GPU (при использовании CUDA)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # Возвращаем RMSE в формате числа (с возможным переводом на CPU для вывода)
        return rmse_real.item(), rmse_imag.item()