import torch
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from MemoryPolynomialNNTrainer import MemoryPolynomialNNTrainer

class MemoryPolynomialNNEvaluator(MemoryPolynomialNNTrainer):
    def __init__(self, input_file, model_path, device=None):
        """
        Инициализация класса для оценки полиномиальных моделей.

        Args:
            input_file (str): Путь к файлу с входными данными.
            model_path (str): Путь к файлу сохраненной модели.
            device (str, optional): Устройство для вычислений ('cpu' или 'cuda').
        """
        # self.df = self.prepare_data(pd.read_csv(input_file))
        self.input_file = input_file
        self.model_path = model_path
        self.device = self.get_device(device)
        # Загрузка данных модели
        memory_depth, polynomial_degree, model_type, batch_size, learning_rate, epochs, hidden_layers, dropout_rate = self.load_model_with_info()
        
        super().__init__(
            df=pd.read_csv(input_file), 
            M=memory_depth,
            K=polynomial_degree,
            model_type=model_type, 
            batch_size=batch_size,
            learning_rate=learning_rate, 
            epochs=epochs, 
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        )
                
        # Загрузка данных и модели
        self.load_model()
        
    def get_device(self, select=None):
        """Определяет устройство для вычислений."""
        return torch.device('cuda' if torch.cuda.is_available() and (select is None or select == 'cuda') else 'cpu')

    def load_model(self):
        """Загружает модель"""
        self.model.load_state_dict(self.model_state_dict)
        self.model.eval()

    def load_model_with_info(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model_state_dict = checkpoint.get('model_state_dict', None)
    
        if self.model_state_dict is None:
            raise ValueError(f"The checkpoint file at {self.model_path} is missing required fields 'model_state_dict'")
        
        memory_depth = checkpoint.get('M', None)
        polynomial_degree = checkpoint.get('K', None)
        model_type = checkpoint.get('model_type', None)
        batch_size = checkpoint.get('batch_size', None)
        learning_rate = checkpoint.get('learning_rate', None)
        epochs = checkpoint.get('epochs', None)
        hidden_layers = checkpoint.get('hidden_layers', None)
        dropout_rate = checkpoint.get('dropout_rate', None)
                
        return memory_depth, polynomial_degree, model_type, batch_size, learning_rate, epochs, hidden_layers, dropout_rate

    def get_device(self, select=None):
        """Определяет устройство для вычислений."""
        return torch.device('cuda' if torch.cuda.is_available() and (select is None or select == 'cuda') else 'cpu')

    def evaluate(self):
        """Оценивает модель и выводит RMSE."""
        all_preds = []
        all_true = []

        with torch.no_grad():
            for X_batch, times_batch, y_batch in self.dataloader:
                X_batch, times_batch, y_batch = X_batch.to(self.device), times_batch.to(self.device), y_batch.to(self.device)
                
                # Объединение временных меток с входными признаками
                X_with_times = torch.cat((X_batch, times_batch.unsqueeze(1)), dim=1)

                pred = self.model(X_with_times)
                
                all_preds.append(pred.cpu().numpy())
                all_true.append(y_batch.cpu().numpy())

        # Конкатенация всех предсказаний и истинных значений
        self.pred = np.concatenate(all_preds)
        self.true = np.concatenate(all_true)
        
        # Извлечение реальной и мнимой частей
        self.pred_real = self.pred[:, 0]
        self.true_real = self.true[:, 0]
        self.pred_imag = self.pred[:, 1]
        self.true_imag = self.true[:, 1]
        
        # Вычисление RMSE для реальной и мнимой части
        rmse_real = np.sqrt(mean_squared_error(self.true_real, self.pred_real))
        rmse_imag = np.sqrt(mean_squared_error(self.true_imag, self.pred_imag))
                
        # Освобождение памяти GPU (при использовании CUDA)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return rmse_real, rmse_imag

def main():
    parser = argparse.ArgumentParser(description='Evaluate Polynomial Model')
    parser.add_argument('-f', '--input_file', type=str, default='Amp_C_train.txt', help='Path to the input data file')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the saved model file')
    parser.add_argument('-d', '--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Device to use for computation')
    
    args = parser.parse_args()
    
    evaluator = MemoryPolynomialNNEvaluator(
        input_file=args.input_file,
        model_path=args.model_path,
        device=args.device
    )
    
    rmse_real, rmse_imag = evaluator.evaluate()
    
    print(f"Evaluation RMSE (Real): {rmse_real:.6f}")
    print(f"Evaluation RMSE (Imag): {rmse_imag:.6f}")

if __name__ == "__main__":
    main()
