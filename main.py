import gc   
import torch
import argparse
import resource
import pandas as pd
import os

from MemoryPolynomialNNTrainer import MemoryPolynomialNNTrainer

def eval_hidden_layers(layer_string):
    """Оценить и вернуть список из строкового выражения слоев."""
    try:
        return [eval(item.strip()) for item in layer_string.split(',')]
    except SyntaxError:
        raise argparse.ArgumentTypeError("Invalid hidden layers expression.")

def argparse_setup():
    """Configuring and processing command line arguments."""
    parser = argparse.ArgumentParser(description='Train Memory Polynomial NN Model')

    parser.add_argument('-f', '--input_file', type=str, default='Amp_C_train.txt', help='Input data file')
    parser.add_argument('-m', '--memory_depth', type=int, default=5, help='Depth of memory')
    parser.add_argument('-k', '--polynomial_degree', type=int, default=3, help='Degree of the polynomial')
    parser.add_argument('-b', '--batch_size', type=int, default=2**10, help='Batch size for training')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs for training')
    # parser.add_argument('-x', '--hidden_layers', type=eval_hidden_layers, default='8,16,8', help='Comma separated list of hidden layer sizes, can include expressions (e.g., 2**8)')
    parser.add_argument('-x', '--hidden_layers', type=eval_hidden_layers, default='2**12,2**10,2**8,2**6,2**4,2**2', help='Comma separated list of hidden layer sizes, can include expressions (e.g., 2**8)')
    parser.add_argument('-d', '--dropout_rate', type=float, default=0.20, help='Dropout rate for the network')
    parser.add_argument('-t', '--model_type', type=str, choices=['memory_polynomial', 'sparse_delay_memory_polynomial', 'non_uniform_memory_polynomial', 'envelope_memory_polynomial'], default='memory_polynomial', help='Type of memory polynomial model')
    parser.add_argument('-s', '--early_stopping', type=int, default=20, help='Early stopping counter')
    parser.add_argument('-p', '--print_model_summary', type=int, default=False, help='Print modely summary and info')

    return parser.parse_args()


def main(args):
    # Печать используемого устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'The device used: {device}')

    # Проверка существования файла перед чтением
    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"File '{args.input_file}' not found.")
        exit(1)

    try:
        # Создание экземпляра класса с настройкой гиперпараметров
        model_nn = MemoryPolynomialNNTrainer(
            df=df, 
            M=args.memory_depth, 
            K=args.polynomial_degree, 
            model_type=args.model_type,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate, 
            epochs=args.epochs, 
            hidden_layers=args.hidden_layers,
            dropout_rate=args.dropout_rate
        )
        if args.print_model_summary: model_nn.print_model_summary()
        
        model_nn.train(max_early_stopping_counter=args.early_stopping)
        rmse_real, rmse_imag = model_nn.evaluate()

        print(f'Evaluation RMSE (Real): {rmse_real:.6f}')
        print(f'Evaluation RMSE (Imag): {rmse_imag:.6f}')
        model_nn.log_hparams_and_metrics(rmse_real, rmse_imag)
        model_nn.save_model_pt()
        
    except (RuntimeError, MemoryError) as e:
        if "can't allocate memory" in str(e) or "out of memory" in str(e):
            print("Mistake: Not enough memory.")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        # Получаем последние значения RMSE из истории, если возможно
        last_rmse_real = float('nan')
        last_rmse_imag = float('nan')
        if model_nn.history["rmse_real"] and model_nn.history["rmse_imag"]:
            last_rmse_real = model_nn.history["rmse_real"][-1]
            last_rmse_imag = model_nn.history["rmse_imag"][-1]

        # Логирование гиперпараметров с последними доступными значениями RMSE
        model_nn.log_hparams_and_metrics(last_rmse_real, last_rmse_imag)
    finally:
        try:
            model_nn.save_model_pt()
            print('Model saved successfully after memory error.')
        except Exception as ex:
            print(f'The model could not be saved due to: {ex}')
            
        # Освобождение памяти
        del df, model_nn
        gc.collect()
        exit(1)
    
if __name__ == '__main__':
    # Установка максимального лимита по памяти (в байтах)
    max_memory = 30 * 2**10 * 2**10 * 2**10  # 30 ГБ
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
    except Exception as e:
        print(f'The memory limit could not be set: {e}')

    args = argparse_setup()
    main(args)