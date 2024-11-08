import gc   
import argparse
import resource
import pandas as pd

from MemoryPolynomial.ModelTrainer import ModelTrainer

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
    parser.add_argument('-x', '--hidden_layers', type=eval_hidden_layers, default='32,16,8', help='Comma separated list of hidden layer sizes, can include expressions (e.g., 2**8)')
    parser.add_argument('-d', '--dropout_rate', type=float, default=0.20, help='Dropout rate for the network')
    parser.add_argument('-t', '--model_type', type=str, choices=['memory_polynomial', 'sparse_delay_memory_polynomial', 'non_uniform_memory_polynomial', 'envelope_memory_polynomial'], default='memory_polynomial', help='Type of memory polynomial model')
    parser.add_argument('-s', '--early_stopping', type=int, default=100, help='Early stopping counter')
    parser.add_argument('-p', '--print_model_summary', type=int, default=False, help='Print modely summary and info')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='Device to use for computation')

    return parser.parse_args()
                
# Основной код для инициализации
def main(args):
    # Проверка существования файла перед чтением
    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"File '{args.input_file}' not found.")
        exit(1)
        
    try:
        # Создание экземпляра класса с настройкой гиперпараметров
        model = ModelTrainer(
            df=df,
            M=args.memory_depth,
            K=args.polynomial_degree,
            model_type=args.model_type,
            batch_size=args.batch_size,
            hidden_layers=args.hidden_layers,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.early_stopping,
            factor=0.9,
            dropout_rate=args.dropout_rate,
            device=args.device
        )
             
        if args.print_model_summary: model.print_model_summary() # Печать информации о модели, если указано
        model.train(max_early_stopping_counter=args.early_stopping)
        
        # Оценка модели
        rmse_real, rmse_imag = model.evaluate()
        print(f'Final RMSE (Real): {rmse_real:.6f}')
        print(f'Final RMSE (Imag): {rmse_imag:.6f}')

        model.logs_tensorboard.log_hparams_and_metrics(rmse_real, rmse_imag)
    except (RuntimeError, MemoryError) as e:
        if "can't allocate memory" in str(e) or "out of memory" in str(e):
            print("Mistake: Not enough memory.")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        # Получаем последние значения RMSE из истории, если возможно
        last_rmse_real = float('nan')
        last_rmse_imag = float('nan')
        if model.history["rmse_real"] and model.history["rmse_imag"]:
            last_rmse_real = model.history["rmse_real"][-1]
            last_rmse_imag = model.history["rmse_imag"][-1]

        # Логирование гиперпараметров с последними доступными значениями RMSE
        model.logs_tensorboard.log_hparams_and_metrics(last_rmse_real, last_rmse_imag)
    finally:
        try:
            model.save_model_pt()
            print('Model saved successfully.')
        except Exception as ex:
            print(f'The model could not be saved due to: {ex}')
            
        # Освобождение памяти
        del df, model
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
