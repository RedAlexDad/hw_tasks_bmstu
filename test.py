import gc   
import argparse
import resource

from ModelEvalutor import ModelEvalutor

def argparse_setup():
    """Configuring and processing command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Polynomial Model')
    
    parser.add_argument('-f', '--input_file', type=str, default='Amp_C_train.txt', help='Path to the input data file')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the saved model file')
    parser.add_argument('-d', '--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Device to use for computation')
    
    return parser.parse_args()
                
# Основной код для инициализации
def main(args):             
    model = ModelEvalutor(
        input_file=args.input_file,
        model_path=args.model_path
    )
    
    rmse_real, rmse_imag = model.evaluate()

    print(f'Evaluation RMSE (Real): {rmse_real:.6f}')
    print(f'Evaluation RMSE (Imag): {rmse_imag:.6f}')

if __name__ == '__main__':
    # Установка максимального лимита по памяти (в байтах)
    max_memory = 30 * 2**10 * 2**10 * 2**10  # 30 ГБ
    
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
    except Exception as e:
        print(f'The memory limit could not be set: {e}')

    args = argparse_setup()
    main(args)
