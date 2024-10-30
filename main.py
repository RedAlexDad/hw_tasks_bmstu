import gc   
import torch
import resource
import pandas as pd
import os

from MemoryPolynomialNNTrainer import MemoryPolynomialNNTrainer
from MemoryPolynomialTransformerTrainer import MemoryPolynomialTransformerTrainer

if __name__ == '__main__':
    # Установка максимального лимита по памяти (в байтах)
    # max_memory = 30 * 2**10  # 30 ГБ
    # resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
    print(f"Используемое устройство: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Проверка существования файла перед чтением
    try:
        df = pd.read_csv('Amp_C_train.txt')
    except FileNotFoundError:
        print("Файл 'Amp_C_train.txt' не найден.")
        exit(1)
        
    M = 3  # Глубина памяти
    K = 3  # Степень полинома
    
    rmse_real, rmse_imag = None, None                                       # Инициализируем переменные
    batch_size=2**10                                                         # Размер батча
    learning_rate=1e-3                                                      # Скорость обучения                         
    epochs=10                                                               # Количество эпох   
    # hidden_layers=[2**6, 2**7, 2**8, 2**9, 2**9, 2**8, 2**7, 2**6]          # Список скрытых слоев
    # hidden_layers=[2**10, 2**8, 2**6, 2**4]
    # hidden_layers=[2**6, 2**7, 2**7, 2**6]
    
    
    # hidden_layers=[2**8, 2**7, 2**6, 2**5, 2**4, 2**3]
    # hidden_layers=[2**7, 2**6, 2**5, 2**4, 2**3]
    # hidden_layers=[2**6, 2**5, 2**4, 2**3]
    hidden_layers=[2**3, 2**5, 2**3]
    # hidden_layers=[2**5, 2**3]
    dropout_rate=0.25
    
    # Создание экземпляра класса с настройкой гиперпараметров
    # model_mp = MemoryPolynomialTransformerTrainer(
    # model_nn = SparseDelayMemoryPolynomial(
    model_nn = MemoryPolynomialNNTrainer(
        df=df, 
        M=M, K=K, 
        batch_size=batch_size,
        learning_rate=learning_rate, 
        epochs=epochs, 
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        # model_type='memory_polynomial'
        # model_type='sparse_delay_memory_polynomial'
        # model_type='non_uniform_memory_polynomial'
        model_type='envelope_memory_polynomial'
    )
    
    model_nn.df.info()
    
    # model_nn.print_model_summary()
    
    # try:
    model_nn.train(max_early_stopping_counter=20)
    
    rmse_real, rmse_imag = model_nn.evaluate()

    print(f"Evaluation RMSE (Real): {rmse_real:.6f}")
    print(f"Evaluation RMSE (Imag): {rmse_imag:.6f}")
    # except Exception as e:
        # print(f"Обучение было прервано с ошибкой: {e}")
    # finally:
        # model_nn.log_hparams_and_metrics(rmse_real, rmse_imag)
        
        # Освобождение памяти
    del df
    gc.collect()