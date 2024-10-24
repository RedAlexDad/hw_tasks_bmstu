import pandas as pd

from MemoryPolynomialNNTrainer import MemoryPolynomialNNTrainer

if __name__ == '__main__':
    df = pd.read_csv('Amp_C_train.txt')

    M = 3  # Глубина памяти
    K = 3  # Степень полинома
    
    batch_size=2**5                                                         # Размер батча
    learning_rate=1e-1                                                      # Скорость обучения                         
    epochs=10                                                               # Количество эпох   
    hidden_layers=[2**6, 2**7, 2**8, 2**9, 2**9, 2**8, 2**7, 2**6]          # Список скрытых слоев
    # hidden_layers=[2**6, 2**7, 2**7, 2**6]
    # hidden_layers=[2**3, 2**5, 2**3]
    dropout_rate=0.20 
    
    # Создание экземпляра класса с настройкой гиперпараметров
    model_nn = MemoryPolynomialNNTrainer(
        df=df, 
        M=M, K=K, 
        batch_size=batch_size,
        learning_rate=learning_rate, 
        epochs=epochs, 
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
    )
    
    model_nn.print_model_summary()
    
    model_nn.train(max_early_stopping_counter=20)
    
    rmse_real, rmse_imag = model_nn.evaluate()

    print(f"Evaluation RMSE (Real): {rmse_real:.6f}")
    print(f"Evaluation RMSE (Imag): {rmse_imag:.6f}")
    
    model_nn.log_hparams_and_metrics(rmse_real, rmse_imag)