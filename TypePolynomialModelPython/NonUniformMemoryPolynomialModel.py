import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class NonUniformMemoryPolynomialModel:
    def __init__(self, df, M, K_list):
        self.df = self.prepare_data(df)
        self.M = M
        self.K_list = K_list
        self.A = None
        self.input_data, self.output_data = self.df['input'].to_numpy(), self.df['output'].to_numpy()
        
    def prepare_data(self, df):
        df.columns = df.columns.str.lower()
        df['input'] = df['input'].apply(lambda x: complex(x))
        df['output'] = df['output'].apply(lambda x: complex(x))
        df['input_real'] = df['input'].apply(lambda x: x.real)
        df['input_imag'] = df['input'].apply(lambda x: x.imag)
        df['output_real'] = df['output'].apply(lambda x: x.real)
        df['output_imag'] = df['output'].apply(lambda x: x.imag)
        df['input'] = df['input_real'] + 1j * df['input_imag']
        df['output'] = df['output_real'] + 1j * df['output_imag']
        return df

    def fit(self):
        phi_NUMP = []
        for n in range(self.M, len(self.input_data)):
            row = []
            for m in range(self.M + 1):
                K_m = self.K_list[m]
                for k in range(1, K_m + 1):
                    term = self.input_data[n - m] * np.abs(self.input_data[n - m])**(k - 1)
                    row.append(term)
            phi_NUMP.append(row)
        
        phi_NUMP = np.array(phi_NUMP)
        y_trimmed = self.output_data[self.M:]
        
        self.A, _, _, _ = np.linalg.lstsq(phi_NUMP, y_trimmed, rcond=None)

    def predict(self):
        if self.A is None:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred = []
        for n in range(self.M, len(self.input_data)):
            phi_n = []
            for m in range(self.M + 1):
                K_m = self.K_list[m]
                for k in range(1, K_m + 1):
                    term = self.input_data[n - m] * np.abs(self.input_data[n - m])**(k - 1)
                    phi_n.append(term)
            
            phi_n = np.array(phi_n)
            y_n = np.dot(phi_n, self.A)
            y_pred.append(y_n)
        
        return np.array(y_pred)

    def calculate_rmse(self, y_true, y_pred):
        rmse_real = np.sqrt(mean_squared_error(y_true.real, y_pred.real))
        rmse_imag = np.sqrt(mean_squared_error(y_true.imag, y_pred.imag))
        return rmse_real, rmse_imag

# Загрузка и подготовка данных
df = pd.read_csv('Amp_C_train.txt')

# Параметры модели
M = 5  # Глубина памяти
K_list = [3, 2, 2, 1, 1, 1]  # Порядки нелинейности для каждой ветви

# Создание и обучение модели
model = NonUniformMemoryPolynomialModel(df, M, K_list)
model.fit()

# Предсказание
y_pred = model.predict()

# Расчет RMSE
y_true = df['output'].to_numpy()[M:]
rmse_real, rmse_imag = model.calculate_rmse(y_true, y_pred)
print(f'RMSE (Real part): {rmse_real}, RMSE (Imaginary part): {rmse_imag}')
