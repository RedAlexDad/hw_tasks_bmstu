import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class GeneralizedMemoryPolynomialModel:
    def __init__(self, df, K_a, M_a, K_b, M_b, P, K_c, M_c, Q):
        self.df = self.prepare_data(df)
        self.K_a = K_a
        self.M_a = M_a
        self.K_b = K_b
        self.M_b = M_b
        self.P = P
        self.K_c = K_c
        self.M_c = M_c
        self.Q = Q
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
        phi_GMP = []
        for n in range(max(self.M_a, self.M_b + self.P, self.M_c + self.Q), len(self.input_data)):
            row = self.construct_aligned_terms(n) + self.construct_lagging_cross_terms(n) + self.construct_leading_cross_terms(n)
            phi_GMP.append(row)
        
        # Определяем максимальную длину и приводим все строки к одной длине
        max_len = max(len(row) for row in phi_GMP)
        phi_GMP = np.array([np.pad(row, (0, max_len - len(row)), mode='constant') for row in phi_GMP])
        
        y_trimmed = self.output_data[max(self.M_a, self.M_b + self.P, self.M_c + self.Q):]
        self.A, _, _, _ = np.linalg.lstsq(phi_GMP, y_trimmed, rcond=None)

    def construct_aligned_terms(self, n):
        terms = []
        for m in range(self.M_a + 1):
            for k in range(1, self.K_a + 1):
                term = self.input_data[n - m] * np.abs(self.input_data[n - m])**(k - 1)
                terms.append(term)
        return terms

    def construct_lagging_cross_terms(self, n):
        terms = []
        for m in range(self.M_b + 1):
            for k in range(2, self.K_b + 1):
                for p in range(1, self.P + 1):
                    if n - m - p >= 0:
                        term = self.input_data[n - m] * np.abs(self.input_data[n - m - p])**(k - 1)
                        terms.append(term)
        return terms

    def construct_leading_cross_terms(self, n):
        terms = []
        for m in range(self.M_c + 1):
            for k in range(2, self.K_c + 1):
                for q in range(1, self.Q + 1):
                    if n - m + q < len(self.input_data):
                        term = self.input_data[n - m] * np.abs(self.input_data[n - m + q])**(k - 1)
                        terms.append(term)
        return terms

    def predict(self):
        if self.A is None:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred = []
        for n in range(max(self.M_a, self.M_b + self.P, self.M_c + self.Q), len(self.input_data)):
            phi_n = self.construct_aligned_terms(n) + self.construct_lagging_cross_terms(n) + self.construct_leading_cross_terms(n)
            
            # Pad phi_n to match dimensions with self.A
            phi_n = np.pad(phi_n, (0, len(self.A) - len(phi_n)), mode='constant')
            
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
K_a = 3  # Порядок нелинейности для выровненных членов
M_a = 2  # Глубина памяти для выровненных членов
K_b = 2  # Порядок нелинейности для лаговых членов
M_b = 2  # Глубина памяти для лаговых членов
P = 1    # Порядок лаговых членов
K_c = 2  # Порядок нелинейности для ведущих членов
M_c = 2  # Глубина памяти для ведущих членов
Q = 1    # Порядок ведущих членов

# Создание и обучение модели
model = GeneralizedMemoryPolynomialModel(df, K_a, M_a, K_b, M_b, P, K_c, M_c, Q)
model.fit()

# Предсказание
y_pred = model.predict()

# Расчет RMSE
y_true = df['output'].to_numpy()[max(M_a, M_b + P, M_c + Q):]
rmse_real, rmse_imag = model.calculate_rmse(y_true, y_pred)
print(f'RMSE (Real part): {rmse_real}, RMSE (Imaginary part): {rmse_imag}')
