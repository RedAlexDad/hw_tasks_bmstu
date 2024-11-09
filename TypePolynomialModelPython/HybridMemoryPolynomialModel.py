import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class HybridMemoryPolynomialModel:
    def __init__(self, df, K, M, K_e, M_e):
        self.df = self.prepare_data(df)
        self.K = K
        self.M = M
        self.K_e = K_e
        self.M_e = M_e
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
        phi_HMP = []
        for n in range(max(self.M, self.M_e), len(self.input_data)):
            row = self.construct_mp_terms(n) + self.construct_envmp_terms(n)
            phi_HMP.append(row)
        
        max_len = max(len(row) for row in phi_HMP)
        phi_HMP = np.array([np.pad(row, (0, max_len - len(row)), mode='constant') for row in phi_HMP])
        
        y_trimmed = self.output_data[max(self.M, self.M_e):]
        self.A, _, _, _ = np.linalg.lstsq(phi_HMP, y_trimmed, rcond=None)

    def construct_mp_terms(self, n):
        terms = []
        for m in range(self.M + 1):
            for k in range(1, self.K + 1):
                term = self.input_data[n - m] * np.abs(self.input_data[n - m])**(k - 1)
                terms.append(term)
        return terms

    def construct_envmp_terms(self, n):
        terms = []
        for m in range(1, self.M_e + 1):
            for k in range(2, self.K_e + 1):
                if n - m >= 0:
                    term = self.input_data[n] * np.abs(self.input_data[n - m])**(k - 1)
                    terms.append(term)
        return terms

    def predict(self):
        if self.A is None:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred = []
        for n in range(max(self.M, self.M_e), len(self.input_data)):
            phi_n = self.construct_mp_terms(n) + self.construct_envmp_terms(n)
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
K = 3    # Порядок нелинейности для полинома памяти
M = 2    # Глубина памяти для полинома памяти
K_e = 3  # Порядок нелинейности для огибающей полинома памяти
M_e = 3  # Глубина памяти для огибающей полинома памяти

# Создание и обучение модели
model = HybridMemoryPolynomialModel(df, K, M, K_e, M_e)
model.fit()

# Предсказание
y_pred = model.predict()

# Расчет RMSE
y_true = df['output'].to_numpy()[max(M, M_e):]
rmse_real, rmse_imag = model.calculate_rmse(y_true, y_pred)
print(f'RMSE (Real part): {rmse_real}, RMSE (Imaginary part): {rmse_imag}')
