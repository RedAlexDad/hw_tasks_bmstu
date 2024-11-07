import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class ExponentiallyShapedMemoryDelayProfileModel:
    def __init__(self, df, K, M, delta_0, alpha):
        self.df = self.prepare_data(df)
        self.K = K
        self.M = M
        self.delta_0 = delta_0
        self.alpha = alpha
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

    def compute_delay(self, m, k):
        if m == 0:
            return 0
        avg_delay = np.mean(np.abs(self.input_data))
        delay = avg_delay + self.delta_0 * np.exp(-self.alpha * k)
        return min(int(delay), len(self.input_data) - 1)

    def fit(self):
        phi_MPSD = []
        for n in range(self.M, len(self.input_data)):
            row = []
            for m in range(self.M + 1):
                for k in range(1, self.K + 1):
                    delta_mk = self.compute_delay(m, k)
                    if n - delta_mk >= 0:
                        term = self.input_data[n - delta_mk] * np.abs(self.input_data[n - delta_mk])**(k - 1)
                        row.append(term)
            if row:  # Ensure row is not empty
                phi_MPSD.append(row)
        
        max_len = max(len(row) for row in phi_MPSD)
        phi_MPSD = np.array([np.pad(row, (0, max_len - len(row)), mode='constant') for row in phi_MPSD])
        y_trimmed = self.output_data[self.M:self.M + len(phi_MPSD)]
        
        self.A, _, _, _ = np.linalg.lstsq(phi_MPSD, y_trimmed, rcond=None)

    def predict(self):
        if self.A is None:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred = []
        for n in range(self.M, len(self.input_data)):
            phi_n = []
            for m in range(self.M + 1):
                for k in range(1, self.K + 1):
                    delta_mk = self.compute_delay(m, k)
                    if n - delta_mk >= 0:
                        term = self.input_data[n - delta_mk] * np.abs(self.input_data[n - delta_mk])**(k - 1)
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
K = 3      # Порядок нелинейности
M = 3      # Глубина памяти
delta_0 = 2   # Максимальная задержка
alpha = 0.5    # Коэффициент уменьшения

# Создание и обучение модели
model = ExponentiallyShapedMemoryDelayProfileModel(df, K, M, delta_0, alpha)
model.fit()

# Предсказание
y_pred = model.predict()

# Расчет RMSE
y_true = df['output'].to_numpy()[M:M + len(y_pred)]
rmse_real, rmse_imag = model.calculate_rmse(y_true, y_pred)
print(f'RMSE (Real part): {rmse_real}, RMSE (Imaginary part): {rmse_imag}')

# RMSE (Real part): 0.20646827348574384, RMSE (Imaginary part): 0.2073441555373588
