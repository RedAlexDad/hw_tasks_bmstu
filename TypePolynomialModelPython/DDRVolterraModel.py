import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class DDRVolterraModel:
    def __init__(self, df, K, M):
        self.df = self.prepare_data(df)
        self.K = K
        self.M = M
        self.h0 = None
        self.h1 = None
        self.h2 = None
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
        y_trimmed = self.output_data[self.M:]
        phi_0 = []
        phi_1 = []
        phi_2 = []

        for n in range(self.M, len(self.input_data)):
            # Zeroth-order terms
            row_0 = [self.input_data[n]**k for k in range(1, self.K + 1)]
            phi_0.append(row_0)

            # First-order deviation reduction terms
            row_1 = []
            for k in range(1, self.K + 1):
                for m in range(1, self.M + 1):
                    if n - m >= 0:
                        term = (self.input_data[n]**(k-1)) * self.input_data[n - m]
                        row_1.append(term)
            phi_1.append(row_1)
            
            # Second-order deviation reduction terms
            row_2 = []
            for k in range(2, self.K + 1):
                for m1 in range(1, self.M + 1):
                    for m2 in range(m1, self.M + 1):
                        if n - m1 >= 0 and n - m2 >= 0:
                            term = (self.input_data[n]**(k-2)) * self.input_data[n - m1] * self.input_data[n - m2]
                            row_2.append(term)
            phi_2.append(row_2)

        phi_0 = np.array(phi_0)
        phi_1 = [np.pad(r, (0, max(len(x) for x in phi_1) - len(r)), 'constant') for r in phi_1]
        phi_1 = np.array(phi_1)
        phi_2 = [np.pad(r, (0, max(len(x) for x in phi_2) - len(r)), 'constant') for r in phi_2]
        phi_2 = np.array(phi_2)

        self.h0, _, _, _ = np.linalg.lstsq(phi_0, y_trimmed, rcond=None)
        self.h1, _, _, _ = np.linalg.lstsq(phi_1, y_trimmed, rcond=None)
        self.h2, _, _, _ = np.linalg.lstsq(phi_2, y_trimmed, rcond=None)

    def predict(self):
        if self.h0 is None or self.h1 is None or self.h2 is None:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred = []
        for n in range(self.M, len(self.input_data)):
            # Zeroth-order terms
            phi_n_0 = np.array([self.input_data[n]**k for k in range(1, self.K + 1)])
            
            # First-order deviation reduction terms
            phi_n_1 = []
            for k in range(1, self.K + 1):
                for m in range(1, self.M + 1):
                    if n - m >= 0:
                        term = (self.input_data[n]**(k-1)) * self.input_data[n - m]
                        phi_n_1.append(term)
            phi_n_1 = np.pad(phi_n_1, (0, len(self.h1) - len(phi_n_1)), 'constant')

            # Second-order deviation reduction terms
            phi_n_2 = []
            for k in range(2, self.K + 1):
                for m1 in range(1, self.M + 1):
                    for m2 in range(m1, self.M + 1):
                        if n - m1 >= 0 and n - m2 >= 0:
                            term = (self.input_data[n]**(k-2)) * self.input_data[n - m1] * self.input_data[n - m2]
                            phi_n_2.append(term)
            phi_n_2 = np.pad(phi_n_2, (0, len(self.h2) - len(phi_n_2)), 'constant')

            # Calculate the predicted output
            y_n = np.dot(phi_n_0, self.h0) + np.dot(phi_n_1, self.h1) + np.dot(phi_n_2, self.h2)
            y_pred.append(y_n)
        
        return np.array(y_pred)

    def calculate_rmse(self, y_true, y_pred):
        rmse_real = np.sqrt(mean_squared_error(y_true.real, y_pred.real))
        rmse_imag = np.sqrt(mean_squared_error(y_true.imag, y_pred.imag))
        return rmse_real, rmse_imag

# Загрузка и подготовка данных
df = pd.read_csv('Amp_C_train.txt')

# Параметры модели
K = 3  # Порядок нелинейности
M = 5  # Глубина памяти

# Создание и обучение модели
model = DDRVolterraModel(df, K, M)
model.fit()

# Предсказание
y_pred = model.predict()

# Расчет RMSE
y_true = df['output'].to_numpy()[M:]
rmse_real, rmse_imag = model.calculate_rmse(y_true, y_pred)
print(f'RMSE (Real part): {rmse_real}, RMSE (Imaginary part): {rmse_imag}')
