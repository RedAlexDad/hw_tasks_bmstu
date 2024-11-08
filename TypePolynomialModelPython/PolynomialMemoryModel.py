import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class PolynomialMemoryModel:
    def __init__(self, df, K, M):
        self.df = self.prepare_data(df)
        self.K = K
        self.M = M
        self.A = None
        self.input_data, self.output_data = self.df['input'].to_numpy(), self.df['output'].to_numpy()
        
    def prepare_data(self, df):
        """
        Предобработка данных: разделение на реальные и мнимые части
        """
        df.columns = df.columns.str.lower()
        df['input'] = df['input'].apply(lambda x: complex(x))
        df['output'] = df['output'].apply(lambda x: complex(x))
        df['input_real'] = df['input'].apply(lambda x: x.real)
        df['input_imag'] = df['input'].apply(lambda x: x.imag)
        df['output_real'] = df['output'].apply(lambda x: x.real)
        df['output_imag'] = df['output'].apply(lambda x: x.imag)
        # df = df.drop(['input', 'output'], axis=1)

        df['input'] = df['input_real'] + 1j * df['input_imag']
        df['output'] = df['output_real'] + 1j * df['output_imag']

        return df

    def fit(self):
        phi_MP = []
        for n in range(self.M, len(self.input_data)):
            row = []
            for m in range(self.M + 1):
                for k in range(1, self.K + 1):
                    term = self.input_data[n - m] * np.abs(self.input_data[n - m])**(k - 1)
                    row.append(term)
            phi_MP.append(row)
        
        phi_MP = np.array(phi_MP)
        y_trimmed = self.output_data[self.M:]
        
        self.A, _, _, _ = np.linalg.lstsq(phi_MP, y_trimmed, rcond=None)

    def predict(self):
        if self.A is None:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred = []
        for n in range(self.M, len(self.input_data)):
            phi_n = []
            for m in range(self.M + 1):
                for k in range(1, self.K + 1):
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
K = 3  # Порядок нелинейности
M = 2  # Глубина памяти

# Создание и обучение модели
model = PolynomialMemoryModel(df, K, M)
model.fit()

# Предсказание
y_pred = model.predict()

# Расчет RMSE
y_true = model.df['output'].to_numpy()[M:]
rmse_real, rmse_imag = model.calculate_rmse(y_true, y_pred)
print(f'RMSE (Real part): {rmse_real}, RMSE (Imaginary part): {rmse_imag}')
