import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

class PolynomialDataset:
    def __init__(self, df, M, K, model_type, batch_size=2**10):
        self.df = self.prepare_data(df)
        self.M = M
        self.K = K
        self.model_type = model_type
        self.batch_size = batch_size
        
        self.X, self.y, self.times = self.prepare_dataset(self.df, model_type, M, K)
        self.dataloader = self._create_tensor_dataset(self.X, self.y, self.times, self.batch_size)

    @staticmethod
    def prepare_data(df):
        df.columns = df.columns.str.lower()
        df['input'] = df['input'].apply(lambda x: complex(x))
        df['output'] = df['output'].apply(lambda x: complex(x))
        df['input_real'] = df['input'].apply(lambda x: x.real)
        df['input_imag'] = df['input'].apply(lambda x: x.imag)
        df['output_real'] = df['output'].apply(lambda x: x.real)
        df['output_imag'] = df['output'].apply(lambda x: x.imag)
        df = df.drop(['input', 'output'], axis=1)
        df = df.set_index('time')  # 'time' колонка теперь индекс
        return df

    @staticmethod
    def _create_tensor_dataset(X, y, times, batch_size, shuffle_status=True):
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(times, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_status)
    
    def prepare_dataset(self, df, model_type, M, K):
        """Подготавливает данные на основе типа модели для оценки."""
        x_real, x_imag, y_real, y_imag, times = self._prepare_data_for_create_dataset(df, M)
        
        if model_type == 'memory_polynomial':
            return self._create_memory_polynomial(x_real, x_imag, y_real, y_imag, times, M, K)
        elif model_type == 'sparse_delay_memory_polynomial':
            delays = range(self.M + 1)
            return self._create_sparse_delay_polynomial(x_real, x_imag, y_real, y_imag, times, M, K, delays)
        elif model_type == 'non_uniform_memory_polynomial':
            K_list = [self.K] * (self.M + 1)
            return self._create_non_uniform_memory_polynominal(x_real, x_imag, y_real, y_imag, times, M, K_list)
        elif model_type == 'envelope_memory_polynomial':
            return self._create_envelope_memory_polynomial(x_real, x_imag, y_real, y_imag, times, M, K)
        else:
            raise ValueError(f"Unknown model type")
    
    @staticmethod
    def _prepare_data_for_create_dataset(df, M):
        x_real = df['input_real'].values
        x_imag = df['input_imag'].values
        y_real = df['output_real'].values[M:]
        y_imag = df['output_imag'].values[M:]
        times = df.index.values[M:]
        return x_real, x_imag, y_real, y_imag, times

    @staticmethod
    def _create_memory_polynomial(x_real, x_imag, y_real, y_imag, times, M, K):
        N = len(x_real)
        X = np.zeros((N, (M + 1) * K * 2), dtype=np.float64)
        y = np.stack([y_real, y_imag], axis=1)
        for n in range(M, N):
            index = 0
            for m in range(M + 1):
                for k in range(1, K + 1):
                    X[n, index] = np.abs(x_real[n - m])**(k-1) * x_real[n - m]
                    X[n, index + 1] = np.abs(x_imag[n - m])**(k-1) * x_imag[n - m]
                    index += 2
        return X[M:], y, times
    
    @staticmethod
    def _create_sparse_delay_polynomial(x_real, x_imag, y_real, y_imag, times, M, K, delays):
        N = len(x_real)
        X = np.zeros((N - M, len(delays) * K * 2), dtype=np.float64)
        y = np.stack([y_real[M:], y_imag[M:]], axis=1)
        for n in range(M, N):
            index = 0
            for m in delays:
                if n - m >= 0:
                    for k in range(1, K + 1):
                        X[n - M, index] = np.abs(x_real[n - m])**(k - 1) * x_real[n - m]
                        X[n - M, index + 1] = np.abs(x_imag[n - m])**(k - 1) * x_imag[n - m]
                    index += 2
        X = X[:len(y)]  # Срез в зависимости от наименьшего размера
        times = times[M:M + len(y)]  # Приводим ко всем одинаковым размерам
        return X, y, times
    
    @staticmethod
    def _create_non_uniform_memory_polynominal(x_real, x_imag, y_real, y_imag, times, M, K_list):
        N = len(x_real)
        X = np.zeros((N, sum(K_list) * 2), dtype=np.float64)
        y = np.stack([y_real, y_imag], axis=1)
        index = 0
        for m in range(self.M + 1):
            for k in range(1, K_list[m] + 1):
                for n in range(self.M, N):
                    X[n, index] = np.abs(x_real[n - m])**(k-1) * x_real[n - m]
                    X[n, index + 1] = np.abs(x_imag[n - m])**(k-1) * x_imag[n - m]
                index += 2
        return X[M:], y, times
    
    @staticmethod
    def _create_envelope_memory_polynomial(x_real, x_imag, y_real, y_imag, times, M, K):
        amplitude = np.sqrt(x_real**2 + x_imag**2)  # Амплитуда сигнала
        N = len(x_real)
        X = np.zeros((N - M, (M + 1) * K), dtype=np.float64)
        y = np.stack([y_real, y_imag], axis=1)
        for n in range(M, N):
            index = 0
            for m in range(M + 1):
                for k in range(1, K + 1):
                    X[n - M, index] = amplitude[n] * amplitude[n - m]**(k-1)
                    index += 1
        X = X[:len(y[M:])] # Срез в зависимости от наименьшего размера
        return X, y[M:], times[M:]

