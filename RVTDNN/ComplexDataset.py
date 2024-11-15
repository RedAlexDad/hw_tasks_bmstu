import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

# Создаем класс датасета
class ComplexDataset(Dataset):
    def __init__(self, input_file, time_delay):
        self.input_file = input_file
        self.df = self.prepare_data(pd.read_csv(self.input_file))
        self.input_real = self.df['input_real'].values
        self.input_imag = self.df['input_imag'].values
        self.output_real = self.df['output_real'].values
        self.output_imag = self.df['output_imag'].values
        self.time_delay = time_delay
    
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
        df = df.set_index('time')
        return df

    def __len__(self):
        return len(self.input_real) - self.time_delay

    def __getitem__(self, idx):
        idx_end = idx + self.time_delay + 1
        I_in = torch.tensor(self.input_real[idx:idx_end], dtype=torch.float32)
        Q_in = torch.tensor(self.input_imag[idx:idx_end], dtype=torch.float32)
        I_out = torch.tensor(self.output_real[idx + self.time_delay], dtype=torch.float32)
        Q_out = torch.tensor(self.output_imag[idx + self.time_delay], dtype=torch.float32)
        return (I_in, Q_in), (I_out, Q_out)