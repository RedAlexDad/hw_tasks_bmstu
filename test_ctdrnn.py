import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from CTDRNN import CTDRNN

def load_model(model_path):
    p = 10
    q = 5
    input_size = 2 * (p + 1 + q)

    ctdrnn = CTDRNN(
        input_size=input_size, 
        hidden_sizes=[128, 256, 256, 128, 64],
        output_size=2, 
        num_layers=2,
        num_epochs=3,
        learning_rate=0.001,
        p=p,
        q=q,
        batch_size=2048
    )

    # Загрузите параметры модели
    ctdrnn.model.load_state_dict(torch.load(model_path))

    ctdrnn.model.eval()

    return ctdrnn

def F(X_in, load_model):    
    with torch.no_grad():
        X_in_tensor = torch.from_numpy(X_in).view(-1, load_model.input_size).to('cuda')
        X_in_tensor = torch.stack([X_in_tensor.real, X_in_tensor.imag], dim=1) 

        predicted_output = load_model.model(X_in_tensor).cpu().numpy()

    return predicted_output

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 main.py dataset.csv path_to_pretrained_model.pth")
        sys.exit(1)

    input_file = sys.argv[1]
    model_path = sys.argv[2]

    df = pd.read_csv(input_file)
    X_in = df['Input'].values.astype(np.complex64)

    # Загружаем модель
    model = load_model(model_path)

    # Делаем предсказания
    predictions = F(X_in, model)

    print("Predicted outputs:")
    for pred in predictions:
        print(pred)