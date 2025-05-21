import torch
from torch.utils.data import Dataset
import pandas as pd

class UpDownStockDataset(Dataset):
    '''`x` is the tensor containing the sequence of closing prices of the S&P 500,
    while `y` is a binary label indicating whether the price will be higher or lower
    than the last price in the sequence by at least the profit_rate'''
    def __init__(self, data_path, seq_length: int = 7, prediction_step: int = 0, profit_rate: float = 0.01, transforms=None):
        # Load CSV data and extract the 'Close' column
        df = pd.read_csv(data_path)
        self.data = torch.tensor(df['Close'].values, dtype=torch.float32)
        self.seq_length = seq_length
        if not prediction_step:
            prediction_step = seq_length
        self.prediction_step = prediction_step
        self.transforms = transforms
        self.profit_rate = profit_rate
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.prediction_step
    
    def __getitem__(self, index):
        # Extract sequence of closing prices
        x = self.data[index: index + self.seq_length].float()
        # Target price after prediction_step
        target = self.data[index + self.seq_length + self.prediction_step - 1].float()
        # Binary label: 1 if target > last price * (1 + profit_rate), 0 otherwise
        y = (target > x[-1] * (1 + self.profit_rate)).long()
        
        # Apply transformations if provided
        if self.transforms is not None:
            x = self.transforms(x)
        return x, y