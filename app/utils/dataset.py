import torch
from torch.utils.data import Dataset

class UpDownStockDataset(Dataset):
    '''`x` is the tensor containing close price of the stock while `y` is whether or not the price will be higher or lower than x[-1]'''
    def __init__(self, data, seq_length:int = 7, prediction_step:int = 0, profit_rate: float = 0.01, transforms = None):
            
        self.data = torch.tensor(data)
        self.seq_length = seq_length
        if not prediction_step:
            prediction_step = seq_length
        self.prediction_step = prediction_step
        self.transforms = transforms
        self.profit_rate = profit_rate
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.prediction_step
    
    def __getitem__(self, index):
        x = self.data[index: index+self.seq_length].float()
        target = self.data[index+self.seq_length+self.prediction_step - 1].float()
        y = (target > x[-1] * (1+self.profit_rate)).long()
        
        # Apply transformations
        if self.transforms is not None:
            x = self.transforms(x)
        return x, y