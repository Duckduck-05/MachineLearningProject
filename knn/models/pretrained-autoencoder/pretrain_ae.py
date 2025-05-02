import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt

class StockDataset(Dataset):
    '''`x` is the tensor containing close price of the stock while `y` contains the close price of it'''
    def __init__(self, data, seq_length: int = 7, step: int = 1, num_x: int = 1, num_y: int = 1, y_length: int = None, transforms = None):
        self.data = torch.tensor(data, dtype=torch.float32)  # Đảm bảo kiểu float32
        self.seq_length = seq_length
        self.step = step
        if y_length is None:
            y_length = seq_length
        self.num_x = num_x
        self.num_y = num_y
        self.y_length = seq_length
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.num_x * self.step
    
    def __getitem__(self, index):
        x = torch.stack([self.data[index + self.step * i: index + self.step * i + self.seq_length].float() for i in range(self.num_x)]).float()
        y = torch.stack([self.data[index + self.step * i: index + self.step * i + self.y_length].float() for i in range(self.num_x, self.num_x + self.num_y)]).float()
        
        # Apply transformations
        if self.transforms is not None:
            x = self.transforms(x)
            y = self.transforms(y)
        return x, y
    
class LinearModel(nn.Module):
    def __init__(self, in_features: int, layers: list, activation: nn.Module = None):
        super(LinearModel, self).__init__()
        if activation is None:
            activation = nn.ReLU()
        self.activation = activation
        if not isinstance(layers, list):
            layers = list(layers)

        # Add the in_features to encoder_layers for easier creation of layers
        layers.insert(0, in_features)

        self.layers = []
        for idx in range(len(layers[:-1])):
            self.layers.append(nn.Linear(in_features=layers[idx],
                                         out_features=layers[idx+1],))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer.forward(x))
        x = self.layers[-1].forward(x)
        return x
    
class LinearAutoEncoder(nn.Module):
    def __init__(self, in_features: int, encoder_layers: list, activation: nn.Module = None, out_features: int = None, *, curve = None):
        super(LinearAutoEncoder, self).__init__()
        if activation is None:
            activation = nn.ReLU()
        self.activation = activation

        if not isinstance(encoder_layers, list):
            encoder_layers = list(encoder_layers)

        encoder_layers.insert(0, in_features)

        self.encoder = LinearModel(in_features=encoder_layers[0], layers = encoder_layers[1:], activation=self.activation)
        if out_features is not None:
            encoder_layers[0] = out_features
        if curve is None:
            self.decoder = LinearModel(in_features=encoder_layers[-1], layers=encoder_layers[::-1][1:], activation=self.activation)
        else:
            self.decoder = curve

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def ae_plot(model, inp, axes, title: str = None):
    '''Plot the result of an autoencoder and print the latent representation of it.'''
    # Get the latent representation
    latent = model.encoder(inp)
    
    # Get the reconstructed sample
    pred = model.decoder(latent)
    
    # Convert to numpy for easier visualization
    latent = latent.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    inp = inp.detach().cpu().numpy()
    
    # The latent representation
    print(f"The latent representation: {', '.join([f'{val:.4f}' for val in latent.flatten()])}")
    
    # Plot the reconstructed and the original samples
    axes.plot(inp, label = "Actual")
    axes.plot(pred, label = "Reconstructed")
    if title:
        axes.set_title(title)
    
def standardize(data: torch.Tensor):
    std, mean = torch.std_mean(data, dim=-1, keepdim=True)
    return (data - mean) / std

def min_max_scale(data: torch.Tensor):
    return (data - data.min(dim=-1, keepdim=True)[0]) / (data.max(dim=-1, keepdim=True)[0] - data.min(dim=-1, keepdim=True)[0])

# Đường dẫn tới dữ liệu
project_dir = os.path.join(os.path.expanduser("~/Documents"), "knn")
data_dir = os.path.join(project_dir, "data")
inp = os.path.join(data_dir, "^GSCP.csv")

# Kiểm tra và tải dữ liệu
if not os.path.exists(inp):
    raise FileNotFoundError(f"Data file not found at {inp}. Please ensure '^GSCP.csv' is in the 'data' folder.")
data_df = pd.read_csv(inp)
# data_df['Date'] = pd.to_datetime(data_df['Date'], format='%d-%b-%y')  # Định dạng ngày: 31-Dec-24
# data_df['Date'] = data_df['Date'].dt.strftime('%Y-%m-%d')  # Chuyển sang định dạng: 2024-12-31
# data_df['Date'] = pd.to_datetime(data_df['Date'])  # Chuyển lại thành datetime để lọc
# data_df = data_df[(data_df['Date'] >= '2020-01-01') & (data_df['Date'] <= '2025-01-01')]

# Kiểm tra và xử lý các cột Open, High, Low, Close, Adj Close, Volume
columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for col in columns_to_convert:
    if col not in data_df.columns:
        raise ValueError(f"Column '{col}' not found in the data file.")
    # Loại bỏ dấu phẩy và chuyển đổi sang float
    data_df[col] = data_df[col].str.replace(',', '').astype(float)

# Xử lý cột Close để sử dụng trong dataset
close_data = data_df['Close'].dropna()  # Loại bỏ giá trị NaN
if close_data.empty:
    raise ValueError("No valid data in 'Close' column after filtering.")
data = close_data.values  # Đã là mảng số float

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}.')

batch_size = 100
num_epochs = 7
split_ratio = [0.8, 0.2]

num_autoencoder_layers = 3  # The number of layers of the autoencoder. Layers' nodes number will be decreased by `ae_node_multiplier` every layers
ae_node_multiplier = 3

def wrapper(seq_length, latent_size):
    dataset = StockDataset(data=data,
                           seq_length=seq_length,
                           transforms=standardize,
                           step=0)

    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    autoencoder = LinearAutoEncoder(in_features=seq_length,  # seq_length and 1 features: volume and close price
                                     encoder_layers=[latent_size*(int(ae_node_multiplier**index)) for index in range(num_autoencoder_layers)[::-1]],
                                     activation=nn.ReLU(),
                                     out_features=seq_length,
                                     curve=None)
    #autoencoder = torch.jit.load(f"autoencoder-in_features{seq_length}-latent_size{latent_size}.pt")
    criterion = nn.L1Loss(reduction="mean")
    ae_learning_rate = 10**-3 * batch_size
    ae_optimizer = torch.optim.SGD(autoencoder.parameters(), lr=ae_learning_rate)
    ae_scheduler = torch.optim.lr_scheduler.ExponentialLR(ae_optimizer, gamma=0.3)

    autoencoder = autoencoder.to(device)
    autoencoder.train()
    for epoch in range(num_epochs):
        for idx, (sample, target) in enumerate(train_dataloader):
            ae_optimizer.zero_grad()

            x = torch.flatten(sample, 2).to(device)
            target = target.to(device)
            prediction = autoencoder(x)
            loss = criterion(prediction, target)  # x acts as both the input and the target
            loss.backward()
            ae_optimizer.step()
            if (torch.isnan(loss) or loss <= 0):
                print("Criterion changed.")
                ae_optimizer.zero_grad()
                criterion = nn.L1Loss()
            print(f"Sequence length: {seq_length}, latent_size: {latent_size} ----- epoch {epoch+1}, sample {idx+1}/{len(train_dataloader)}, loss: {loss.item():.6f}", end="\r")
        ae_scheduler.step()

    autoencoder.eval()
    autoencoder = autoencoder.cpu()
    print(f"Sequence length: {seq_length}, latent_size: {latent_size} ----- epoch {epoch+1}, sample {idx+1}/{len(train_dataloader)}, loss: {loss.item():.6f}")
    autoencoder = autoencoder.cpu()
    fig, axes = plt.subplots(3, 3)
    fig.set_size_inches(18.5, 10.5)
    axes = axes.flatten()
    for ax in axes:
        # Get a random sample and preprocess it
        idx = torch.randint(0, len(dataset), size=[1])
        x = torch.flatten(dataset[idx][0])
        ae_plot(autoencoder, x, ax)
    plt.legend()
    plt.show()
    # Lưu mô hình với đường dẫn đúng
    save_path = os.path.join(project_dir, "models", "pretrained-autoencoder", f"autoencoder-in_features{seq_length}-latent_size{latent_size}.pt")
    torch.jit.script(autoencoder).save(save_path)
    print(f"Model saved at {save_path}")

# Pairs of seq_length and latent_size
seq_length = [180]  
latent_size = [3, 6, 12, 30, 60]
for sl in seq_length:
    for ls in latent_size:
        print(f"({sl}, {ls})", end="\r")
        #if os.path.exists(f"autoencoder-in_features{sl}-latent_size{ls}.pt"): continue
        wrapper(sl, ls)