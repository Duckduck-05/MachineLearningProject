import logging
logging.basicConfig(level=logging.INFO)
import os
import numpy as np
import pandas as pd
import parameters as pr
import torch
from torch import nn
from create_dataset import array_trainX, array_trainY, val_dataloader, create
from models.knn import WeightedKNearestNeighbors
from utils.helper import get_file_name, metric
import matplotlib.pyplot as plt

# Định nghĩa lớp LinearModel và LinearAutoEncoder từ pretrain_ae.py
class LinearModel(nn.Module):
    def __init__(self, in_features: int, layers: list, activation: nn.Module = None):
        super(LinearModel, self).__init__()
        if activation is None:
            activation = nn.ReLU()
        self.activation = activation
        if not isinstance(layers, list):
            layers = list(layers)

        layers.insert(0, in_features)

        self.layers = []
        for idx in range(len(layers[:-1])):
            self.layers.append(nn.Linear(in_features=layers[idx], out_features=layers[idx+1]))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer.forward(x))
        x = self.layers[-1].forward(x)
        return x

class LinearAutoEncoder(nn.Module):
    def __init__(self, in_features: int, encoder_layers: list, activation: nn.Module = None, out_features: int = None, *, curve=None):
        super(LinearAutoEncoder, self).__init__()
        if activation is None:
            activation = nn.ReLU()
        self.activation = activation

        if not isinstance(encoder_layers, list):
            encoder_layers = list(encoder_layers)

        encoder_layers.insert(0, in_features)

        self.encoder = LinearModel(in_features=encoder_layers[0], layers=encoder_layers[1:], activation=self.activation)
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

def load_autoencoder(seq_length, latent_size, device='cpu'):
    """Tải mô hình Autoencoder từ file .pt."""
    project_dir = os.path.join(os.path.expanduser("~/Documents"), "knn")
    model_path = os.path.join(project_dir, "models", "pretrained-autoencoder", f"autoencoder-in_features{seq_length}-latent_size{latent_size}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Autoencoder model not found at {model_path}")
    autoencoder = torch.jit.load(model_path)
    autoencoder.eval()
    return autoencoder.to(device)

def encode_data(data, autoencoder, device='cpu'):
    """Mã hóa dữ liệu bằng Autoencoder."""
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        encoded_data = autoencoder.encoder(data_tensor)
        return encoded_data.cpu().numpy()

# Gọi create() để tạo dữ liệu nếu chưa có
if array_trainX is None or array_trainY is None or val_dataloader is None:
    train_dataloader, val_dataloader, array_trainX, array_trainY = create()
    if array_trainX is None:
        raise ValueError("Failed to create dataset. Please check the logs for errors.")

# Các tham số
latent_size = 30  # Sử dụng latent_size=30
seq_length = 180  # Giả định seq_length từ Autoencoder
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tải mô hình Autoencoder
autoencoder = load_autoencoder(seq_length, latent_size, device)

# Mã hóa dữ liệu huấn luyện
encoded_trainX = encode_data(array_trainX, autoencoder, device)

# Mã hóa dữ liệu kiểm tra từ val_dataloader và lấy giá thực tế
val_data = []
val_labels = []
val_actual_prices = []
data_path = os.path.join(os.path.expanduser("~/Documents"), "knn", "data", "^GSCP.csv")
data_df = pd.read_csv(data_path)
data_df['Date'] = pd.to_datetime(data_df['Date'], format='%Y-%m-%d', errors='coerce')
data_df = data_df[(data_df['Date'] >= pr.start_day) & (data_df['Date'] <= pr.end_day)]
close_prices = data_df['Close'].str.replace(',', '').astype(float).values

# Lấy giá thực tế từ tập kiểm tra
val_indices = []
for i, (x, y) in enumerate(val_dataloader):
    encoded_x = encode_data(x.numpy(), autoencoder, device)
    val_data.append(encoded_x)
    val_labels.append(y.numpy())
    # Tìm chỉ số của mẫu trong tập kiểm tra
    start_idx = len(array_trainX) + i * x.shape[0]
    end_idx = start_idx + x.shape[0]
    val_indices.extend(list(range(start_idx, end_idx)))
val_data = np.concatenate(val_data, axis=0)
val_labels = np.concatenate(val_labels, axis=0)
val_actual_prices = close_prices[val_indices]

# Khởi tạo weights mặc định để tránh lỗi (nhưng không sử dụng feature weighting)
weights = np.ones(encoded_trainX.shape[1])

# Khởi tạo và huấn luyện K-NN
knn = WeightedKNearestNeighbors(x=encoded_trainX,
                                y=array_trainY,
                                k=1000,
                                similarity='cosine',
                                weights=weights,
                                learning_rate=0.1,
                                device=pr.device,
                                train_split_ratio=pr.wknn_train_split_ratio)

print("Training K-NN with Autoencoder...")
for epoch in range(100):  # Huấn luyện 100 lần
    knn.train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/100 completed.")
print("Training completed.")

# Dự đoán giá cổ phiếu
predicted_prices = []
for i in range(len(val_data)):
    x = torch.tensor(val_data[i], dtype=torch.float32).unsqueeze(0)
    prediction = knn.predict(x, reduction="score")
    predicted_label = prediction[1][0]  # Nhãn dự đoán (0 hoặc 1: giảm/tăng)
    last_price = val_actual_prices[i - 1] if i > 0 else close_prices[val_indices[0] - 1]  # Giá trước đó
    # Chuyển nhãn thành giá dự đoán (giả định nhãn 1 là tăng, 0 là giảm)
    profit_rate = np.median([(close_prices[j + pr.prediction_step] / close_prices[j]) - 1 for j in range(len(close_prices) - pr.prediction_step)])
    if predicted_label == 1:  # Tăng
        predicted_price = last_price * (1 + profit_rate)
    else:  # Giảm
        predicted_price = last_price * (1 - profit_rate)
    predicted_prices.append(predicted_price)

# Vẽ biểu đồ so sánh giá thực tế và giá dự đoán
plt.figure(figsize=(12, 6))
plt.plot(val_actual_prices, label='Actual Price', color='blue', marker='o')
plt.plot(predicted_prices, label='Predicted Price', color='orange', marker='x')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Lưu biểu đồ
output_path = os.path.join(os.path.expanduser("~/Documents"), "knn", "stock_price_prediction.png")
plt.savefig(output_path)
print(f"Prediction chart saved to {output_path}")