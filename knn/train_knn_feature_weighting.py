import logging
logging.basicConfig(level=logging.INFO)
import os
import numpy as np
import pandas as pd
import parameters as pr
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from create_dataset import array_trainX, array_trainY, val_dataloader, create
from models.knn import WeightedKNearestNeighbors
from utils.helper import get_file_name, metric
import matplotlib.pyplot as plt

# Định nghĩa lớp LinearModel và LinearAutoEncoder
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

# Hàm huấn luyện Autoencoder
def train_autoencoder(array_trainX, seq_length, latent_size, device):
    encoder_layers = [seq_length // 2, latent_size]
    autoencoder = LinearAutoEncoder(in_features=seq_length, encoder_layers=encoder_layers).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    num_epochs = 100
    data_tensor = torch.tensor(array_trainX, dtype=torch.float32).to(device)
    
    for epoch in range(num_epochs):
        autoencoder.train()
        optimizer.zero_grad()
        output = autoencoder(data_tensor)
        loss = criterion(output, data_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.2f}")
    
    save_dir = os.path.join(os.path.expanduser("~/Documents"), "knn", "models", "pretrained-autoencoder")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"autoencoder-in_features{seq_length}-latent_size{latent_size}.pt")
    torch.jit.script(autoencoder).save(save_path)
    return autoencoder

# Hàm tải Autoencoder
def load_autoencoder(seq_length, latent_size, device='cpu'):
    model_path = os.path.join(os.path.expanduser("~/Documents"), "knn", "models", "pretrained-autoencoder", f"autoencoder-in_features{seq_length}-latent_size{latent_size}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Autoencoder model not found at {model_path}")
    autoencoder = torch.jit.load(model_path)
    autoencoder.eval()
    return autoencoder.to(device)

# Hàm giảm chiều dữ liệu
def encode_data(data, autoencoder, device='cpu'):
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        encoded_data = autoencoder.encoder(data_tensor)
        return encoded_data.cpu().numpy()

# Gọi create() để tạo dữ liệu nếu chưa có
if array_trainX is None or array_trainY is None or val_dataloader is None:
    train_dataloader, val_dataloader, array_trainX, array_trainY = create()
    if array_trainX is None:
        raise ValueError("Failed to create dataset. Please check the logs for errors.")

# Lấy close_prices từ file CSV để sử dụng trong dự đoán giá
data_path = os.path.join(os.path.expanduser("~/Documents"), "knn", "data", "^GSCP.csv")
data_df = pd.read_csv(data_path)
data_df['Date'] = pd.to_datetime(data_df['Date'], format='%Y-%m-%d', errors='coerce')
data_df = data_df[(data_df['Date'] >= pd.to_datetime('2020-01-01')) & (data_df['Date'] <= pd.to_datetime('2025-01-01'))]
close_prices = data_df['Close'].str.replace(',', '').astype(float).values

# Các tham số
seq_length = 180  # Dựa trên array_trainX.shape[1]
latent_sizes = [3, 6, 12, 30, 60]
percentiles = [5, 10, 25, 100]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
total_samples = 1670  # Theo bảng
samples = {5: 84, 10: 167, 25: 417, 100: 1670}  # Số mẫu theo bảng

# Tạo danh sách cột một lần duy nhất
columns = ['Latent size']
for p in percentiles:
    columns.extend([
        f"Top {p}% ({samples[p]}/{total_samples} samples) Accuracy",
        "Precision",
        "Recall"
    ])

# Lặp qua các latent sizes để huấn luyện và đánh giá
for latent_size in latent_sizes:
    print(f"\nTraining and evaluating with latent size: {latent_size}")

    # Huấn luyện hoặc tải Autoencoder
    try:
        autoencoder = load_autoencoder(seq_length, latent_size, device)
    except FileNotFoundError:
        autoencoder = train_autoencoder(array_trainX, seq_length, latent_size, device)

    # Giảm chiều dữ liệu
    encoded_trainX = encode_data(array_trainX, autoencoder, device)
    val_data = []
    val_labels = []
    val_actual_prices = []
    val_indices = []
    for i, (x, y) in enumerate(val_dataloader):
        encoded_x = encode_data(x.numpy(), autoencoder, device)
        val_data.append(encoded_x)
        val_labels.append(y.numpy())
        start_idx = len(array_trainX) + i * x.shape[0]
        end_idx = start_idx + x.shape[0]
        val_indices.extend(list(range(start_idx, end_idx)))
    val_data = np.concatenate(val_data, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_actual_prices = close_prices[val_indices]

    # Khởi tạo trọng số đặc trưng (feature weights) cho dữ liệu đã giảm chiều
    weights = np.ones(encoded_trainX.shape[1])  # Khởi tạo toàn số 1

    # Khởi tạo K-NN với feature weighting trên dữ liệu đã giảm chiều
    knn = WeightedKNearestNeighbors(x=encoded_trainX,
                                    y=array_trainY,
                                    k=1000,
                                    similarity='cosine',
                                    weights=weights,
                                    learning_rate=0.1,
                                    device=pr.device,
                                    train_split_ratio=pr.wknn_train_split_ratio)

    print("Training K-NN with feature weighting...")
    for epoch in range(100):
        knn.train()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/100 completed.")
    print("Training completed.")

    # Đánh giá trên tập kiểm tra
    pred = []
    logits = []
    targ = []
    for i in range(len(val_data)):
        x = torch.tensor(val_data[i], dtype=torch.float32).unsqueeze(0)
        prediction = knn.predict(x, reduction="score")
        pred.append(prediction[0][0])
        logits.append(prediction[1][0])
        targ.append(val_labels[i])
    pred = torch.tensor(pred)
    logits = torch.tensor(logits)
    targ = torch.tensor(targ)

    # Tạo bảng để lưu metrics
    metrics_table = {p: {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0} for p in percentiles}

    # Tính metrics theo các ngưỡng percentile
    for percentile in percentiles:
        lim_score = np.percentile(pred, 100 - percentile)
        confusion_matrix = np.zeros((2, 2), int)
        for idx, (x, y) in enumerate(zip(logits, targ)):
            if pred[idx] < lim_score:
                continue
            confusion_matrix[x, y] += 1
        metrics = metric(confusion_matrix, verbose=False)
        metrics_table[percentile]['Accuracy'] = metrics[0]
        metrics_table[percentile]['Precision'] = metrics[1]
        metrics_table[percentile]['Recall'] = metrics[2]

    # Chuẩn bị dữ liệu cho bảng
    table_data = [[latent_size]]
    for percentile in percentiles:
        acc = metrics_table[percentile]['Accuracy']
        prec = metrics_table[percentile]['Precision']
        rec = metrics_table[percentile]['Recall']
        table_data[0].extend([f"{acc:.2f}", f"{prec:.2f}", f"{rec:.2f}"])

    # Tạo DataFrame
    df = pd.DataFrame(table_data, columns=columns)

    # Hiển thị bảng
    print("\nTable 3: K-NN classifier with Autoencoder for dimensional reduction")
    print(df.to_string(index=False))

    # Lưu bảng vào file CSV
    output_path = os.path.join(os.path.expanduser("~/Documents"), "knn", f"knn_ae_metrics_latent_{latent_size}.csv")
    df.to_csv(output_path, index=False)
    print(f"\nMetrics table saved to {output_path}")

    # In trọng số đặc trưng đã tối ưu
    print("\nOptimized feature weights:")
    print(knn.weights)

    # Dự đoán giá cổ phiếu
    predicted_prices = []
    current_idx = 0
    for i in range(len(val_data)):
        x = torch.tensor(val_data[i], dtype=torch.float32).unsqueeze(0)
        prediction = knn.predict(x, reduction="score")
        predicted_label = prediction[1][0]
        last_price = val_actual_prices[current_idx - 1] if current_idx > 0 else val_actual_prices[0]
        profit_rate = np.median([(close_prices[j + 1] / close_prices[j]) - 1 for j in range(len(close_prices) - 1)])
        if predicted_label == 1:
            predicted_price = last_price * (1 + profit_rate)
        else:
            predicted_price = last_price * (1 - profit_rate)
        predicted_prices.append(predicted_price)
        current_idx += 1

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(val_actual_prices, label='Test Target', color='blue')
    plt.plot(predicted_prices, label='Predict Target', color='orange')
    plt.xlabel('Day')
    plt.ylabel('Close Price ($)')
    plt.legend()
    plt.grid(True)

    # Lưu biểu đồ
    output_dir = os.path.join(os.path.expanduser("~/Documents"), "knn", "prediction")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"stock_price_prediction_wknn_ae_latent_{latent_size}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Prediction chart saved to {plot_path}")