# import logging
# logging.basicConfig(level=logging.INFO)
# import os

# import numpy as np
# import parameters as pr
# import torch
# from create_dataset import array_trainX, array_trainY, val_dataloader, create
# from models.knn import WeightedKNearestNeighbors
# from utils.helper import get_file_name, metric

# weights = 0
# knn = WeightedKNearestNeighbors(x=array_trainX,
#                                 y=array_trainY,
#                                 k=1000,
#                                 similarity='cosine',
#                                 weights=weights,
#                                 learning_rate=10**-1,
#                                 device=pr.device,
#                                 train_split_ratio=pr.wknn_train_split_ratio)
# def report():
#     pred = []
#     logits = []
#     targ = []
#     for (x, y) in val_dataloader:
#         prediction = knn.predict(x, reduction="score")
#         pred.extend(prediction[0])
#         logits.extend(prediction[1])
#         targ.extend(y.tolist())
#     pred = torch.tensor(pred)
#     logits = torch.tensor(logits)
#     targ = torch.tensor(targ)

#     confusion_matrix = np.zeros((2, 2), int)
#     for idx, (x, y) in enumerate(zip(logits, targ)):
#         confusion_matrix[x, y] += 1
#     print("*"*os.get_terminal_size().columns)
#     metric(confusion_matrix, verbose=True)
#     print(confusion_matrix)

#     limit_n = [5, 10, 25, 100]
#     res = []
#     for val in limit_n:
#         lim_score = np.percentile(pred, 100-val)
#         confusion_matrix = np.zeros((2, 2), int)
#         for idx, (x, y) in enumerate(zip(logits, targ)):
#             if pred[idx] < lim_score:
#                 continue
#             confusion_matrix[x, y] += 1
#         res.extend(metric(confusion_matrix, verbose=False)[:-1])
#     print(*[f"{val:.3f}" for val in res])

# report()
# knn.train(100, 10)
# report()

# import logging
# logging.basicConfig(level=logging.INFO)
# import os
# import numpy as np
# import parameters as pr
# import torch
# from create_dataset import array_trainX, array_trainY, val_dataloader, create
# from models.knn import WeightedKNearestNeighbors
# from utils.helper import get_file_name, metric
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Bỏ dòng thiết lập style vì seaborn tự động áp dụng style riêng
# # Nếu cần style khác, có thể dùng style mặc định của matplotlib
# # print(plt.style.available)  # Kiểm tra các style có sẵn
# # plt.style.use('ggplot')  # Ví dụ: dùng style 'ggplot' nếu muốn

# weights = 0
# knn = WeightedKNearestNeighbors(x=array_trainX,
#                                 y=array_trainY,
#                                 k=1000,
#                                 similarity='cosine',
#                                 weights=weights,
#                                 learning_rate=10**-1,
#                                 device=pr.device,
#                                 train_split_ratio=pr.wknn_train_split_ratio)

# def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
#     """Vẽ heatmap của confusion matrix."""
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Predicted 0', 'Predicted 1'],
#                 yticklabels=['Actual 0', 'Actual 1'])
#     plt.title(title)
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.show()

# def plot_metrics_by_threshold(metrics_list, limit_n, title="Metrics by Threshold"):
#     """Vẽ biểu đồ các metrics theo ngưỡng."""
#     plt.figure(figsize=(10, 6))
#     metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
#     for i, metric_values in enumerate(zip(*[iter(metrics_list)]*4)):
#         plt.plot(limit_n, metric_values, marker='o', label=metrics_labels[i])
#     plt.xlabel('Top N Percentile Threshold')
#     plt.ylabel('Score')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def report():
#     pred = []
#     logits = []
#     targ = []
#     for (x, y) in val_dataloader:
#         prediction = knn.predict(x, reduction="score")
#         pred.extend(prediction[0])
#         logits.extend(prediction[1])
#         targ.extend(y.tolist())
#     pred = torch.tensor(pred)
#     logits = torch.tensor(logits)
#     targ = torch.tensor(targ)

#     # Tính confusion matrix ban đầu
#     confusion_matrix = np.zeros((2, 2), int)
#     for idx, (x, y) in enumerate(zip(logits, targ)):
#         confusion_matrix[x, y] += 1
#     print("*"*os.get_terminal_size().columns)
#     metrics = metric(confusion_matrix, verbose=True)
#     print(confusion_matrix)
#     plot_confusion_matrix(confusion_matrix, title=f"Confusion Matrix (Before Training)")

#     # Tính metrics theo ngưỡng
#     limit_n = [5, 10, 25, 100]
#     res = []
#     for val in limit_n:
#         lim_score = np.percentile(pred, 100-val)
#         confusion_matrix_threshold = np.zeros((2, 2), int)
#         for idx, (x, y) in enumerate(zip(logits, targ)):
#             if pred[idx] < lim_score:
#                 continue
#             confusion_matrix_threshold[x, y] += 1
#         res.extend(metric(confusion_matrix_threshold, verbose=False)[:-1])  # Loại bỏ phần verbose nếu có
#     print(*[f"{val:.3f}" for val in res])
#     plot_metrics_by_threshold(res, limit_n, title="Metrics by Threshold (Before Training)")

#     return metrics  # Trả về metrics để sử dụng sau training nếu cần

# # Vẽ biểu đồ trước khi huấn luyện
# report()

# # Huấn luyện mô hình
# knn.train(100, 10)

# # Vẽ biểu đồ sau khi huấn luyện
# report()

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
import seaborn as sns

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

# Các tham số
seq_length = 180
latent_sizes = [3, 6, 12, 30, 60]
percentiles = [5, 10, 25, 100]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tạo bảng để lưu metrics
metrics_table = {ls: {p: {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0} for p in percentiles} for ls in latent_sizes}

# Vòng lặp qua các latent_size
for latent_size in latent_sizes:
    print(f"\nProcessing latent_size={latent_size}...")
    
    # Tải mô hình Autoencoder
    autoencoder = load_autoencoder(seq_length, latent_size, device)
    
    # Mã hóa dữ liệu huấn luyện
    encoded_trainX = encode_data(array_trainX, autoencoder, device)
    
    # Mã hóa dữ liệu kiểm tra từ val_dataloader
    val_data = []
    val_labels = []
    for x, y in val_dataloader:
        encoded_x = encode_data(x.numpy(), autoencoder, device)
        val_data.append(encoded_x)
        val_labels.append(y.numpy())
    val_data = np.concatenate(val_data, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    # Khởi tạo và huấn luyện K-NN
    knn = WeightedKNearestNeighbors(x=encoded_trainX,
                                    y=array_trainY,
                                    k=1000,
                                    similarity='cosine',
                                    weights=0,
                                    learning_rate=10**-1,
                                    device=pr.device,
                                    train_split_ratio=pr.wknn_train_split_ratio)
    knn.train(100, 10)

    # Đánh giá trên tập kiểm tra
    pred = []
    logits = []
    targ = []
    for i in range(len(val_data)):
        x = torch.tensor(val_data[i], dtype=torch.float32).unsqueeze(0)  # Thêm batch dimension
        prediction = knn.predict(x, reduction="score")
        pred.append(prediction[0][0])  # prediction[0] là score
        logits.append(prediction[1][0])  # prediction[1] là nhãn dự đoán
        targ.append(val_labels[i])

    pred = torch.tensor(pred)
    logits = torch.tensor(logits)
    targ = torch.tensor(targ)

    # Tính metrics theo các ngưỡng percentile
    for percentile in percentiles:
        lim_score = np.percentile(pred, 100 - percentile)
        confusion_matrix = np.zeros((2, 2), int)
        for idx, (x, y) in enumerate(zip(logits, targ)):
            if pred[idx] < lim_score:
                continue
            confusion_matrix[x, y] += 1
        metrics = metric(confusion_matrix, verbose=False)  # Giả định metric trả về (accuracy, precision, recall, ...)
        metrics_table[latent_size][percentile]['Accuracy'] = metrics[0]
        metrics_table[latent_size][percentile]['Precision'] = metrics[1]
        metrics_table[latent_size][percentile]['Recall'] = metrics[2]

# Tạo bảng kết quả với làm tròn đến 4 chữ số thập phân
table_data = []
for latent_size in latent_sizes:
    row = [latent_size]
    for percentile in percentiles:
        acc = metrics_table[latent_size][percentile]['Accuracy']
        prec = metrics_table[latent_size][percentile]['Precision']
        rec = metrics_table[latent_size][percentile]['Recall']
        row.extend([f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}"])
    table_data.append(row)

# Tạo DataFrame
columns = ['Latent size']
for p in percentiles:
    columns.extend([f"Top {p}% Accuracy", f"Top {p}% Precision", f"Top {p}% Recall"])
df = pd.DataFrame(table_data, columns=columns)

# Hiển thị bảng
print("\nTable 3: K-NN classifier with Autoencoder for dimensional reduction")
print(df.to_string(index=False))

# Lưu bảng vào file CSV
output_path = os.path.join(os.path.expanduser("~/Documents"), "knn", "metrics_table.csv")
df.to_csv(output_path, index=False)
print(f"\nMetrics table saved to {output_path}")