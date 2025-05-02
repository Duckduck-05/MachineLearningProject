import logging
logging.basicConfig(level=logging.INFO)
import os
import numpy as np
import pandas as pd
import parameters as pr
import torch
from create_dataset import array_trainX, array_trainY, val_dataloader, create
from models.knn import WeightedKNearestNeighbors
from utils.helper import get_file_name, metric
from sklearn.decomposition import PCA
import pickle

# Gọi create() để tạo dữ liệu nếu chưa có
if array_trainX is None or array_trainY is None or val_dataloader is None:
    train_dataloader, val_dataloader, array_trainX, array_trainY = create()
    if array_trainX is None:
        raise ValueError("Failed to create dataset. Please check the logs for errors.")

# Các tham số
n_components_list = [3, 6, 12, 30, 60]  # Tương tự latent_size trong Autoencoder
percentiles = [5, 10, 25, 100]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tạo thư mục để lưu PCA (nếu cần)
pca_save_dir = os.path.join(os.path.expanduser("~/Documents"), "knn", "models", "pca")
os.makedirs(pca_save_dir, exist_ok=True)

# Tạo bảng để lưu metrics
metrics_table = {n: {p: {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0} for p in percentiles} for n in n_components_list}

# Vòng lặp qua các số lượng components (n_components)
for n_components in n_components_list:
    print(f"\nProcessing n_components={n_components}...")

    # Áp dụng PCA để giảm chiều dữ liệu huấn luyện
    pca = PCA(n_components=n_components)
    pca.fit(array_trainX)  # Huấn luyện PCA trên dữ liệu huấn luyện
    encoded_trainX = pca.transform(array_trainX)  # Giảm chiều dữ liệu huấn luyện

    # Giảm chiều dữ liệu kiểm tra từ val_dataloader
    val_data = []
    val_labels = []
    for x, y in val_dataloader:
        encoded_x = pca.transform(x.numpy())  # Giảm chiều dữ liệu kiểm tra
        val_data.append(encoded_x)
        val_labels.append(y.numpy())
    val_data = np.concatenate(val_data, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    # Lưu PCA để tái sử dụng (nếu cần)
    pca_save_path = os.path.join(pca_save_dir, f"pca_n_components_{n_components}.pkl")
    with open(pca_save_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA model saved at {pca_save_path}")

    # Khởi tạo weights mặc định (vector toàn số 1) để tránh lỗi
    weights = np.ones(encoded_trainX.shape[1])  # Kích thước weights phù hợp với dữ liệu đã giảm chiều

    # Khởi tạo và huấn luyện K-NN
    knn = WeightedKNearestNeighbors(x=encoded_trainX,
                                    y=array_trainY,
                                    k=1000,
                                    similarity='cosine',
                                    weights=weights,  # Truyền weights mặc định
                                    learning_rate=0.1,
                                    device=pr.device,
                                    train_split_ratio=pr.wknn_train_split_ratio)

    print(f"Training K-NN with n_components={n_components}...")
    for epoch in range(100):  # Huấn luyện 100 lần (giả định train() không nhận num_epochs)
        knn.train()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/100 completed.")
    print("Training completed.")

    # Đánh giá trên tập kiểm tra
    pred = []
    logits = []
    targ = []
    for i in range(len(val_data)):
        x = torch.tensor(val_data[i], dtype=torch.float32).unsqueeze(0)  # Thêm batch dimension
        prediction = knn.predict(x, reduction="score")
        pred.append(prediction[0][0])
        logits.append(prediction[1][0])
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
        metrics_table[n_components][percentile]['Accuracy'] = metrics[0]
        metrics_table[n_components][percentile]['Precision'] = metrics[1]
        metrics_table[n_components][percentile]['Recall'] = metrics[2]

# Tạo bảng kết quả với làm tròn đến 4 chữ số thập phân
table_data = []
for n_components in n_components_list:
    row = [n_components]
    for percentile in percentiles:
        acc = metrics_table[n_components][percentile]['Accuracy']
        prec = metrics_table[n_components][percentile]['Precision']
        rec = metrics_table[n_components][percentile]['Recall']
        row.extend([f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}"])
    table_data.append(row)

# Tạo DataFrame
columns = ['n_components']
for p in percentiles:
    columns.extend([f"Top {p}% Accuracy", f"Top {p}% Precision", f"Top {p}% Recall"])
df = pd.DataFrame(table_data, columns=columns)

# Hiển thị bảng
print("\nTable: K-NN classifier with PCA for dimensional reduction")
print(df.to_string(index=False))

# Lưu bảng vào file CSV
output_path = os.path.join(os.path.expanduser("~/Documents"), "knn", "pca_metrics_table.csv")
df.to_csv(output_path, index=False)
print(f"\nMetrics table saved to {output_path}")