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

# Gọi create() để tạo dữ liệu nếu chưa có
if array_trainX is None or array_trainY is None or val_dataloader is None:
    train_dataloader, val_dataloader, array_trainX, array_trainY = create()
    if array_trainX is None:
        raise ValueError("Failed to create dataset. Please check the logs for errors.")

# Các tham số
percentiles = [5, 10, 25, 100]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Khởi tạo trọng số đặc trưng (feature weights)
# Giả sử array_trainX có shape [num_samples, seq_length], thì weights có shape [seq_length]
weights = np.ones(array_trainX.shape[1])  # Khởi tạo toàn số 1

# Khởi tạo K-NN với feature weighting
knn = WeightedKNearestNeighbors(x=array_trainX,
                                y=array_trainY,
                                k=1000,
                                similarity='cosine',
                                weights=weights,
                                learning_rate=0.1,
                                device=pr.device,
                                train_split_ratio=pr.wknn_train_split_ratio)

print("Training K-NN with feature weighting...")
# Thay vì truyền num_epochs trực tiếp, sử dụng vòng lặp để huấn luyện 100 lần
for epoch in range(100):
    # Giả định train() không cần batch_size trực tiếp, hoặc sử dụng toàn bộ dữ liệu
    knn.train()  # Gọi train() mà không truyền tham số, hoặc kiểm tra tham số hợp lệ
    if epoch % 10 == 0:  # In tiến độ mỗi 10 epoch
        print(f"Epoch {epoch}/100 completed.")
print("Training completed.")

# Đánh giá trên tập kiểm tra
pred = []
logits = []
targ = []
for x, y in val_dataloader:
    prediction = knn.predict(x, reduction="score")
    pred.extend(prediction[0])
    logits.extend(prediction[1])
    targ.extend(y.tolist())

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
    metrics = metric(confusion_matrix, verbose=False)  # Giả định metric trả về (accuracy, precision, recall, ...)
    metrics_table[percentile]['Accuracy'] = metrics[0]
    metrics_table[percentile]['Precision'] = metrics[1]
    metrics_table[percentile]['Recall'] = metrics[2]

# Tạo bảng kết quả với làm tròn đến 4 chữ số thập phân
table_data = []
for percentile in percentiles:
    acc = metrics_table[percentile]['Accuracy']
    prec = metrics_table[percentile]['Precision']
    rec = metrics_table[percentile]['Recall']
    table_data.append([f"Top {percentile}%", f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}"])

# Tạo DataFrame
columns = ['Threshold', 'Accuracy', 'Precision', 'Recall']
df = pd.DataFrame(table_data, columns=columns)

# Hiển thị bảng
print("\nTable 4: Feature weighting improves the performance of K Nearest Neighbors classifier")
print(df.to_string(index=False))

# Lưu bảng vào file CSV
output_path = os.path.join(os.path.expanduser("~/Documents"), "knn", "feature_weighting_metrics.csv")
df.to_csv(output_path, index=False)
print(f"\nMetrics table saved to {output_path}")

# In trọng số đặc trưng đã tối ưu
print("\nOptimized feature weights:")
print(knn.weights)