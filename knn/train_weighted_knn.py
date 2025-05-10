import logging
logging.basicConfig(level=logging.INFO)
import os
import numpy as np
import pandas as pd
import parameters as pr
import torch
from create_dataset import array_trainX, array_trainY, val_dataloader, test_dataloader, create
from models.knn import WeightedKNearestNeighbors
from utils.helper import get_file_name, metric
import matplotlib.pyplot as plt
import pickle  # Thêm import pickle để lưu mô hình

# Chuẩn bị val_data từ val_dataloader để đánh giá metrics
val_data = []
val_labels = []
for x, y in val_dataloader:
    val_data.append(x.numpy())
    val_labels.append(y.numpy())
val_data = np.concatenate(val_data, axis=0)
val_labels = np.concatenate(val_labels, axis=0)

# Chuẩn bị test_data từ test_dataloader để dự đoán giá cổ phiếu
test_data = []
test_labels = []
for x, y in test_dataloader:
    test_data.append(x.numpy())
    test_labels.append(y.numpy())
test_data = np.concatenate(test_data, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

# Khởi tạo và huấn luyện Weighted KNN trên tập train
weights = np.ones(array_trainX.shape[1])  # Khởi tạo trọng số ban đầu
knn = WeightedKNearestNeighbors(
    x=array_trainX,
    y=array_trainY,
    k=300,
    similarity='cosine',
    weights=weights,
    learning_rate=10**-1,
    device=pr.device,
    train_split_ratio=pr.wknn_train_split_ratio
)

# Lấy dữ liệu giá thực tế từ file CSV
data_path = os.path.join(os.path.expanduser("~/Documents"), "knn", "data", "^GSCP.csv")
data_df = pd.read_csv(data_path)
data_df['Date'] = pd.to_datetime(data_df['Date'], format='%Y-%m-%d', errors='coerce')
data_df = data_df[(data_df['Date'] >= pd.to_datetime(pr.start_day)) & (data_df['Date'] <= pd.to_datetime(pr.end_day))]
close_prices = data_df['Close'].str.replace(',', '').astype(float).values

# Evaluate validation
val_indices = []
for i, (x, y) in enumerate(val_dataloader):
    start_idx = len(array_trainX) + i * x.shape[0]
    end_idx = start_idx + x.shape[0]
    val_indices.extend(list(range(start_idx, end_idx)))
val_actual_prices = close_prices[val_indices]

# Predict in the test set
test_indices = []
for i, (x, y) in enumerate(test_dataloader):
    start_idx = len(array_trainX) + len(val_data) + i * x.shape[0]  # Bắt đầu từ sau train và val
    end_idx = start_idx + x.shape[0]
    test_indices.extend(list(range(start_idx, end_idx)))
test_actual_prices = close_prices[test_indices]

def report():
    # Evaluate validation
    pred_val = []
    logits_val = []
    targ_val = []
    for (x, y) in val_dataloader:
        prediction = knn.predict(x, reduction="score")
        pred_val.extend(prediction[0])
        logits_val.extend(prediction[1])
        targ_val.extend(y.tolist())
    pred_val = torch.tensor(pred_val)
    logits_val = torch.tensor(logits_val)
    targ_val = torch.tensor(targ_val)

    # Calculate metrics for whole validation set
    confusion_matrix_val = np.zeros((2, 2), int)
    for idx, (x, y) in enumerate(zip(logits_val, targ_val)):
        confusion_matrix_val[x, y] += 1
    print("*" * os.get_terminal_size().columns)
    print("Metrics on Validation Set:")
    metric(confusion_matrix_val, verbose=True)
    print(confusion_matrix_val)

    # Calculate metrics for each percentile threshold over validation set
    limit_n = [5, 10, 25, 100]
    res_val = []
    metrics_data_val = {'Top %': [], 'Accuracy': [], 'Precision': [], 'Recall': []}
    for val in limit_n:
        lim_score = np.percentile(pred_val, 100 - val)
        confusion_matrix = np.zeros((2, 2), int)
        for idx, (x, y) in enumerate(zip(logits_val, targ_val)):
            if pred_val[idx] < lim_score:
                continue
            confusion_matrix[x, y] += 1
        acc, prec, rec, _ = metric(confusion_matrix, verbose=False)
        res_val.extend([acc, prec, rec])
        metrics_data_val['Top %'].append(f"{val}%")
        metrics_data_val['Accuracy'].append(f"{acc:.2f}")
        metrics_data_val['Precision'].append(f"{prec:.2f}")
        metrics_data_val['Recall'].append(f"{rec:.2f}")

    # In kết quả trên validation
    print("\nValidation Metrics:")
    print(*[f"{val:.3f}" for val in res_val])

    # # Lưu vào file CSV cho validation
    # df_val = pd.DataFrame(metrics_data_val)
    # output_path_val = os.path.join(os.path.expanduser("~/Documents"), "knn", "metrics_table_val.csv")
    # df_val.to_csv(output_path_val, index=False)
    # print(f"\nValidation metrics table saved to {output_path_val}")

    # # Dự đoán giá cổ phiếu trên tập test
    # predicted_prices_test = []
    # current_idx = 0
    # profit_rate = np.median([(close_prices[j + 1] / close_prices[j]) - 1 for j in range(len(close_prices) - 1)])
    # for i in range(len(test_data)):
    #     x = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0)
    #     prediction = knn.predict(x, reduction="score")
    #     predicted_label = prediction[1][0]
    #     last_price = test_actual_prices[current_idx - 1] if current_idx > 0 else test_actual_prices[0]
    #     if predicted_label == 1:
    #         predicted_price = last_price * (1 + profit_rate)
    #     else:
    #         predicted_price = last_price * (1 - profit_rate)
    #     predicted_prices_test.append(predicted_price)
    #     current_idx += 1

    # # Prediction plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(test_actual_prices, label='Test Target', color='blue')
    # plt.plot(predicted_prices_test, label='Predicted Target', color='orange')
    # plt.xlabel('Day')
    # plt.ylabel('Close Price ($)')
    # plt.legend()
    # plt.grid(True)
    # output_dir = os.path.join(os.path.expanduser("~/Documents"), "knn", "prediction")
    # os.makedirs(output_dir, exist_ok=True)
    # plot_path_test = os.path.join(output_dir, "stock_price_prediction_test.png")
    # plt.savefig(plot_path_test)
    # plt.close()
    # print(f"Prediction chart for test saved to {plot_path_test}")

if __name__ == '__main__':
    knn.train(100, 10)  # 100 epochs, in progress every 10 epochs
    # Lưu mô hình sau khi huấn luyện
    output_dir = os.path.join(os.path.expanduser("~/Documents"), "knn", "models")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "weighted_knn_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)
    print(f"Model saved to {model_path}")
    report()