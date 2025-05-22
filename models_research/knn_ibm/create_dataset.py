import logging
import os
import numpy as np
import pandas as pd
import parameters as pr
from torch.utils.data import DataLoader
from utils.dataset import UpDownStockDataset
from utils.helper import *
import matplotlib.pyplot as plt

def create(*, code: str = pr.code,
            start_day: str = pr.start_day,
            end_day: str = pr.end_day,
            cols: list[str] = pr.cols,
            prediction_step: int = pr.prediction_step,
            profit_rate: float = pr.profit_rate,
            use_median: bool = pr.use_median,
            seq_length: int = pr.seq_length,
            transform = pr.transform,
            split_func = pr.split_func,
            split_ratio: list[float] = pr.split_ratio,
            batch_size: int = pr.batch_size):
    logger = logging.getLogger("CreateDataset")
    data_folder = os.path.join(os.path.expanduser("~/Documents"), "knn_ibm", "data")
    data_path = os.path.join(data_folder, "sp500.csv")  # Đường dẫn đến tệp dữ liệu
    os.makedirs(data_folder, exist_ok=True)

    # Read, load CSV
    if not os.path.isfile(data_path):
        logger.error(f"File {data_path} not found. Please ensure 'sp500.csv' is in the 'data' folder.")
        return None, None, None, None, None
    else:
        logger.info(f"Loading data from {data_path}")
        dat = pd.read_csv(data_path)

    # Xử lý cột Date
    dat['Date'] = pd.to_datetime(dat['Date'], format='%m/%d/%Y', errors='coerce')
    # Lọc dữ liệu theo khoảng thời gian
    dat = dat[(dat['Date'] >= pd.to_datetime(start_day, format='%m/%d/%Y')) & 
              (dat['Date'] <= pd.to_datetime(end_day, format='%m/%d/%Y'))]
    dat = dat.sort_values('Date')

    # Kiểm tra các cột cần thiết
    columns_to_check = ['Open', 'High', 'Low', 'Close']
    for col in columns_to_check:
        if col not in dat.columns:
            logger.error(f"Column '{col}' not found in the data file.")
            return None, None, None, None, None
    # Cột Volume đã là số nguyên, các cột giá đã là số thực, không cần chuyển đổi

    # Lấy cột dữ liệu cần thiết (mặc định là "Close")
    data = np.array(dat[cols])
    data = data.flatten() if data.shape[-1] == 1 else data
    logger.info(f"Data loaded. Shape: {data.shape}")

    # Calculate profit and define threshold
    profit = [(data[idx + prediction_step] / data[idx]) - 1 for idx in range(len(data) - prediction_step)]
    profit_rate = np.median(profit) if use_median else profit_rate
    logger.info(f"Median of profit rate: {profit_rate}")

    # Tạo dataset
    logger.info(f"Creating dataset object with parameters: {dict(seq_length=seq_length, transforms=transform.__name__ if transform is not None else None, prediction_step=prediction_step, profit_rate=profit_rate)}")
    dataset = UpDownStockDataset(data_path=data_path,
                                 seq_length=seq_length,
                                 prediction_step=prediction_step,
                                 profit_rate=float(profit_rate),
                                 transforms=transform)
    logger.info("Done")

    # Chia tập dữ liệu thành huấn luyện, kiểm tra và thử nghiệm
    logger.info(f"Creating Subset and DataLoader objects with {split_func.__name__} using split_ratio = {split_ratio} and batch_size = {batch_size}")
    
    # Bước 1: Chia thành tập huấn luyện (60%) và tập còn lại (40%)
    train_ratio = split_ratio[0]  # 0.60
    remaining_ratio = 1 - train_ratio  # 0.40
    train_set, remaining_set = split_func(dataset, [train_ratio, remaining_ratio])
    
    # Bước 2: Chia tập còn lại (40%) thành Validation (20%) và Test (20%)
    val_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])  # 0.5
    test_ratio = 1 - val_ratio  # 0.5
    val_set, test_set = split_func(remaining_set, [val_ratio, test_ratio])

    # Tạo DataLoader cho cả 3 tập
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  drop_last=False,
                                  shuffle=False)
    val_dataloader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                drop_last=False,
                                shuffle=True)
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=batch_size,
                                 drop_last=False,
                                 shuffle=True)
    logger.info("Done.")

    # Ghi log số lượng mẫu
    logger.info(f"Number of training samples: {len(train_dataloader) * batch_size}")
    logger.info(f"Number of validation samples: {len(val_dataloader) * batch_size}")
    logger.info(f"Number of test samples: {len(test_dataloader) * batch_size}")
    for x, y in train_dataloader:
        logger.info(f"X's shape: {x.shape}")
        logger.info(f"Y's shape: {y.shape}")
        break

    # Chuyển dữ liệu huấn luyện thành mảng numpy
    logger.info("Creating an array of data")
    array_trainX, array_trainY = [], []
    for val in train_dataloader:
        array_trainX.extend(list(val[0].numpy()))
        array_trainY.extend(list(val[1].numpy()))
    array_trainX, array_trainY = np.array(array_trainX), np.array(array_trainY)
    logger.info("Done.")

    # Vẽ biểu đồ dataset với phân đoạn màu
    logger.info("Plotting dataset with split visualization")
    # Tính số điểm dữ liệu gốc trong khoảng thời gian đã lọc
    total_data_points = len(data)
    # Điều chỉnh tỷ lệ dựa trên số điểm dữ liệu gốc
    train_end = int(split_ratio[0] * total_data_points)
    val_end = train_end + int(split_ratio[1] * total_data_points)

    # Tạo chỉ số cho từng tập
    train_indices = range(0, train_end)
    val_indices = range(train_end, val_end)
    test_indices = range(val_end, total_data_points)

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    plt.plot(dat['Date'].iloc[train_indices], dat['Close'].iloc[train_indices], label='Training Set (60%)', color='green', linewidth=2)
    plt.plot(dat['Date'].iloc[val_indices], dat['Close'].iloc[val_indices], label='Validation Set (20%)', color='orange', linewidth=2)
    plt.plot(dat['Date'].iloc[test_indices], dat['Close'].iloc[test_indices], label='Test Set (20%)', color='red', linewidth=2)

    # Thiết lập nhãn và tiêu đề
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.title('S&P 500 Stock Price Dataset with Train-Validation-Test Split')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Điều chỉnh layout
    plt.tight_layout()

    # Lưu biểu đồ
    output_dir = os.path.join(os.path.expanduser("~/Documents"), "knn_ibm", "prediction")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "sp500_split_data_plot.png")
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Split dataset plot saved to {plot_path}")

    return train_dataloader, val_dataloader, test_dataloader, array_trainX, array_trainY

# Gọi hàm để tạo dataset
train_dataloader, val_dataloader, test_dataloader, array_trainX, array_trainY = create()