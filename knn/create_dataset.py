# import logging
# import os

# import numpy as np
# import pandas as pd
# import parameters as pr
# import yfinance
# from torch.utils.data import DataLoader
# from utils.dataset import UpDownStockDataset
# from utils.helper import *


# def create(*, code: str = pr.code,
#             start_day: str = pr.start_day,
#             end_day: str = pr.end_day,
#             cols: list[str] = pr.cols,
#             prediction_step: int = pr.prediction_step,
#             profit_rate: float = pr.profit_rate,
#             use_median: bool = pr.use_median,
#             seq_length: int = pr.seq_length,
#             transform = pr.transform,
#             split_func = pr.split_func,
#             split_ratio: list[int] = pr.split_ratio,
#             batch_size: int = pr.batch_size):
#     logger = logging.getLogger("CreateDataset")
#     data_folder = "./data"
#     data_path = os.path.join(data_folder, f"{code}.csv")
#     os.makedirs(data_folder, exist_ok=True)
#     if not os.path.isfile(data_path):
#         logger.info(f"Downloading {code} data from yfinance...")
#         dat = yfinance.download(code, start=start_day, end=end_day, progress=False)
#         dat.to_csv(data_path)
#     else:
#         logger.info(f"Loading data from {data_path}")
#         dat = pd.read_csv(data_path)
#     data = np.array(dat[cols])
#     data = data.flatten() if data.shape[-1] == 1 else data
#     logger.info(f"Data downloaded.")

#     profit = [(data[idx+prediction_step]/data[idx]) - 1 for idx in range(len(data) - prediction_step)]
#     profit_rate = np.median(profit) if use_median else profit_rate
#     logger.info(f"Median of profit rate: {profit_rate}")

#     logger.info(f"Create dataset object with parameters: {dict(seq_length=seq_length, trainsforms=transform.__name__ if transform is not None else None, prediction_step=prediction_step, profit_rate=profit_rate)}")
#     dataset = UpDownStockDataset(data=data,
#                                     seq_length=seq_length,
#                                     transforms=transform,
#                                     prediction_step=prediction_step,
#                                     profit_rate = float(profit_rate))
#     logger.info("Done")

#     logger.info(f"Create Subset and DataLoader objects with {split_func.__name__} using split_ratio = {split_ratio} and batch_size = {batch_size}")
#     train_set, val_set = split_func(dataset, split_ratio)


#     train_dataloader = DataLoader(dataset=train_set,
#                                     batch_size=batch_size,
#                                     drop_last=False,
#                                     shuffle=False)
#     val_dataloader = DataLoader(dataset=val_set,
#                                 batch_size=batch_size,
#                                 drop_last=False,
#                                 shuffle=True)
#     logger.info("Done.")

#     logger.info(f"Number of training samples: {len(train_dataloader) * batch_size}")
#     logger.info(f"Number of validating samples: {len(val_dataloader) * batch_size}")
#     for x, y in train_dataloader:
#         logger.info(f"X's shape: {x.shape}")
#         logger.info(f"Y's shape: {y.shape}")
#         break


#     logger.info("Creating an array of data")
#     array_trainX, array_trainY = [], []
#     for val in train_dataloader:
#         array_trainX.extend(list(val[0].numpy()))
#         array_trainY.extend(list(val[1].numpy()))
#     array_trainX, array_trainY = np.array(array_trainX), np.array(array_trainY)
#     logger.info("Done.")
#     return train_dataloader, val_dataloader, array_trainX, array_trainY

# train_dataloader, val_dataloader, array_trainX, array_trainY = create()

import logging
import os

import numpy as np
import pandas as pd
import parameters as pr
from torch.utils.data import DataLoader
from utils.dataset import UpDownStockDataset
from utils.helper import *

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
            split_ratio: list[int] = pr.split_ratio,
            batch_size: int = pr.batch_size):
    logger = logging.getLogger("CreateDataset")
    data_folder = "./data"
    data_path = os.path.join(data_folder, "^GSCP.csv")  # Đường dẫn đến tệp dữ liệu
    os.makedirs(data_folder, exist_ok=True)

    # Kiểm tra và tải dữ liệu từ tệp CSV
    if not os.path.isfile(data_path):
        logger.error(f"File {data_path} not found. Please ensure '^GSCP.csv' is in the 'data' folder.")
        return None, None, None, None
    else:
        logger.info(f"Loading data from {data_path}")
        dat = pd.read_csv(data_path)

    # Xử lý định dạng ngày một cách linh hoạt
    try:
        # Thử với định dạng YYYY-MM-DD trước (dựa trên lỗi)
        dat['Date'] = pd.to_datetime(dat['Date'], format='%Y-%m-%d', errors='coerce')
        if dat['Date'].isna().all():
            # Nếu không thành công, thử với định dạng DD-MMM-YY
            dat['Date'] = pd.to_datetime(dat['Date'], format='%d-%b-%y', errors='coerce')
            if dat['Date'].isna().all():
                logger.error("Unable to parse 'Date' column with either %Y-%m-%d or %d-%b-%y formats.")
                return None, None, None, None
            else:
                dat['Date'] = dat['Date'].dt.strftime('%Y-%m-%d')
                dat['Date'] = pd.to_datetime(dat['Date'])
                logger.info("Date format converted from DD-MMM-YY to YYYY-MM-DD.")
        else:
            logger.info("Date format is already YYYY-MM-DD or compatible.")
    except Exception as e:
        logger.error(f"Error parsing 'Date' column: {e}")
        return None, None, None, None

    # Lọc dữ liệu theo khoảng thời gian
    dat = dat[(dat['Date'] >= start_day) & (dat['Date'] <= end_day)]
    if dat.empty:
        logger.error(f"No data found in the specified date range ({start_day} to {end_day}).")
        return None, None, None, None

    # Xử lý các cột số
    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in columns_to_convert:
        if col not in dat.columns:
            logger.error(f"Column '{col}' not found in the data file.")
            return None, None, None, None
        dat[col] = dat[col].str.replace(',', '').astype(float)

    # Lấy cột dữ liệu cần thiết (mặc định là "Close")
    data = np.array(dat[cols])
    data = data.flatten() if data.shape[-1] == 1 else data
    logger.info(f"Data loaded. Shape: {data.shape}")

    # Tính toán lợi nhuận và xác định ngưỡng
    profit = [(data[idx + prediction_step] / data[idx]) - 1 for idx in range(len(data) - prediction_step)]
    profit_rate = np.median(profit) if use_median else profit_rate
    logger.info(f"Median of profit rate: {profit_rate}")

    # Tạo dataset
    logger.info(f"Creating dataset object with parameters: {dict(seq_length=seq_length, transforms=transform.__name__ if transform is not None else None, prediction_step=prediction_step, profit_rate=profit_rate)}")
    dataset = UpDownStockDataset(data=data,
                                 seq_length=seq_length,
                                 transforms=transform,
                                 prediction_step=prediction_step,
                                 profit_rate=float(profit_rate))
    logger.info("Done")

    # Chia tập dữ liệu thành huấn luyện và kiểm tra
    logger.info(f"Creating Subset and DataLoader objects with {split_func.__name__} using split_ratio = {split_ratio} and batch_size = {batch_size}")
    train_set, val_set = split_func(dataset, split_ratio)

    # Tạo DataLoader
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  drop_last=False,
                                  shuffle=False)
    val_dataloader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                drop_last=False,
                                shuffle=True)
    logger.info("Done.")

    # Ghi log số lượng mẫu
    logger.info(f"Number of training samples: {len(train_dataloader) * batch_size}")
    logger.info(f"Number of validating samples: {len(val_dataloader) * batch_size}")
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
    return train_dataloader, val_dataloader, array_trainX, array_trainY

# Gọi hàm để tạo dataset
train_dataloader, val_dataloader, array_trainX, array_trainY = create()