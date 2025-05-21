# app/config.py
import os
from typing import List, Dict

APP_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ticker details
TICKERS = ['^GSPC', 'IBM']

# Data path
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data")
TICKER_RAW_DATA_PATH = os.path.join(DATA_PATH, "{ticker_key}_raw_data.csv")
TICKER_PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "{ticker_key}_processed_data.csv")

def get_raw_data_path(ticker_key: str) -> str:
    return TICKER_RAW_DATA_PATH.format(ticker_key=ticker_key.lower())

def get_processed_data_path(ticker_key: str) -> str:
    return TICKER_PROCESSED_DATA_PATH.format(ticker_key=ticker_key.lower())

# Model path
MODELS_STORE_DIR = os.path.join(PROJECT_ROOT_DIR, "models_store")

KNN_MODEL_NAME_TEMPLATE = "{ticker_key}_knn_{problem_type}.pkl"
# KNN_SCALER_NAME_TEMPLATE = "{ticker_key}_knn_scaler_x.pkl"
XGBOOST_MODEL_NAME_TEMPLATE = "{ticker_key}_xgboost_{problem_type}.xgb"
RANDOM_FOREST_MODEL_NAME_TEMPLATE = "{ticker_key}_random_forest_{problem_type}.pkl"
LSTM_MODEL_NAME_TEMPLATE = "{ticker_key}_lstm_{problem_type}.pth"
LSTM_SCALER_NAME_TEMPLATE = "{ticker_key}_lstm_{scaler_type}_scaler_{problem_type}.pkl"
GRU_MODEL_NAME_TEMPLATE = "{ticker_key}_gru_{problem_type}.pth"
GRU_SCALER_NAME_TEMPLATE = "{ticker_key}_gru_{scaler_type}_scaler_{problem_type}.pkl"
# Problem types are "classification" or "regression"

def get_model_path(ticker_key: str, model_type: str, problem_type: str) -> str:
    if ticker_key == "^GSPC":
        ticker_key = "GSPC"
    if model_type == "xgboost":
        model_filename = XGBOOST_MODEL_NAME_TEMPLATE.format(
            ticker_key=ticker_key, problem_type=problem_type)
    elif model_type == "random_forest":
        model_filename = RANDOM_FOREST_MODEL_NAME_TEMPLATE.format(
            ticker_key=ticker_key, problem_type=problem_type)
    elif model_type == "lstm":
        model_filename = LSTM_MODEL_NAME_TEMPLATE.format(
            ticker_key=ticker_key, problem_type=problem_type)
    elif model_type == "knn":
        model_filename = KNN_MODEL_NAME_TEMPLATE.format(
            ticker_key=ticker_key, problem_type=problem_type)
    elif model_type == "gru":
        model_filename = GRU_MODEL_NAME_TEMPLATE.format(
            ticker_key=ticker_key, problem_type=problem_type)
    return os.path.join(MODELS_STORE_DIR, model_filename)

# Database path
DATABASE_NAME = "stock_database.sqlite"
DATABASE_DIR = os.path.join(PROJECT_ROOT_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)

# Problem Settings
# Classification settings
TARGET_COLUMN_CLASSIFICATION = 'Price_Direction'

ENSEMBLE_TARGET_COLUMN = 'Close'
TARGET_COLUMN = 'Close'
LAG_DAYS = 14
EXPECTED_FEATURES_ENSEMBLE: List[str] = [
    f'Close_lag_{i}' for i in range(1, LAG_DAYS + 1)
] + [f'Close_pct_change_1d']
LSTM_TARGET_COLUMN = 'Close'
GRU_TARGET_COLUMN = 'Close'

# LSTM Settings
LSTM_SEQUENCE_LENGTH = 20
LSTM_NUM_FEATURES = 4
LSTM_INPUT_FEATURE_COLUMNS = ['Open', 'High', 'Low', LSTM_TARGET_COLUMN]

def get_lstm_scaler_path(ticker_key: str, scaler_name: str) -> str:
    if ticker_key == "^GSPC": ticker_key = "GSPC"
    filename = f"{ticker_key}_lstm_{scaler_name}.pkl"
    scaler_dir = os.path.join(MODELS_STORE_DIR, "lstm_scalers")
    return os.path.join(scaler_dir, filename)

# GRU Settings
GRU_SEQUENCE_LENGTH = 20
GRU_NUM_FEATURES = 4
GRU_INPUT_FEATURE_COLUMNS = ['Open', 'High', 'Low', GRU_TARGET_COLUMN]

def get_gru_scaler_path(ticker_key: str, scaler_name: str) -> str:
    if ticker_key == "^GSPC": ticker_key = "GSPC"
    filename = f"{ticker_key}_gru_{scaler_name}.pkl"
    scaler_dir = os.path.join(MODELS_STORE_DIR, "gru_scalers")
    return os.path.join(scaler_dir, filename)

# KNN Settings
KNN_PARAMS = {
    "k": 300,  # Number of neighbors
    "similarity": "cosine",  # Similarity metric: "euclid", "cosine", or "manhattan"
    "seq_length": 10,  # Sequence length for input
    "prediction_step": 1,  # Number of steps to predict ahead
    "profit_rate": 0.03,  # Profit rate threshold for classification
    "use_median": True,  # Whether to use median profit rate
    "transform": "standardize",  # Transformation method: "standardize" or "min_max_scale"
    "split_ratio": [0.60, 0.20, 0.20],  # Train/val/test split ratio
    "batch_size": 64,  # Batch size for training
    "learning_rate": 10**-1,  # Learning rate for weighted KNN
    "wknn_train_split_ratio": 0.8  # Train split ratio for weighted KNN
}

# Scheduler Settings
SCHEDULER_TIMEZONE = 'America/New_York'
DATA_UPDATE_HOUR_ET, DATA_UPDATE_MINUTE_ET = 17, 15
PREDICTION_HOUR_ET, PREDICTION_MINUTE_ET = 17, 45
FASTAPI_BASE_URL_DEFAULT = "http://localhost:8000"
FASTAPI_URL= os.getenv("FASTAPI_URL", FASTAPI_BASE_URL_DEFAULT)

# Scheduler Settings for Testing
SCHEDULER_TIMEZONE_TEST = 'Asia/Ho_Chi_Minh'
DATA_UPDATE_HOUR_ET_TEST, DATA_UPDATE_MINUTE_ET_TEST = 19, 5 
PREDICTION_HOUR_ET_TEST, PREDICTION_MINUTE_ET_TEST = 19, 8

# RFC Settings
RFC_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "max_features": 4,
    "random_state": 36,
    "prediction_window": 30
}
