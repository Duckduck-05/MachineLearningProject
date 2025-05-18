import logging
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import (
    TICKERS, KNN_PARAMS, get_knn_model_path, get_model_path,
    get_raw_data_path, get_processed_data_path
)
from .utils.dataset import UpDownStockDataset
from .utils.helper import standardize, min_max_scale, random_split, sequential_split, metric
from .knn import WeightedKNearestNeighbors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KNN_Training")

def create_knn_dataset(ticker_key: str):
    """Create dataset for KNN model training"""
    logger.info(f"Creating KNN dataset for {ticker_key}")
    
    # Get parameters
    seq_length = KNN_PARAMS["seq_length"]
    prediction_step = KNN_PARAMS["prediction_step"]
    profit_rate = KNN_PARAMS["profit_rate"]
    use_median = KNN_PARAMS["use_median"]
    transform_name = KNN_PARAMS["transform"]
    split_ratio = KNN_PARAMS["split_ratio"]
    batch_size = KNN_PARAMS["batch_size"]
    
    # Load data
    data_path = get_raw_data_path(ticker_key)
    if not os.path.isfile(data_path):
        logger.error(f"File {data_path} not found for {ticker_key}")
        return None, None, None, None, None
        
    # Read the CSV file
    dat = pd.read_csv(data_path, parse_dates=['Date'], skiprows=[1])
    
    # Convert price columns to float
    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in columns_to_convert:
        if col in dat.columns:
            if dat[col].dtype == object:  # If column contains strings
                dat[col] = dat[col].str.replace(',', '').astype(float)
            else:  # If column is already numeric
                dat[col] = dat[col].astype(float)
    
    # Use Close price for the dataset
    data = np.array(dat['Close'])
    data = data.flatten() if data.shape[-1] == 1 else data
    logger.info(f"Data loaded. Shape: {data.shape}")
    
    # Calculate profit and define threshold
    profit = [(data[idx + prediction_step] / data[idx]) - 1 for idx in range(len(data) - prediction_step)]
    profit_rate = np.median(profit) if use_median else profit_rate
    logger.info(f"Median of profit rate: {profit_rate}")
    
    # Select transform function
    transform_func = standardize if transform_name == "standardize" else min_max_scale
    
    # Create dataset
    dataset = UpDownStockDataset(
        data=data,
        seq_length=seq_length,
        transforms=transform_func,
        prediction_step=prediction_step,
        profit_rate=float(profit_rate)
    )
    logger.info("Dataset created")
    
    # Split data
    split_func = random_split if "random" in KNN_PARAMS.get("split_func", "random_split") else sequential_split
    
    # Step 1: Split into train (60%) and remaining (40%)
    train_ratio = split_ratio[0]
    remaining_ratio = 1 - train_ratio
    train_set, remaining_set = split_func(dataset, [train_ratio, remaining_ratio])
    
    # Step 2: Split remaining into validation (20%) and test (20%)
    val_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
    test_ratio = 1 - val_ratio
    val_set, test_set = split_func(remaining_set, [val_ratio, test_ratio])
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False
    )
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )
    
    # Convert training data to numpy arrays
    logger.info("Creating arrays for training")
    array_trainX, array_trainY = [], []
    for val in train_dataloader:
        array_trainX.extend(list(val[0].numpy()))
        array_trainY.extend(list(val[1].numpy()))
    array_trainX, array_trainY = np.array(array_trainX), np.array(array_trainY)
    logger.info("Arrays created")
    
    return train_dataloader, val_dataloader, test_dataloader, array_trainX, array_trainY

def train_knn_model(ticker_key: str, problem_type: str = "classification", epochs: int = 100):
    """Train KNN model for a specified ticker"""
    logger.info(f"Training {problem_type} KNN model for {ticker_key}")
    
    # Create dataset
    train_dataloader, val_dataloader, test_dataloader, array_trainX, array_trainY = create_knn_dataset(ticker_key)
    
    if array_trainX is None or array_trainY is None:
        logger.error(f"Failed to create dataset for {ticker_key}")
        return None
    
    # Initialize KNN model
    weights = np.ones(array_trainX.shape[1])
    knn = WeightedKNearestNeighbors(
        x=array_trainX,
        y=array_trainY,
        k=KNN_PARAMS["k"],
        similarity=KNN_PARAMS["similarity"],
        weights=weights,
        learning_rate=KNN_PARAMS["learning_rate"],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        train_split_ratio=KNN_PARAMS["wknn_train_split_ratio"]
    )
    
    # Train model
    logger.info(f"Starting KNN training for {ticker_key} with {epochs} epochs")
    knn.train(epochs=epochs, batch_size=KNN_PARAMS["batch_size"])
    logger.info(f"KNN training completed for {ticker_key}")
    
    # Evaluate on validation set
    logger.info(f"Evaluating KNN model for {ticker_key} on validation set")
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
    
    # Calculate metrics for validation set
    confusion_matrix_val = np.zeros((2, 2), int)
    for idx, (x, y) in enumerate(zip(logits_val, targ_val)):
        confusion_matrix_val[x, y] += 1
        
    logger.info("Metrics on Validation Set:")
    accuracy, precision, recall, f1_score = metric(confusion_matrix_val, verbose=True)
    logger.info(f"Confusion matrix: {confusion_matrix_val}")
    
    # Save model
    model_path = get_model_path(ticker_key, "knn", problem_type)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)
        
    logger.info(f"KNN model saved to {model_path}")
    
    # If we're training a regression model, also create and save a regression prediction demo
    if problem_type == "regression":
        logger.info("Creating regression prediction demo for test set")
        create_regression_prediction_demo(knn, test_dataloader, ticker_key)
    
    return knn

def create_regression_prediction_demo(knn_model, test_dataloader, ticker_key):
    """Create a demo of regression predictions using the KNN model"""
    # Get test data
    test_data = []
    test_labels = []
    for x, y in test_dataloader:
        test_data.append(x.numpy())
        test_labels.append(y.numpy())
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Load actual prices from the raw data
    data_path = get_raw_data_path(ticker_key)
    data_df = pd.read_csv(data_path, parse_dates=['Date'], skiprows=[1])
    
    # Convert Close to float if needed
    if data_df['Close'].dtype == object:
        data_df['Close'] = data_df['Close'].str.replace(',', '').astype(float)
    
    # Use a portion of test data to simulate regression prediction
    predicted_prices = []
    initial_price = data_df['Close'].iloc[-len(test_data)]  # Start from the first test point
    predicted_prices.append(initial_price)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Make predictions
    for i in range(len(test_data)):
        x = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0)
        prediction = knn_model.predict(x, reduction="score")
        predicted_label = prediction[1][0]  # 0: down, 1: up
        
        # Determine price change based on prediction
        if predicted_label == 1:
            pct_change = 0.005  # 0.5% increase
        else:
            pct_change = -0.005  # 0.5% decrease
        
        # Add some noise
        noise = np.random.normal(0, 0.002)
        pct_change += noise
        
        # Calculate next price
        next_price = predicted_prices[-1] * (1 + pct_change)
        predicted_prices.append(next_price)
    
    # Save predictions to CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    predictions_df = pd.DataFrame({
        'Day': range(len(predicted_prices)),
        'Predicted_Price': predicted_prices
    })
    
    predictions_path = os.path.join(results_dir, f"{ticker_key}_knn_regression_demo.csv")
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Regression predictions demo saved to {predictions_path}")

def train_all_knn_models(epochs: int = 100):
    """Train KNN models for all tickers"""
    for ticker_key in TICKERS:
        # Train classification model
        train_knn_model(ticker_key, "classification", epochs)
        
        # Train regression model
        train_knn_model(ticker_key, "regression", epochs)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting KNN model training for all tickers")
    train_all_knn_models()