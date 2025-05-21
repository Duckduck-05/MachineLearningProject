import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.config import (
    TICKERS, RFC_PARAMS, 
    get_model_path, get_raw_data_path
)
from app.utils.tech_indicators import calculate_technical_indicators
from .models.rfc import StockPriceRFC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RFC_Training")

def create_target_labels(df, window=30):
    smooth = df['Close'].ewm(span=window, adjust=False).mean()
    target = np.sign(smooth.shift(-window) - smooth)
    target.name = f'Target_{window}'
    return target

def train_rfc_model(ticker_key: str, window: int = None):
    if window is None:
        window = RFC_PARAMS["prediction_window"]
        
    logger.info(f"Training RFC model for {ticker_key} with {window}-day window")
    
    # Load data
    data_path = get_raw_data_path(ticker_key)
    if not os.path.isfile(data_path):
        logger.error(f"File {data_path} not found for {ticker_key}")
        return None
    
    # Read the CSV file
    df = pd.read_csv(data_path, parse_dates=['Date'], skiprows=[1])
    df.set_index('Date', inplace=True)
    
    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in columns_to_convert:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(',', '').astype(float)
            else:
                df[col] = df[col].astype(float)
    
    # Calculate technical indicators
    features_df = calculate_technical_indicators(df)
    
    # Create target variable
    target = create_target_labels(df, window)
    
    # Combine features and target
    full_df = pd.concat([features_df, target], axis=1)
    full_df.dropna(inplace=True)
    
    logger.info(f"Created dataset with {len(full_df)} samples and {len(features_df.columns)} features")
    
    # Split into features and target
    X = full_df.drop(columns=[f'Target_{window}'])
    y = full_df[f'Target_{window}']
    
    # Train-test split with time series consideration (test is most recent 20%)
    test_size = int(0.2 * len(X))
    train_idx = X.index[:-test_size]
    test_idx = X.index[-test_size:]
    
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    
    # Create and train model
    model = StockPriceRFC(
        n_estimators=RFC_PARAMS["n_estimators"],
        max_depth=RFC_PARAMS["max_depth"],
        max_features=RFC_PARAMS["max_features"],
        random_state=RFC_PARAMS["random_state"]
    )
    
    logger.info("Training RFC model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Model evaluation metrics:")
    logger.info(f"- Accuracy: {accuracy:.4f}")
    logger.info(f"- Precision: {precision:.4f}")
    logger.info(f"- Recall: {recall:.4f}")
    logger.info(f"- F1 Score: {f1:.4f}")
    logger.info(f"- Confusion Matrix:\n{conf_matrix}")
    
    # Important features
    feature_importance = pd.Series(model.feature_importance_, index=X.columns).sort_values(ascending=False)
    logger.info(f"Top 5 important features: {feature_importance.head(5)}")
    
    # Save model
    model_path = get_model_path(ticker_key, "random_forest", "classification")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Retrain on full dataset for deployment
    logger.info("Retraining on full dataset for deployment...")
    final_model = StockPriceRFC(
        n_estimators=RFC_PARAMS["n_estimators"],
        max_depth=RFC_PARAMS["max_depth"],
        max_features=RFC_PARAMS["max_features"],
        random_state=RFC_PARAMS["random_state"]
    )
    final_model.fit(X, y)
    
    # Save final model
    final_model.save(model_path)
    logger.info(f"Final model saved to {model_path}")
    
    return final_model

def train_all_rfc_models(windows=None):
    if windows is None:
        windows = [RFC_PARAMS["prediction_window"]]
        
    for ticker_key in TICKERS:
        for window in windows:
            train_rfc_model(ticker_key, window)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting RFC model training for all tickers")
    train_all_rfc_models()