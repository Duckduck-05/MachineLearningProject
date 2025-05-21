# app/data_processing.py
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

from .config import (
    GRU_INPUT_FEATURE_COLUMNS, GRU_TARGET_COLUMN, LSTM_TARGET_COLUMN, TICKERS, get_gru_scaler_path, get_raw_data_path, get_processed_data_path,
    ENSEMBLE_TARGET_COLUMN, LAG_DAYS,
    LSTM_SEQUENCE_LENGTH, LSTM_INPUT_FEATURE_COLUMNS, get_lstm_scaler_path
)

def create_ensemble_features(df_input: pd.DataFrame) -> pd.DataFrame:
    target_col = 'Close'
    df = df_input.copy()
    df_features = df[[target_col]].copy()

    # 1. Create Lag Features
    for i in range(1, LAG_DAYS + 1):
        feature_name = f'{target_col}_lag_{i}'
        df_features[feature_name] = df_features[target_col].shift(i)

    # 2. Create Percentage Change Feature
    # (Today's target_col - Yesterday's target_col) / Yesterday's target_col
    df_features['Close_pct_change_1d'] = df_features[target_col].pct_change(periods=1).shift(1)

    # 3. Create 'close_target' (next day's target_col value)
    df_features['close_target'] = df_features[target_col].shift(-1)
    df_features['target_pct_change'] = (df_features['close_target'] - df_features['Close']) / df_features['Close']

    # 4. Drop rows with NaNs created by shifting (for lags, pct_change, and close_target)
    df_features.dropna(inplace=True)
  
    return df_features

def prepare_and_scale_for_lstm(df_input: pd.DataFrame, ticker_key: str,
                               feature_cols: List[str], target_col_for_lstm: str) -> tuple[pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    print(f"Preparing and scaling LSTM data for {ticker_key}...")
    df = df_input.copy()
    
    # 1. Select and scale input features for LSTM
    features_to_scale_df = df[feature_cols].copy()
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_feature_values = feature_scaler.fit_transform(features_to_scale_df)
    
    # 2. Select and scale the target column for LSTM
    target_to_scale_series = df[[target_col_for_lstm]].copy()
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target_values = target_scaler.fit_transform(target_to_scale_series)

    # 3. Save scalers
    feature_scaler_path = get_lstm_scaler_path(ticker_key, "feature")
    target_scaler_path = get_lstm_scaler_path(ticker_key, "target")

    with open(feature_scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"LSTM scalers for {ticker_key} saved to {os.path.dirname(feature_scaler_path)}.")

    # 4. Create a new DataFrame with scaled values
    scaled_df = pd.DataFrame(scaled_feature_values, columns=feature_cols, index=df.index)
    scaled_df[f'scaled_{target_col_for_lstm}_target'] = np.roll(scaled_target_values.flatten(), -1)
    scaled_df = scaled_df.iloc[:-1]

    return scaled_df, feature_scaler, target_scaler

def prepare_and_scale_for_gru(df_input: pd.DataFrame, ticker_key: str,
                              feature_cols: List[str], target_col_for_lstm: str) -> tuple[pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    print(f"Preparing and scaling GRU data for {ticker_key}...")
    df = df_input.copy()
    
    # 1. Select and scale input features for GRU
    features_to_scale_df = df[feature_cols].copy()
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_feature_values = feature_scaler.fit_transform(features_to_scale_df)
    
    # 2. Select and scale the target column for GRU
    target_to_scale_series = df[[target_col_for_lstm]].copy()
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target_values = target_scaler.fit_transform(target_to_scale_series)

    # 3. Save scalers
    feature_scaler_path = get_gru_scaler_path(ticker_key, "feature")
    target_scaler_path = get_gru_scaler_path(ticker_key, "target")

    with open(feature_scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"GRU scalers for {ticker_key} saved to {os.path.dirname(feature_scaler_path)}.")

    # 4. Create a new DataFrame with scaled values
    scaled_df = pd.DataFrame(scaled_feature_values, columns=feature_cols, index=df.index)
    scaled_df[f'scaled_{target_col_for_lstm}_target'] = np.roll(scaled_target_values.flatten(), -1)
    scaled_df = scaled_df.iloc[:-1]

    return scaled_df, feature_scaler, target_scaler

def run_processing_for_all_tickers():
    for ticker_key in TICKERS:
        print(f"\n--- Processing data for {ticker_key} ---")
        
        raw_file_path = get_raw_data_path(ticker_key)
        df_raw = pd.read_csv(raw_file_path, skiprows=[1], parse_dates=['Date'])
        df_raw.set_index('Date', inplace=True)
        df_raw.sort_index(inplace=True)
        
        df_to_process = df_raw.copy()

        # 1. Process for ensemble Models (XGBoost, RandomForest)
        print(f"Processing features for {ticker_key}...")
        df_ensemble_processed = create_ensemble_features(df_to_process)
        processed_file_path = get_processed_data_path(ticker_key)
        df_ensemble_processed.to_csv(processed_file_path, index=True)

        # 2. Process for LSTM and GRU Model (Scaling and saving scalers)
        print(f"Preparing and scaling data for LSTM for {ticker_key}...")
        scaled_lstm_df, feature_scaler, target_scaler = prepare_and_scale_for_lstm(
            df_to_process,
            ticker_key,
            feature_cols=LSTM_INPUT_FEATURE_COLUMNS,
            target_col_for_lstm=LSTM_TARGET_COLUMN
        )
        print(f"LSTM data preparation (scaling and scaler saving) done for {ticker_key}.")

        print(f"Preparing and scaling data for GRU for {ticker_key}...")
        scaled_gru_df, feature_scaler, target_scaler = prepare_and_scale_for_gru(
            df_to_process,
            ticker_key,
            feature_cols=GRU_INPUT_FEATURE_COLUMNS,
            target_col_for_lstm=GRU_TARGET_COLUMN
        )
        print(f"GRU data preparation (scaling and scaler saving) done for {ticker_key}.")

if __name__ == "__main__":
    run_processing_for_all_tickers()