# app/historical_seeder.py
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys
import numpy as np
from typing import List

from .config import (
    TICKERS, ENSEMBLE_TARGET_COLUMN, LSTM_TARGET_COLUMN,
    LAG_DAYS, EXPECTED_FEATURES_ENSEMBLE,
    LSTM_SEQUENCE_LENGTH, LSTM_NUM_FEATURES, LSTM_INPUT_FEATURE_COLUMNS,
    get_raw_data_path, get_model_path, get_lstm_scaler_path
)
from .model_utils import (
    load_model,
    prepare_features_for_ensemble_model,
    prepare_input_sequence_for_lstm,
    make_prediction
)
from .db_utils import init_db, save_actual_prices, save_prediction, update_actual_price_for_prediction

from sklearn.preprocessing import MinMaxScaler
import torch

def populate_all_stock_prices_from_raw_csv():
    print("SEEDER: Starting to populate 'stock_prices' table from raw CSVs...")
    init_db()

    for ticker_key in TICKERS:
        print(f"SEEDER: Populating stock_prices for {ticker_key}...")
        raw_file_path = get_raw_data_path(ticker_key)

        df_raw = pd.read_csv(raw_file_path, parse_dates=['Date'])
        df_raw.dropna(subset=['Date'], inplace=True)
        
        df_raw.set_index('Date', inplace=True)
        df_raw.sort_index(inplace=True)

        effective_close_series = pd.to_numeric(df_raw['Close'], errors='coerce')
        
        df_for_db = pd.DataFrame(effective_close_series.rename('Close'))
        df_for_db.dropna(subset=['Close'], inplace=True)

        save_actual_prices(ticker_key, df_for_db)
    print("SEEDER: Finished populating 'stock_prices' table.")

def seed_historical_predictions_for_all(start_date_str: str, end_date_str: str,
                                        models_to_seed: List[str] = ["xgboost", "random_forest", "lstm"]):
    print(f"SEEDER: Starting historical prediction seeding from {start_date_str} to {end_date_str}...")
    init_db()

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    for ticker_key in TICKERS:
        print(f"\nSEEDER: --- Seeding predictions for Ticker: {ticker_key} ---")

        # 1. Load full historical data for this ticker
        raw_file_path = get_raw_data_path(ticker_key)
        
        df_full_history_raw = pd.read_csv(raw_file_path, parse_dates=['Date'])
        df_full_history_raw.dropna(subset=['Date'], inplace=True)
        df_full_history_raw.set_index('Date', inplace=True)
        df_full_history_raw.sort_index(inplace=True)

        df_history_for_features = pd.DataFrame(index=df_full_history_raw.index)        

        all_needed_cols_for_features = list(set(LSTM_INPUT_FEATURE_COLUMNS + [ENSEMBLE_TARGET_COLUMN]))

        for col in all_needed_cols_for_features:
            if col in df_full_history_raw.columns:
                df_history_for_features[col] = pd.to_numeric(df_full_history_raw[col], errors='coerce')
            else:
                print(f"SEEDER: CRITICAL - Column '{col}' needed for features not found in raw data for {ticker_key}. Filling with NaN.")
                df_history_for_features[col] = np.nan

        df_history_for_features.dropna(subset=[c for c in all_needed_cols_for_features if c in df_history_for_features.columns],
                                    how='any', inplace=True)

        if df_history_for_features.empty:
            print(f"SEEDER: df_history_for_features for {ticker_key} is empty after NA drop. Skipping ticker.")
            continue

        # 2. Load models
        loaded_models = {}
        # LSTM scalers will be loaded inside prepare_input_sequence_for_lstm or make_prediction in model_utils
        for mt in models_to_seed:
            model = load_model(ticker_key, mt, 'regression') # From model_utils
            loaded_models[mt] = model

        # 3. Iterate through dates and make predictions
        current_pred_date = start_date
        while current_pred_date <= end_date:
            prediction_target_date_str = current_pred_date.strftime('%Y-%m-%d')
            historical_data_cutoff = current_pred_date - pd.Timedelta(days=1)
            data_for_current_features = df_history_for_features[df_history_for_features.index <= historical_data_cutoff].copy()

            for model_type, model_obj in loaded_models.items():
                prediction_input = None
                final_predicted_price_to_save = None

                if model_type in ["xgboost", "random_forest"]:
                    min_rows_needed = LAG_DAYS + 1
                    if len(data_for_current_features) < min_rows_needed: continue
                    
                    ensemble_feature_input_df = data_for_current_features[['Close']]
                    prediction_input_df = prepare_features_for_ensemble_model(ensemble_feature_input_df)
                    
                    predicted_pct_change = make_prediction(model_obj, model_type, "regression", prediction_input_df, ticker_key=ticker_key)

                    last_actual_close = data_for_current_features['Close'].iloc[-1]
                    if pd.notna(last_actual_close) and last_actual_close != 0:
                        final_predicted_price_to_save = last_actual_close * (1 + predicted_pct_change)
                
                elif model_type == "lstm":
                    min_rows_needed = LSTM_SEQUENCE_LENGTH
                    if len(data_for_current_features) < min_rows_needed: continue
                    
                    lstm_feature_df = data_for_current_features[LSTM_INPUT_FEATURE_COLUMNS]
                    prediction_input_tensor = prepare_input_sequence_for_lstm(lstm_feature_df, ticker_key)
                    
                    final_predicted_price_to_save = make_prediction(model_obj, model_type, "regression", prediction_input_tensor, ticker_key=ticker_key)

                if final_predicted_price_to_save is not None and pd.notna(final_predicted_price_to_save):
                    save_prediction(ticker_key, prediction_target_date_str, final_predicted_price_to_save, model_type)
                    
                    if current_pred_date in df_history_for_features.index:
                        actual_close_for_target_date = df_history_for_features.loc[current_pred_date, LSTM_TARGET_COLUMN]
                        if pd.notna(actual_close_for_target_date):
                            update_actual_price_for_prediction(ticker_key, prediction_target_date_str, float(actual_close_for_target_date))

            current_pred_date += pd.Timedelta(days=1)
        print(f"SEEDER: --- Finished seeding for Ticker: {ticker_key} ---")
    print("SEEDER: Historical prediction seeding finished.")


if __name__ == "__main__":
    print("===== Running Historical Seeder Script (app/historical_seeder.py) =====")
    print("\n--- Step 1: Populating 'stock_prices' table ---")
    # populate_all_stock_prices_from_raw_csv()

    start_date_for_predictions = "2025-01-01"
    end_date_for_predictions = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"\n--- Step 2: Seeding historical predictions from {start_date_for_predictions} to {end_date_for_predictions} ---")
    seed_historical_predictions_for_all(
        start_date_str=start_date_for_predictions,
        end_date_str=end_date_for_predictions,
        models_to_seed=["lstm"]
    )
    print("\n===== Historical Seeder Script Finished =====")