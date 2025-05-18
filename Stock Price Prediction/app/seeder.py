# app/seeder.py
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
    KNN_PARAMS,
    get_raw_data_path, get_model_path, get_lstm_scaler_path
)
from .model_utils import (
    load_model,
    prepare_features_for_ensemble_model,
    prepare_features_for_rfc,
    prepare_input_sequence_for_lstm,
    prepare_features_for_knn,
    make_prediction
)
from .db_utils import init_db, save_actual_prices, save_classification_prediction, save_prediction, update_actual_direction_for_classification, update_actual_price_for_prediction

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

        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_for_db = pd.DataFrame(index=df_raw.index)
        all_cols_present = True

        for col in cols_to_keep:
            if col in df_raw.columns:
                df_for_db[col] = pd.to_numeric(df_raw[col], errors='coerce')
            else:
                print(f"SEEDER: WARNING - Column '{col}' not found in raw data for {ticker_key}. Will be NaN in DB.")
                df_for_db[col] = np.nan # Hoặc bỏ qua nếu cột đó bắt buộc
                if col == 'Close': all_cols_present = False

        save_actual_prices(ticker_key, df_for_db)
    print("SEEDER: Finished populating 'stock_prices' table.")

def seed_historical_predictions_for_all(start_date_str: str, end_date_str: str,
                                        models_to_seed: List[str] = ["xgboost", "random_forest", "lstm", "knn"]):
    print(f"SEEDER: Starting historical prediction seeding from {start_date_str} to {end_date_str}...")
    init_db()

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    for ticker_key in TICKERS:
        print(f"\nSEEDER: --- Seeding predictions for Ticker: {ticker_key} ---")

        # 1. Load full historical data for this ticker
        raw_file_path = get_raw_data_path(ticker_key)
        
        df_full_history_raw = pd.read_csv(raw_file_path, parse_dates=['Date'], skiprows=[1])
        df_full_history_raw.dropna(subset=['Date'], inplace=True)
        df_full_history_raw.set_index('Date', inplace=True)
        df_full_history_raw.sort_index(inplace=True)

        df_history_for_features = pd.DataFrame(index=df_full_history_raw.index)        

        all_needed_cols_for_features = list(set(LSTM_INPUT_FEATURE_COLUMNS + [ENSEMBLE_TARGET_COLUMN] + ["Volume"]))

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
        for mt in models_to_seed:
            # For KNN and RFC, load both classification and regression models
            if mt == "knn":
                knn_class_model = load_model(ticker_key, "knn", "classification")
                if knn_class_model is not None:
                    loaded_models["knn_classification"] = knn_class_model
                
                knn_reg_model = load_model(ticker_key, "knn", "regression")
                if knn_reg_model is not None:
                    loaded_models["knn_regression"] = knn_reg_model
            elif mt == "random_forest":
                rfc_class_model = load_model(ticker_key, mt, "classification")
                if rfc_class_model is not None:
                    loaded_models["random_forest_classification"] = rfc_class_model
                rfc_reg_model = load_model(ticker_key, mt, "regression")
                if rfc_reg_model is not None:
                    loaded_models["random_forest_regression"] = rfc_reg_model
            else:
                model = load_model(ticker_key, mt, "regression")
                if model is not None:
                    loaded_models[mt] = model

        # Skip this ticker if no models were loaded
        if not loaded_models:
            print(f"SEEDER: No models loaded for {ticker_key}. Skipping ticker.")
            continue

        # 3. Iterate through dates and make predictions
        current_pred_date = start_date
        while current_pred_date <= end_date:
            prediction_target_date_str = current_pred_date.strftime('%Y-%m-%d')
            historical_data_cutoff = current_pred_date - pd.Timedelta(days=1)
            data_for_current_features = df_history_for_features[df_history_for_features.index <= historical_data_cutoff].copy()

            for model_type, model_obj in loaded_models.items():
                prediction_input = None
                final_predicted_price_to_save = None  # Initialize here for all model types
                problem_type = "regression"  # Default
                mt_for_db = model_type  # Model type to save in database

                # Determine problem type based on model type
                if model_type in ["knn_classification", "random_forest_classification"]:
                    problem_type = "classification"
                    mt_for_db = model_type.replace("_classification", "")
                # Prepare features based on model type
                if model_type == "xgboost" or model_type == "random_forest_regression":
                    min_rows_needed = LAG_DAYS + 1
                    if len(data_for_current_features) < min_rows_needed: continue
                    
                    ensemble_feature_input_df = data_for_current_features[['Close']]
                    prediction_input = prepare_features_for_ensemble_model(ensemble_feature_input_df)
                
                elif model_type == "lstm":
                    min_rows_needed = LSTM_SEQUENCE_LENGTH
                    if len(data_for_current_features) < min_rows_needed: continue
                    
                    lstm_feature_df = data_for_current_features[LSTM_INPUT_FEATURE_COLUMNS]
                    prediction_input = prepare_input_sequence_for_lstm(lstm_feature_df, ticker_key)
                
                elif model_type in ["knn_classification", "knn_regression"]:
                    min_rows_needed = KNN_PARAMS["seq_length"]
                    if len(data_for_current_features) < min_rows_needed: continue
                    
                    # Prepare KNN input features
                    knn_feature_series = data_for_current_features['Close']
                    prediction_input = prepare_features_for_knn(knn_feature_series, transform_name=KNN_PARAMS["transform"])
                
                elif model_type == "random_forest_classification":
                    min_rows_needed = 30  # Need at least 30 days for technical indicators
                    if len(data_for_current_features) < min_rows_needed: continue
                    
                    # Prepare RFC input features
                    prediction_input = prepare_features_for_rfc(data_for_current_features)
                    if prediction_input is None:
                        print(f"WARNING: Could not prepare features for RFC for {ticker_key}")
                        continue
                
                # Make prediction based on problem type
                if prediction_input is not None:
                    # Pass the current prediction date as as_of_date parameter
                    prediction_result = make_prediction(
                        model_obj, 
                        mt_for_db, 
                        problem_type, 
                        prediction_input, 
                        ticker_key=ticker_key,
                        as_of_date=current_pred_date  # Add this parameter
                    )
                    
                    if prediction_result is not None:
                        # Handle different result types
                        if problem_type == "classification" and isinstance(prediction_result, dict):
                            # Save classification prediction
                            direction = prediction_result.get("direction")
                            confidence = prediction_result.get("confidence")
                            window = prediction_result.get("window", 30)
                            
                            # Calculate the target date (30 days ahead)
                            target_date = current_pred_date + pd.Timedelta(days=window)
                            target_date_str = target_date.strftime('%Y-%m-%d')
                            
                            save_classification_prediction(
                                ticker_key=ticker_key,
                                prediction_date_str=target_date_str,
                                predicted_direction=direction,
                                confidence_score=confidence,
                                prediction_window=window,
                                model_used=mt_for_db
                            )
                            
                            # Check if we can update actual direction
                            if target_date in df_history_for_features.index:
                                start_price = data_for_current_features['Close'].iloc[-1]
                                end_price = df_history_for_features.loc[target_date, 'Close']
                                actual_direction = "up" if end_price > start_price else "down"
                                update_actual_direction_for_classification(
                                    ticker_key=ticker_key,
                                    date_str=target_date_str,
                                    actual_direction=actual_direction,
                                    prediction_window=window
                                )
                        else:
                            # For regression models, the result is already the predicted price
                            final_predicted_price_to_save = prediction_result
                            
                            # Save regression prediction
                            if final_predicted_price_to_save is not None and pd.notna(final_predicted_price_to_save):
                                save_prediction(ticker_key, prediction_target_date_str, float(final_predicted_price_to_save), mt_for_db)
                                
                                if current_pred_date in df_history_for_features.index:
                                    actual_close_for_target_date = df_history_for_features.loc[current_pred_date, LSTM_TARGET_COLUMN]
                                    if pd.notna(actual_close_for_target_date):
                                        update_actual_price_for_prediction(ticker_key, prediction_target_date_str, float(actual_close_for_target_date))

            current_pred_date += pd.Timedelta(days=1)
        print(f"SEEDER: --- Finished seeding for Ticker: {ticker_key} ---")
    print("SEEDER: Historical prediction seeding finished.")


if __name__ == "__main__":
    print("===== Running Historical Seeder Script (app/seeder.py) =====")
    print("\n--- Step 1: Populating 'stock_prices' table ---")
    populate_all_stock_prices_from_raw_csv()

    start_date_for_predictions = "2025-01-01"
    end_date_for_predictions = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"\n--- Step 2: Seeding historical predictions from {start_date_for_predictions} to {end_date_for_predictions} ---")
    seed_historical_predictions_for_all(
        start_date_str=start_date_for_predictions,
        end_date_str=end_date_for_predictions,
        models_to_seed=["xgboost", "random_forest", "knn", "lstm"]
    )
    print("\n===== Historical Seeder Script Finished =====")