# app/scheduler_tasks.py
import requests
from datetime import datetime
import pandas as pd
import os
import sys
import time

from .config import (
    FASTAPI_URL, TICKERS, ENSEMBLE_TARGET_COLUMN,
    get_raw_data_path
)
from .data_ingestion import fetch_all_tickers
from .db_utils import save_actual_prices, update_actual_price_for_prediction

def daily_data_ingestion_and_db_update_job():
    """
    Scheduled job to:
    1. Ingest fresh raw data for all configured tickers (saves to CSV).
    2. Read these fresh raw CSVs, extract values for 'Open', 'High', 'Low', 'Close', and 'Volume'.
    3. Save these latest prices to the 'stock_prices' table in the DB.
    4. Update 'actual_price' in the 'predictions' table for past prediction dates.
    """
    print(f"SCHEDULER_TASK: [{datetime.now()}] Running daily data ingestion and DB update job...")
    
    # 1. Ingest fresh raw data for all configured tickers (saves to CSV).
    print(f"SCHEDULER_TASK: Calling fetch_all_tickers() from data_ingestion module...")
    fetch_all_tickers()

    # 2. Read these fresh raw CSVs, extract values for 'Open', 'High', 'Low', 'Close', and 'Volume'.
    for ticker_key in TICKERS:
        print(f"SCHEDULER_TASK: Processing DB update for {ticker_key} from its raw CSV...")
        raw_file_path = get_raw_data_path(ticker_key)
        try:
            if not os.path.exists(raw_file_path):
                print(f"SCHEDULER_TASK: Raw data file {raw_file_path} not found for {ticker_key}. "
                      "Ingestion might have failed or path is incorrect. Skipping DB update for this ticker.")
                continue

            df_raw_today = pd.read_csv(raw_file_path, parse_dates=['Date'], skiprows=[1])
            if df_raw_today.empty:
                print(f"SCHEDULER_TASK: Raw data for {ticker_key} from CSV is empty. Skipping DB update.")
                continue
            
            df_raw_today.dropna(subset=['Date'], inplace=True)
            if df_raw_today.empty: continue
            df_raw_today.set_index('Date', inplace=True)
            df_raw_today.sort_index(inplace=True)

            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df_raw_today.columns]
            
            if missing_columns:
                print(f"SCHEDULER_TASK: Raw data for {ticker_key} is missing required columns: {missing_columns}. Trying to adapt...")
                
                df_for_db_update = df_raw_today.copy()
                
                if 'Close' not in missing_columns:
                    df_for_db_update.loc[:, 'Close'] = pd.to_numeric(df_for_db_update['Close'], errors='coerce')
                    
                    # If missing Open/High/Low, approximate them from Close
                    if 'Open' in missing_columns:
                        df_for_db_update.loc[:, 'Open'] = df_for_db_update['Close']
                    if 'High' in missing_columns:
                        df_for_db_update.loc[:, 'High'] = df_for_db_update['Close'] * 1.001
                    if 'Low' in missing_columns:
                        df_for_db_update.loc[:, 'Low'] = df_for_db_update['Close'] * 0.999
                    if 'Volume' in missing_columns:
                        df_for_db_update.loc[:, 'Volume'] = 0
                    
                    print(f"SCHEDULER_TASK: Created approximate OHLCV data for {ticker_key} based on Close prices")
                else:
                    print(f"SCHEDULER_TASK: Missing 'Close' column for {ticker_key}. Cannot proceed with DB update.")
                    continue
            else:
                df_for_db_update = df_raw_today.copy()
                
                for col in required_columns:
                    df_for_db_update.loc[:, col] = pd.to_numeric(df_for_db_update[col], errors='coerce')
                
                df_for_db_update = df_for_db_update[required_columns].copy()
            
            df_for_db_update = df_for_db_update.dropna(subset=required_columns)

            if not df_for_db_update.empty:
                # 3. Save these latest prices to the 'stock_prices' table
                save_actual_prices(ticker_key, df_for_db_update)

                # 4. Update 'actual_price' in the 'predictions' table for past prediction dates
                for date_val, row_data in df_for_db_update.iterrows():
                    date_str = date_val.strftime('%Y-%m-%d')
                    actual_close = row_data['Close']
                    update_actual_price_for_prediction(ticker_key, date_str, actual_close)
            else:
                print(f"SCHEDULER_TASK: No valid OHLCV prices to update in DB for {ticker_key} from raw CSV.")

        except FileNotFoundError:
             print(f"SCHEDULER_TASK: FileNotFoundError for {raw_file_path}. Check ingestion for {ticker_key}.")
        except pd.errors.EmptyDataError:
            print(f"SCHEDULER_TASK: EmptyDataError for {raw_file_path}. File might be empty for {ticker_key}.")
        except Exception as e:
            print(f"SCHEDULER_TASK: Error processing DB update for {ticker_key} from its raw CSV: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"SCHEDULER_TASK: [{datetime.now()}] Daily data ingestion and DB update job finished.")

def daily_prediction_trigger_job():
    print(f"SCHEDULER_TASK: [{datetime.now()}] Running daily prediction trigger job...")
    
    # Models to run predictions for - both regression and classification
    regression_models_to_predict_with = ["xgboost", "random_forest", "lstm", "gru"]
    classification_models_to_predict_with = ["random_forest", "knn"]

    # First, run all regression model predictions
    for ticker_key in TICKERS:
        for model_type in regression_models_to_predict_with:
            print(f"SCHEDULER_TASK: Triggering regression prediction for {ticker_key} using {model_type} model...")
            predict_url = f"{FASTAPI_URL}/predict"
            params = {"ticker_key": ticker_key, "model_type": model_type, "problem_type": "regression"}
            
            try:
                response = requests.post(predict_url, params=params, timeout=60)
                
                if 200 <= response.status_code < 300:
                    prediction_data = response.json()
                    print(f"SCHEDULER_TASK: API Regression prediction successful for {ticker_key} ({model_type}): "
                          f"Date: {prediction_data.get('prediction_date')}, Price: {prediction_data.get('predicted_close_price')}")
                else:
                    print(f"SCHEDULER_TASK: API Regression prediction FAILED for {ticker_key} ({model_type}). Status: {response.status_code}")
                    print(f"    Response: {response.text[:500]}")
            except Exception as e:
                print(f"SCHEDULER_TASK: Error calling regression prediction API for {ticker_key} ({model_type}): {str(e)}")
            
            time.sleep(1)  # Prevent overwhelming the API

    # Next, run all classification model predictions
    for ticker_key in TICKERS:
        for model_type in classification_models_to_predict_with:
            print(f"SCHEDULER_TASK: Triggering classification prediction for {ticker_key} using {model_type} model...")
            predict_url = f"{FASTAPI_URL}/predict"
            params = {"ticker_key": ticker_key, "model_type": model_type, "problem_type": "classification"}
            
            try:
                response = requests.post(predict_url, params=params, timeout=60)
                
                if 200 <= response.status_code < 300:
                    prediction_data = response.json()
                    print(f"SCHEDULER_TASK: API Classification prediction successful for {ticker_key} ({model_type}): "
                          f"Date: {prediction_data.get('prediction_date')}, Direction: {prediction_data.get('predicted_direction')}, "
                          f"Confidence: {prediction_data.get('confidence_score')}")
                else:
                    print(f"SCHEDULER_TASK: API Classification prediction FAILED for {ticker_key} ({model_type}). Status: {response.status_code}")
                    print(f"    Response: {response.text[:500]}")
            except Exception as e:
                print(f"SCHEDULER_TASK: Error calling classification prediction API for {ticker_key} ({model_type}): {str(e)}")
            
            time.sleep(1)  # Prevent overwhelming the API

    print(f"SCHEDULER_TASK: [{datetime.now()}] Daily prediction trigger job finished.")


if __name__ == "__main__":
    print("Testing scheduler tasks (make sure API server is running for prediction job)...")
    
    # Test data ingestion and DB update
    print("\n--- Testing Data Ingestion & DB Update Job ---")
    daily_data_ingestion_and_db_update_job()

    # Test prediction trigger
    print("\n--- Testing Prediction Trigger Job ---")
    os.environ["FASTAPI_URL"] = "http://localhost:8000"
    daily_prediction_trigger_job()
    
    print("Finished testing scheduler tasks. To run scheduled jobs, execute main_worker.py.")