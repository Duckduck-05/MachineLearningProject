# app/scheduler_tasks.py
import requests
from datetime import datetime
import pandas as pd
import os
import sys
import time # Added for a small delay between API calls

# Use relative imports, assuming this is run as part of the 'app' package
from .config import (
    FASTAPI_URL, TICKERS, ENSEMBLE_TARGET_COLUMN, # ENSEMBLE_TARGET_COLUMN is 'Close'
    get_raw_data_path # To check if raw data was fetched for db update
)
from .data_ingestion import fetch_all_tickers # Updated function name
from .db_utils import save_actual_prices, update_actual_price_for_prediction
# data_processing is not directly called by standard scheduler jobs here,
# as feature creation for prediction is handled by model_utils on-demand,
# and feature creation for training is part of model_training.py.
# If you had a separate "process all raw data to processed CSVs" job, you'd import it.



def daily_data_ingestion_and_db_update_job():
    """
    Scheduled job to:
    1. Ingest fresh raw data for all configured tickers (saves to CSV).
    2. Read these fresh raw CSVs, extract latest 'Close' prices.
    3. Save these latest 'Close' prices to the 'stock_prices' table in the DB.
    4. Update 'actual_price' in the 'predictions' table for past prediction dates.
    """
    print(f"SCHEDULER_TASK: [{datetime.now()}] Running daily data ingestion and DB update job...")
    
    # 1. Ingest fresh raw data for all tickers using the function from data_ingestion.py
    print(f"SCHEDULER_TASK: Calling fetch_all_tickers() from data_ingestion module...")
    fetch_all_tickers() # This function now saves raw data to CSV files for each ticker

    # 2. For each ticker, read its newly ingested raw CSV, extract 'Close', and update DB
    for ticker_key in TICKERS:
        print(f"SCHEDULER_TASK: Processing DB update for {ticker_key} from its raw CSV...")
        raw_file_path = get_raw_data_path(ticker_key) # config.py handles local/docker path
        try:
            if not os.path.exists(raw_file_path):
                print(f"SCHEDULER_TASK: Raw data file {raw_file_path} not found for {ticker_key}. "
                      "Ingestion might have failed or path is incorrect. Skipping DB update for this ticker.")
                continue

            df_raw_today = pd.read_csv(raw_file_path, parse_dates=['Date'])
            if df_raw_today.empty:
                print(f"SCHEDULER_TASK: Raw data for {ticker_key} from CSV is empty. Skipping DB update.")
                continue
            
            # Ensure 'Date' is index and sorted for reliable processing
            df_raw_today.dropna(subset=['Date'], inplace=True) # Remove rows with invalid dates
            if df_raw_today.empty: continue # Check again after dropna
            df_raw_today.set_index('Date', inplace=True)
            df_raw_today.sort_index(inplace=True)

            # Use 'Close' column as per new requirements (NO 'Adj_Close' preference)
            if 'Close' not in df_raw_today.columns:
                print(f"SCHEDULER_TASK: 'Close' column not found for {ticker_key} in raw data. Skipping DB update.")
                continue
                
            effective_close_series = pd.to_numeric(df_raw_today['Close'], errors='coerce')
            
            # Prepare DataFrame for save_actual_prices (expects 'Date' index, 'Close' column)
            df_for_db_update = pd.DataFrame(effective_close_series.rename('Close'))
            df_for_db_update.dropna(subset=['Close'], inplace=True) # Remove any NaNs after numeric conversion

            if not df_for_db_update.empty:
                # 3. Save these latest 'Close' prices to the 'stock_prices' table
                save_actual_prices(ticker_key, df_for_db_update)

                # 4. Update 'actual_price' in the 'predictions' table for past prediction dates
                for date_val, row_data in df_for_db_update.iterrows():
                    date_str = date_val.strftime('%Y-%m-%d')
                    actual_close = row_data['Close']
                    update_actual_price_for_prediction(ticker_key, date_str, actual_close)
            else:
                print(f"SCHEDULER_TASK: No valid close prices to update in DB for {ticker_key} from raw CSV.")

        except FileNotFoundError: # Should be caught by os.path.exists, but as a fallback
             print(f"SCHEDULER_TASK: FileNotFoundError for {raw_file_path}. Check ingestion for {ticker_key}.")
        except pd.errors.EmptyDataError:
            print(f"SCHEDULER_TASK: EmptyDataError for {raw_file_path}. File might be empty for {ticker_key}.")
        except Exception as e:
            print(f"SCHEDULER_TASK: Error processing DB update for {ticker_key} from its raw CSV: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"SCHEDULER_TASK: [{datetime.now()}] Daily data ingestion and DB update job finished.")


def daily_prediction_trigger_job():
    """
    Scheduled job to trigger predictions for all configured tickers and relevant regression models via API.
    """
    print(f"SCHEDULER_TASK: [{datetime.now()}] Running daily prediction trigger job...")
    
    # Define which regression models to run predictions for
    # Classification models would be handled separately if/when implemented
    regression_models_to_predict_with = ["xgboost", "random_forest", "lstm", "knn"]

    for ticker_key in TICKERS:
        for model_type in regression_models_to_predict_with:
            print(f"SCHEDULER_TASK: Triggering prediction for {ticker_key} using {model_type} model...")
            # API endpoint requires problem_type, but we are focusing on regression here.
            # The API server's /predict endpoint should default or handle 'regression'.
            # If API needs problem_type explicitly: params = {"ticker_key": ticker_key, "model_type": model_type, "problem_type": "regression"}
            
            predict_url = f"{FASTAPI_URL}/predict"
            params = {"ticker_key": ticker_key, "model_type": model_type} # problem_type default is regression in API
            
            # NO TRY-EXCEPT for the request itself, as per your strict requirement
            # If API call fails, the worker log will show the error from requests library or an unhandled exception.
            # However, for robustness in a real system, error handling for network issues is crucial.
            
            response = requests.post(predict_url, params=params, timeout=60) # Increased timeout
            
            # Assuming successful call if no exception was raised by requests.post() itself
            # For more robust checking without try-except, you'd check response.status_code
            if 200 <= response.status_code < 300:
                prediction_data = response.json() # This can fail if response is not JSON
                print(f"SCHEDULER_TASK: API Prediction successful for {ticker_key} ({model_type}): "
                      f"Date: {prediction_data.get('prediction_date')}, Price: {prediction_data.get('predicted_close_price')}")
            else:
                print(f"SCHEDULER_TASK: API Prediction FAILED for {ticker_key} ({model_type}). Status: {response.status_code}")
                print(f"    Response: {response.text[:500]}") # Print part of the response text

            time.sleep(1) # Small delay between API calls to be polite to the API server

    print(f"SCHEDULER_TASK: [{datetime.now()}] Daily prediction trigger job finished.")


if __name__ == "__main__":
    # For testing individual jobs directly
    # Ensure necessary environment variables are set (e.g., ALPHA_VANTAGE_API_KEY)
    # and that the API server is running if testing daily_prediction_trigger_job.
    print("Testing scheduler tasks directly (make sure API server is running for prediction job)...")
    
    # Test data ingestion and DB update
    # print("\n--- Testing Data Ingestion & DB Update Job ---")
    # daily_data_ingestion_and_db_update_job()

    # Test prediction trigger
    # print("\n--- Testing Prediction Trigger Job ---")
    # os.environ["FASTAPI_URL"] = "http://localhost:8000" # Set for local test if API is local
    # daily_prediction_trigger_job()
    
    print("Finished testing scheduler tasks. To run scheduled jobs, execute main_worker.py.")