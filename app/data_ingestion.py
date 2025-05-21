# app/data_ingestion.py
import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime
from .config import TICKERS, get_raw_data_path

FETCH_START_DATE = "2000-01-01"

def fetch_yfinance_data(api_ticker: str, start_date: str = FETCH_START_DATE) -> pd.DataFrame:
    """Fetches all available historical data using yfinance from start_date to the current date."""
    print(f"Fetching data for {api_ticker} using yfinance (from {start_date})...")
    
    data = yf.download(api_ticker, start=start_date, progress=False, auto_adjust=False, actions=False)
    if data.empty:
        print(f"No data returned for {api_ticker} from yfinance for period starting {start_date}.")
        return pd.DataFrame()
    
    data.reset_index(inplace=True)
    
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', ascending=True, inplace=True)
    
    print(f"Successfully fetched {len(data)} rows for {api_ticker} from yfinance.")
    return data

def fetch_all_tickers(start_date: str = FETCH_START_DATE):
    """Runs data ingestion for all tickers defined in TICKERS_CONFIG."""
    for ticker_key in TICKERS:
        print(f"\n--- Ingesting data for {ticker_key} from {start_date} ---")

        df_raw = fetch_yfinance_data(ticker_key, start_date=start_date)
        raw_file_path = get_raw_data_path(ticker_key)
        df_raw.to_csv(raw_file_path, index=False)
        print(f"Raw data for {ticker_key} saved to: {raw_file_path}")

if __name__ == "__main__":
    fetch_all_tickers()