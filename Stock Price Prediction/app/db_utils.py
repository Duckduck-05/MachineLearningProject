# app/db_utils.py
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

from .config import DATABASE_PATH, TICKERS, LSTM_INPUT_FEATURE_COLUMNS


def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH, timeout=10.0, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row # Access columns by name
    return conn

def init_db():
    """Initializes the database tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table to store historical actual closing prices for each ticker
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker_key TEXT NOT NULL,
            date TEXT NOT NULL,         -- Format YYYY-MM-DD
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL NOT NULL,
            volume INTEGER,
            PRIMARY KEY (ticker_key, date)
        )
    ''')

    # Table to store regression predictions (price values)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker_key TEXT NOT NULL,
            prediction_date TEXT NOT NULL, -- The date FOR WHICH the prediction is made (YYYY-MM-DD)
            predicted_price REAL NOT NULL,
            model_used TEXT NOT NULL,      -- e.g., "xgboost", "lstm"
            actual_price REAL,             -- Actual price for prediction_date, updated later
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- When the prediction record was inserted/updated
            UNIQUE (ticker_key, prediction_date, model_used) -- A ticker can have one prediction per model per day
        )
    ''')
    
    # New table for classification predictions (up/down)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classification_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker_key TEXT NOT NULL,
            prediction_date TEXT NOT NULL,    -- The date FOR WHICH the prediction is made (YYYY-MM-DD)
            predicted_direction TEXT NOT NULL, -- 'up' or 'down'
            confidence_score REAL,            -- Confidence of the prediction (0-1)
            prediction_window INTEGER NOT NULL, -- Days ahead (e.g., 30)
            model_used TEXT NOT NULL,         -- e.g., "knn", "rfc"
            actual_direction TEXT,            -- Actual direction, updated later
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (ticker_key, prediction_date, model_used, prediction_window)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"DB_UTILS: Database initialized/checked at {DATABASE_PATH}")

def save_actual_prices(ticker_key: str, df_ohlcv: pd.DataFrame):
    """
    Saves or updates actual closing prices for a given ticker.
    df_prices: DataFrame with 'Date' (DatetimeIndex) as index and a 'Close' column.
    """
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] # Các cột cần thiết
    missing_cols = [col for col in required_cols if col not in df_ohlcv.columns]
    if missing_cols:
        print(f"ERROR (DB_UTILS): df_ohlcv for {ticker_key} is missing required columns: {missing_cols}.")
        return
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    saved_count = 0
    for date_val, row in df_ohlcv.iterrows():
        if pd.isna(date_val):
            print(f"ERROR (DB_UTILS): Invalid date value for {ticker_key}. Skipping.")
            continue

        date_str = date_val.strftime('%Y-%m-%d')

        open_p = float(row['Open']) if pd.notna(row['Open']) else None
        high_p = float(row['High']) if pd.notna(row['High']) else None
        low_p = float(row['Low']) if pd.notna(row['Low']) else None
        close_p = float(row['Close']) if pd.notna(row['Close']) else None
        volume_v = int(row['Volume']) if pd.notna(row['Volume']) else None
        
        if close_p is None: # Giá Close là bắt buộc
            print(f"DB_UTILS: Skipping {ticker_key} on {date_str} due to NaN Close price.")
            continue

        cursor.execute("""
            INSERT INTO stock_prices (ticker_key, date, open_price, high_price, low_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker_key, date) DO UPDATE SET
                open_price=excluded.open_price,
                high_price=excluded.high_price,
                low_price=excluded.low_price,
                close_price=excluded.close_price,
                volume=excluded.volume
        """, (ticker_key, date_str, open_p, high_p, low_p, close_p, volume_v))
        saved_count += 1
    
    conn.commit()
    conn.close()
    print(f"DB_UTILS: Attempted to save/update {saved_count} OHLCV records for {ticker_key}.")


def save_prediction(ticker_key: str, prediction_date_str: str, predicted_price: float, model_used: str):
    """Saves or updates a prediction for a given ticker, date, and model."""
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO predictions (ticker_key, prediction_date, predicted_price, model_used)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker_key, prediction_date, model_used) DO UPDATE SET
            predicted_price=excluded.predicted_price,
            created_at=CURRENT_TIMESTAMP
    """, (ticker_key, prediction_date_str, predicted_price, model_used))
    conn.commit()
    print(f"DB_UTILS: Saved/Updated prediction for {ticker_key} on {prediction_date_str} using {model_used}.")
    conn.close()

def update_actual_price_for_prediction(ticker_key: str, date_str: str, actual_price: float):
    """Updates the actual_price in the predictions table for records where it's NULL."""
    conn = get_db_connection()
    try:
        # Update all model predictions for that ticker and date
        cursor = conn.execute("""
            UPDATE predictions SET actual_price = ?
            WHERE ticker_key = ? AND prediction_date = ? AND actual_price IS NULL
        """, (actual_price, ticker_key, date_str))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"DB_UTILS: Updated actual price for {cursor.rowcount} predictions for {ticker_key} on {date_str}.")
    except Exception as e:
        print(f"ERROR (DB_UTILS): Failed to update actual price for predictions of {ticker_key} on {date_str}: {e}")
    finally:
        conn.close()

def get_prediction_history(ticker_key: str, limit: int = 100) -> pd.DataFrame:
    """Retrieves prediction history for a specific ticker, joined with actual prices."""
    conn = get_db_connection()
    df = pd.DataFrame()

    # Join predictions with stock_prices on ticker_key and date
    query = """
            SELECT
                p.prediction_date,
                p.predicted_price,
                p.model_used,
                p.actual_price,
                sp.close_price as historical_actual -- Dấu chấm ở đây sp.close_price là hợp lệ
            FROM predictions p  -- p là alias cho predictions
            LEFT JOIN stock_prices sp -- sp là alias cho stock_prices
                ON UPPER(p.ticker_key) = UPPER(sp.ticker_key) AND p.prediction_date = sp.date -- Các dấu chấm ở đây cũng hợp lệ
            WHERE UPPER(p.ticker_key) = UPPER(?) -- UPPER(?) là cách truyền tham số, không phải lỗi
            ORDER BY p.prediction_date DESC
            LIMIT ?
        """
    df = pd.read_sql_query(query, conn, params=(ticker_key, limit))
    if 'prediction_date' in df.columns and not df.empty:
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    conn.close()
    return df

def get_latest_ohlcv_prices(ticker_key: str, days: int = 30, end_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Retrieves the latest 'days' of actual closing prices for a specific ticker.
    
    Args:
        ticker_key: The ticker symbol
        days: Number of days to retrieve
        end_date: Optional end date to limit the data (for historical simulations)
    """
    conn = get_db_connection()
    
    date_condition = ""
    params = [ticker_key, days]
    
    if end_date is not None:
        date_condition = "AND date <= ?"
        params = [ticker_key, end_date.strftime('%Y-%m-%d'), days]
    
    query = f"""
        SELECT date, open_price as Open, high_price as High, low_price as Low, 
               close_price as Close, volume as Volume
        FROM stock_prices
        WHERE UPPER(ticker_key) = UPPER(?)
        {date_condition}
        ORDER BY date DESC
        LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index(ascending=True)
    
    return df
def save_classification_prediction(ticker_key: str, prediction_date_str: str, predicted_direction: str, 
                                 confidence_score: float, prediction_window: int, model_used: str):
    """
    Saves or updates a classification prediction for a given ticker, date, and model.
    
    Args:
        ticker_key: The stock ticker symbol
        prediction_date_str: Date for which the prediction is made (YYYY-MM-DD)
        predicted_direction: 'up' or 'down'
        confidence_score: Confidence level of the prediction (0-1)
        prediction_window: Days ahead for prediction (e.g., 30)
        model_used: Model name that made the prediction
    """
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO classification_predictions 
        (ticker_key, prediction_date, predicted_direction, confidence_score, prediction_window, model_used)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker_key, prediction_date, model_used, prediction_window) DO UPDATE SET
            predicted_direction=excluded.predicted_direction,
            confidence_score=excluded.confidence_score,
            created_at=CURRENT_TIMESTAMP
    """, (ticker_key, prediction_date_str, predicted_direction, confidence_score, prediction_window, model_used))
    conn.commit()
    print(f"DB_UTILS: Saved classification prediction for {ticker_key} on {prediction_date_str} using {model_used} (window: {prediction_window} days).")
    conn.close()

def update_actual_direction_for_classification(ticker_key: str, date_str: str, actual_direction: str, prediction_window: int):
    """
    Updates the actual direction in the classification_predictions table.
    
    Args:
        ticker_key: The stock ticker symbol
        date_str: The prediction date to update
        actual_direction: The actual direction that occurred ('up' or 'down')
        prediction_window: The prediction window in days
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute("""
            UPDATE classification_predictions SET actual_direction = ?
            WHERE ticker_key = ? AND prediction_date = ? AND prediction_window = ? AND actual_direction IS NULL
        """, (actual_direction, ticker_key, date_str, prediction_window))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"DB_UTILS: Updated actual direction for {cursor.rowcount} classification predictions for {ticker_key} on {date_str}.")
    except Exception as e:
        print(f"ERROR (DB_UTILS): Failed to update actual direction for {ticker_key} on {date_str}: {e}")
    finally:
        conn.close()

def get_classification_prediction_history(ticker_key: str, prediction_window: int = 30, limit: int = 100) -> pd.DataFrame:
    """
    Retrieves classification prediction history for a specific ticker.
    
    Args:
        ticker_key: The stock ticker symbol
        prediction_window: The prediction window in days (default 30)
        limit: Maximum number of records to retrieve
        
    Returns:
        DataFrame with classification prediction history
    """
    conn = get_db_connection()
    query = """
        SELECT
            cp.prediction_date,
            cp.predicted_direction,
            cp.confidence_score,
            cp.prediction_window,
            cp.model_used,
            cp.actual_direction
        FROM classification_predictions cp
        WHERE UPPER(cp.ticker_key) = UPPER(?) AND cp.prediction_window = ?
        ORDER BY cp.prediction_date DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(ticker_key, prediction_window, limit))
    if 'prediction_date' in df.columns and not df.empty:
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    conn.close()
    return df

if __name__ == "__main__":
    print("Running DB Utils direct test (initializing DB)...")
    # This will create the DB file and tables if they don't exist
    # at the DATABASE_PATH defined in config.py (when config.py is in the same dir or PYTHONPATH is set)
    init_db()

    # Example: Test saving and fetching some data (requires config.py to be accessible)
    if "^GSPC" in TICKERS:
        print("\nTesting with GSPC...")
        # Create dummy price data for GSPC
        # test_dates = pd.to_datetime([datetime.now() - timedelta(days=i) for i in range(5, 0, -1)])
        # test_prices = pd.DataFrame({
            # 'Close': [100.0, 101.0, 100.5, 102.0, 101.5]
        # }, index=test_dates)
        # save_actual_prices("^GSPC", test_prices)
        # Fetch latest prices for GSPC
        latest_gspc = get_latest_ohlcv_prices("^GSPC", days=10)
        print("Latest GSPC prices from DB:")
        print(latest_gspc.to_string())

        # Save a dummy prediction
        # today_str = datetime.now().strftime('%Y-%m-%d')
        # save_prediction("^GSPC", today_str, 103.0, "xgboost_test")
        
        # Update actual for that prediction
        # update_actual_price_for_prediction("^GSPC", today_str, 102.5)

        # Get history
        history_gspc = get_prediction_history("^GSPC", limit=5)
        print("\nGSPC Prediction History from DB:")
        print(history_gspc.to_string())

    print("\nDB Utils direct test finished.")