# app/db_utils.py
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

from .config import DATABASE_PATH, TICKERS, LSTM_INPUT_FEATURE_COLUMNS


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH, timeout=10.0, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

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

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker_key TEXT NOT NULL,
            prediction_date TEXT NOT NULL, -- Format YYYY-MM-DD
            predicted_price REAL NOT NULL,
            model_used TEXT NOT NULL,
            actual_price REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (ticker_key, prediction_date, model_used)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classification_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker_key TEXT NOT NULL,
            prediction_date TEXT NOT NULL,    -- Format YYYY-MM-DD
            predicted_direction TEXT NOT NULL,
            confidence_score REAL,
            prediction_window INTEGER NOT NULL,
            model_used TEXT NOT NULL,
            actual_direction TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (ticker_key, prediction_date, model_used, prediction_window)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"DB_UTILS: Database initialized/checked at {DATABASE_PATH}")

def save_actual_prices(ticker_key: str, df_ohlcv: pd.DataFrame):
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
        
        if close_p is None:
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
    conn = get_db_connection()
    cursor = conn.execute("""
        UPDATE predictions SET actual_price = ?
        WHERE ticker_key = ? AND prediction_date = ? AND actual_price IS NULL
    """, (actual_price, ticker_key, date_str))
    conn.commit()
    if cursor.rowcount > 0:
        print(f"DB_UTILS: Updated actual price for {cursor.rowcount} predictions for {ticker_key} on {date_str}.")
    conn.close()

def get_prediction_history(ticker_key: str, limit: int = 100) -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.DataFrame()
    query = """
            SELECT
                p.prediction_date,
                p.predicted_price,
                p.model_used,
                p.actual_price,
                sp.close_price as historical_actual
            FROM predictions p
            LEFT JOIN stock_prices sp
                ON UPPER(p.ticker_key) = UPPER(sp.ticker_key) AND p.prediction_date = sp.date
            WHERE UPPER(p.ticker_key) = UPPER(?)
            ORDER BY p.prediction_date DESC
            LIMIT ?
        """
    df = pd.read_sql_query(query, conn, params=(ticker_key, limit))
    if 'prediction_date' in df.columns and not df.empty:
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    conn.close()
    return df

def get_latest_ohlcv_prices(ticker_key: str, days: int = 30, end_date: pd.Timestamp = None) -> pd.DataFrame:
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

def get_classification_prediction_history_window(ticker_key: str, model_type: str, days_limit: int = 30) -> pd.DataFrame:
    """
    Get classification predictions for the next 30 days window
    
    Args:
        ticker_key: Ticker symbol
        model_type: Model type (random_forest, knn)
        days_limit: How many days of history to retrieve
        
    Returns:
        DataFrame with predictions for the next days within the prediction window
    """
    conn = get_db_connection()
    
    query = """
    SELECT 
        prediction_date,
        predicted_direction,
        confidence_score,
        model_used,
        prediction_window
    FROM classification_predictions
    WHERE ticker_key = ? 
      AND model_used = ?
      AND prediction_date >= date('now')
    ORDER BY prediction_date ASC
    LIMIT ?
    """
    
    try:
        df = pd.read_sql_query(query, conn, params=(ticker_key, model_type, days_limit))
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        return df
    except Exception as e:
        print(f"DB_UTILS: Error retrieving classification prediction window history: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

if __name__ == "__main__":
    print("Running DB Utils test...")
    init_db()

    if "^GSPC" in TICKERS:
        print("\nTesting with GSPC...")
        latest_gspc = get_latest_ohlcv_prices("^GSPC", days=10)
        print("Latest GSPC prices from DB:")
        print(latest_gspc.to_string())
        history_gspc = get_prediction_history("^GSPC", limit=5)
        print("\nGSPC Prediction History from DB:")
        print(history_gspc.to_string())

    print("\nDB Utils test finished.")