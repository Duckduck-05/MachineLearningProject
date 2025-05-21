# app/api_server.py
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import os

# --- Imports from local app package ---
from .config import (
    GRU_INPUT_FEATURE_COLUMNS, TICKERS, TARGET_COLUMN,
    LAG_DAYS,
    LSTM_INPUT_FEATURE_COLUMNS, LSTM_SEQUENCE_LENGTH,
    ENSEMBLE_TARGET_COLUMN,
)
from .model_utils import (
    load_model,
    prepare_features_for_ensemble_model,
    prepare_input_sequence_for_gru,
    prepare_input_sequence_for_lstm,
    make_prediction
)
from .db_utils import (
    init_db,
    save_prediction,
    get_prediction_history,
    get_latest_ohlcv_prices,
    update_actual_price_for_prediction,
    save_classification_prediction,
    get_classification_prediction_history,
    get_classification_prediction_history_window
)
from .data_ingestion import fetch_all_tickers
from .data_processing import run_processing_for_all_tickers
from .seeder import populate_all_stock_prices_from_raw_csv

app = FastAPI(title="Stock & Index Price Prediction API")

# --- Dependency for loading models ---
async def get_model_for_prediction(
    ticker_key: str = Query(..., enum=TICKERS),
    model_type: str = Query("xgboost", enum=["xgboost", "random_forest", "lstm", "knn", "gru"]),
    problem_type: str = Query("regression", enum=["regression", "classification"])
) -> Any:
    print(f"API_DEPENDENCY: Attempting to load model: {ticker_key}, {model_type}, {problem_type}")
    model = load_model(ticker_key, model_type, problem_type)
    if model is None:
        print(f"API_DEPENDENCY: CRITICAL - Model {model_type} for {ticker_key} ({problem_type}) could not be loaded.")
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_type}' for ticker '{ticker_key}' is currently unavailable or failed to load."
        )
    return model

@app.on_event("startup")
async def startup_event():
    print("API_SERVER: Application startup...")
    init_db()
    print("API_SERVER: Database initialized.")
    print("API_SERVER: Startup complete.")


# --- Pydantic Models ---
class PredictionResponseAPI(BaseModel):
    ticker_key: str
    prediction_date: str
    predicted_close_price: Optional[float] = None
    predicted_direction: Optional[str] = None
    confidence_score: Optional[float] = None
    prediction_window: Optional[int] = None
    model_used: str
    predicted_value_type: str
    message: str

class HistoryDataPointAPI(BaseModel):
    prediction_date: str
    predicted_price: Optional[float] = None
    actual_price: Optional[float] = None
    model_used: Optional[str] = None

class HistoryResponseAPI(BaseModel):
    ticker_key: str
    data: List[HistoryDataPointAPI]

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Stock & Index Price Prediction API!"}

@app.post("/predict", response_model=PredictionResponseAPI)
async def predict_next_trading_day_endpoint(
    ticker_key: str = Query(..., enum=TICKERS),
    model_type: str = Query("xgboost", enum=["xgboost", "random_forest", "lstm", "knn", "gru"]),
    problem_type: str = Query("regression", enum=["regression", "classification"]),
    loaded_model_object: Any = Depends(get_model_for_prediction)
):
    print(f"API_SERVER: Prediction request for {ticker_key} using {model_type} model (problem type: {problem_type}, object type: {type(loaded_model_object)}).")

    days_needed_for_features = 0
    if model_type == 'xgboost' or (model_type == 'random_forest' and problem_type == 'regression'):
        days_needed_for_features = LAG_DAYS + 1
    elif model_type == "lstm" or model_type == "gru":
        days_needed_for_features = LSTM_SEQUENCE_LENGTH
    elif model_type == "knn":
        from .config import KNN_PARAMS
        days_needed_for_features = KNN_PARAMS.get("seq_length", 10)
    elif model_type == "random_forest" and problem_type == "classification":
        days_needed_for_features = 30
    
    historical_prices_df = get_latest_ohlcv_prices(ticker_key, days=days_needed_for_features + 15)

    if model_type == "random_forest" and problem_type == "classification":
        print(f"API_SERVER: RFC - Got {len(historical_prices_df)} days of historical data for {ticker_key}")
        print(f"API_SERVER: RFC - First and last dates: {historical_prices_df.index.min()} to {historical_prices_df.index.max()}")
        print(f"API_SERVER: RFC - Columns available: {historical_prices_df.columns.tolist()}")

    if historical_prices_df.empty or len(historical_prices_df) < days_needed_for_features:
        print(f"API_SERVER: Not enough data for {ticker_key}. Need {days_needed_for_features}, Got {len(historical_prices_df)}")
        raise HTTPException(status_code=404,
                            detail=f"Not enough historical OHLCV data in DB for {ticker_key} "
                                  f"(need at least {days_needed_for_features} days, "
                                  f"got {len(historical_prices_df)}) to make prediction with {model_type}.")

    feature_base_df = historical_prices_df.copy() 
    if ENSEMBLE_TARGET_COLUMN not in feature_base_df.columns:
        print(f"API_SERVER: CRITICAL - TARGET_COLUMN '{ENSEMBLE_TARGET_COLUMN}' missing in data from get_latest_ohlcv_data. Available: {feature_base_df.columns.tolist()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: Missing base column '{ENSEMBLE_TARGET_COLUMN}' in fetched data.")
    
    last_known_date = feature_base_df.index.max()
    prediction_target_dt = last_known_date
    days_to_add = 1
    while True:
        prediction_target_dt = last_known_date + pd.Timedelta(days=days_to_add)
        if prediction_target_dt.weekday() < 5: break
        days_to_add += 1
    prediction_date_str = prediction_target_dt.strftime("%Y-%m-%d")
    
    prediction_input_features = None
    predicted_value_type = "unknown"

    if model_type == "xgboost" or (model_type == "random_forest" and problem_type == "regression"):
        prediction_input_features = prepare_features_for_ensemble_model(feature_base_df[[ENSEMBLE_TARGET_COLUMN]].copy())
        predicted_value_type = "percentage_change_of_close"
    elif model_type == "lstm":
        if not all(col in feature_base_df.columns for col in LSTM_INPUT_FEATURE_COLUMNS):
            print(f"API_SERVER: CRITICAL - Missing columns for LSTM in feature_base_df. Needed: {LSTM_INPUT_FEATURE_COLUMNS}, Got: {feature_base_df.columns.tolist()}")
            raise HTTPException(status_code=500, detail="Internal server error: Missing data columns for LSTM processing.")
        prediction_input_features = prepare_input_sequence_for_lstm(feature_base_df[LSTM_INPUT_FEATURE_COLUMNS], ticker_key)
        predicted_value_type = "absolute_close"
    elif model_type == "gru":
        if not all(col in feature_base_df.columns for col in GRU_INPUT_FEATURE_COLUMNS):
            print(f"API_SERVER: CRITICAL - Missing columns for GRU in feature_base_df. Needed: {GRU_INPUT_FEATURE_COLUMNS}, Got: {feature_base_df.columns.tolist()}")
            raise HTTPException(status_code=500, detail="Internal server error: Missing data columns for GRU processing.")
        prediction_input_features = prepare_input_sequence_for_gru(feature_base_df[GRU_INPUT_FEATURE_COLUMNS], ticker_key)
        predicted_value_type = "absolute_close"
    elif model_type == "knn":
        from .model_utils import prepare_features_for_knn
        from .config import KNN_PARAMS
        
        print(f"API_SERVER: Preparing KNN features for {ticker_key} with problem type {problem_type}")
        
        close_series = feature_base_df[ENSEMBLE_TARGET_COLUMN]
        prediction_input_features = prepare_features_for_knn(close_series, transform_name=KNN_PARAMS.get("transform", "standardize"))
        
        if problem_type == "classification":
            predicted_value_type = "classification_based_price"
        else:
            predicted_value_type = "regression_based_price"
    elif model_type == "random_forest" and problem_type == "classification":
        from .model_utils import prepare_features_for_rfc
        
        print(f"API_SERVER: Preparing RFC features for {ticker_key}")
        prediction_input_features = prepare_features_for_rfc(feature_base_df)
        
        # Validate RFC features
        if prediction_input_features is None:
            raise HTTPException(status_code=500, 
                              detail=f"Failed to generate technical indicators for {ticker_key}. Check logs for details.")
        
        if prediction_input_features.empty:
            raise HTTPException(status_code=500,
                              detail=f"Technical indicators generated, but result is empty for {ticker_key}")
        
        # Make sure we have at least one row of data
        if len(prediction_input_features) < 1:
            raise HTTPException(status_code=500,
                              detail=f"Not enough rows in the technical indicators output for {ticker_key}")
        
        prediction_input_features = prediction_input_features.iloc[[-1]]
        predicted_value_type = "classification_based_price"

    if prediction_input_features is None or \
      (isinstance(prediction_input_features, pd.DataFrame) and prediction_input_features.empty) or \
      (isinstance(prediction_input_features, pd.DataFrame) and prediction_input_features.isnull().values.any()):
        nan_info = ""
        if isinstance(prediction_input_features, pd.DataFrame):
            na_columns = prediction_input_features.columns[prediction_input_features.isna().any()].tolist()
            if na_columns:
                nan_info = f" NaN found in columns: {na_columns}"
        raise HTTPException(status_code=500, detail=f"Could not create valid input features for {model_type} on {ticker_key}.{nan_info}")
    # print(f"model_type: {model_type}, prediction_input_features: {prediction_input_features}, problem_type: {problem_type}")
    raw_prediction = make_prediction(loaded_model_object, model_type, problem_type, prediction_input_features, ticker_key=ticker_key)
    if raw_prediction is None:
        raise HTTPException(status_code=500, detail=f"Prediction failed for {ticker_key} with {model_type}.")

    # For classification models
    if problem_type == "classification" and isinstance(raw_prediction, dict):
        direction = raw_prediction.get("direction")
        confidence = raw_prediction.get("confidence", 0.5)
        window = raw_prediction.get("window", 30)
        
        # Calculate target date (30 days ahead) for classification
        target_dt = last_known_date + pd.Timedelta(days=window)
        # Skip weekends
        while target_dt.weekday() >= 5:
            target_dt = target_dt + pd.Timedelta(days=1)
        target_date_str = target_dt.strftime("%Y-%m-%d")
        
        # Save to database
        save_classification_prediction(
            ticker_key=ticker_key,
            prediction_date_str=target_date_str,
            predicted_direction=direction,
            confidence_score=confidence,
            prediction_window=window,
            model_used=model_type
        )
        
        return PredictionResponseAPI(
            ticker_key=ticker_key,
            prediction_date=target_date_str,
            predicted_direction=direction,
            confidence_score=confidence,
            prediction_window=window,
            model_used=model_type,
            predicted_value_type="classification_direction",
            message=f"{window}-day prediction by {model_type} for {ticker_key}: {direction.upper()} with {confidence:.2f} confidence"
        )
    
    # For regression models
    final_predicted_close_price = None
    if model_type == "xgboost" or (model_type == "random_forest" and problem_type == "regression"):
        if not feature_base_df.empty and ENSEMBLE_TARGET_COLUMN in feature_base_df.columns:
            last_actual_close = feature_base_df[ENSEMBLE_TARGET_COLUMN].iloc[-1]
            if pd.notna(last_actual_close):
                final_predicted_close_price = raw_prediction
            else: 
                raise HTTPException(status_code=500, detail=f"Internal error: last actual close is NaN for {ticker_key}.")
        else: 
            raise HTTPException(status_code=500, detail=f"Internal error: base data empty for {ticker_key}.")
    elif model_type in ["lstm", "gru"]:
        final_predicted_close_price = raw_prediction

    if final_predicted_close_price is None or pd.isna(final_predicted_close_price):
        raise HTTPException(status_code=500, detail=f"Internal error: final prediction is invalid for {ticker_key} with {model_type}.")

    save_prediction(ticker_key, prediction_date_str, final_predicted_close_price, model_type)

    return PredictionResponseAPI(
        ticker_key=ticker_key,
        prediction_date=prediction_date_str,
        predicted_close_price=final_predicted_close_price,
        model_used=model_type,
        predicted_value_type=predicted_value_type,
        message=f"Prediction by {model_type} ({problem_type}) for {ticker_key}."
    )

@app.get("/history", response_model=HistoryResponseAPI)
async def get_history_endpoint(
    ticker_key: str = Query(..., enum=TICKERS),
    days_limit: int = Query(90, ge=1, le=730)
):
    print(f"API_SERVER: History request for {ticker_key}, limit {days_limit} days.")
    history_df = get_prediction_history(ticker_key, limit=days_limit)
    
    response_data_points = []
    if not history_df.empty:
        for _, row in history_df.iterrows():
            final_actual_price_for_response = row['actual_price'] if pd.notna(row['actual_price']) else row['historical_actual']
            
            response_data_points.append(HistoryDataPointAPI(
                prediction_date=row['prediction_date'].strftime('%Y-%m-%d'),
                predicted_price=float(row['predicted_price']) if pd.notna(row['predicted_price']) else None,
                model_used=row['model_used'],
                actual_price=float(final_actual_price_for_response) if pd.notna(final_actual_price_for_response) else None
            ))
    
    return HistoryResponseAPI(
        ticker_key=ticker_key,
        data=response_data_points
    )

def _run_data_pipeline_background():
    print("API_SERVER (Background): Starting full data pipeline...")
    print("API_SERVER (Background): Step 1 - Ingesting all tickers...")
    fetch_all_tickers()
    print("API_SERVER (Background): Step 2 - Processing all tickers (creating processed CSVs)...")
    run_processing_for_all_tickers()
    print("API_SERVER (Background): Step 3 - Populating stock_prices table from new raw data...")
    populate_all_stock_prices_from_raw_csv()
    print("API_SERVER (Background): Full data pipeline finished.")

@app.post("/trigger-data-update-all", status_code=202)
async def trigger_full_data_pipeline_endpoint(background_tasks: BackgroundTasks):
    print("API_SERVER: Received request to trigger full data pipeline (will run in background).")
    background_tasks.add_task(_run_data_pipeline_background)
    return {"message": "Full data update pipeline has been triggered and is running in the background."}

@app.get("/classification-history", response_model=HistoryResponseAPI)
async def get_classification_history_endpoint(
    ticker_key: str = Query(..., enum=TICKERS),
    prediction_window: int = Query(30, ge=5, le=90),
    days_limit: int = Query(180, ge=30, le=365)
):
    try:
        history_df = get_classification_prediction_history(ticker_key, prediction_window, days_limit)
        
        if history_df.empty:
            return HistoryResponseAPI(
                ticker_key=ticker_key,
                data=[],
                message=f"No classification prediction history found for {ticker_key} with {prediction_window}-day window"
            )
        
        data_points = []
        for _, row in history_df.iterrows():
            data_points.append(
                HistoryDataPointAPI(
                    prediction_date=row['prediction_date'].strftime('%Y-%m-%d'),
                    predicted_price=row['predicted_direction'],
                    actual_price=row['actual_direction'] if not pd.isna(row['actual_direction']) else None,
                    model_used=row['model_used']
                )
            )
            
        return HistoryResponseAPI(
            ticker_key=ticker_key,
            data=data_points,
            message=f"Classification prediction history for {ticker_key} with {prediction_window}-day window"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving classification history: {str(e)}")

@app.get("/classification-window")
async def get_classification_window_endpoint(
    ticker_key: str = Query(..., enum=TICKERS),
    model_type: str = Query(..., enum=["random_forest", "knn"]),
    days_limit: int = Query(30, ge=1, le=60)
):
    """Get classification predictions for the future window"""
    try:
        df = get_classification_prediction_history_window(ticker_key, model_type, days_limit)
        
        if df.empty:
            return {
                "ticker_key": ticker_key,
                "model_type": model_type,
                "data": []
            }
        
        result = []
        for _, row in df.iterrows():
            result.append({
                "prediction_date": row["prediction_date"].strftime("%Y-%m-%d"),
                "predicted_direction": row["predicted_direction"],
                "confidence_score": float(row["confidence_score"]) if pd.notna(row["confidence_score"]) else None
            })
            
        return {
            "ticker_key": ticker_key,
            "model_type": model_type,
            "data": result
        }
        
    except Exception as e:
        print(f"API_SERVER: Error retrieving classification window: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving classification window: {str(e)}")