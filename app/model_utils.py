# app/model_utils.py
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import torch
import torch.nn as nn
import os
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Union, Any

from .utils.helper import standardize, min_max_scale

from .config import (
    ENSEMBLE_TARGET_COLUMN, GRU_INPUT_FEATURE_COLUMNS, GRU_SEQUENCE_LENGTH, KNN_PARAMS, LAG_DAYS, TICKERS, EXPECTED_FEATURES_ENSEMBLE,
    LSTM_SEQUENCE_LENGTH, LSTM_NUM_FEATURES, LSTM_INPUT_FEATURE_COLUMNS, get_lstm_scaler_path,
    get_model_path
)

# LSTM Model Definition
class LSTMRegressor(nn.Module):
  def __init__(self, input_size, hidden_size, output_size=1):
    super(LSTMRegressor, self).__init__()
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    # x shape: (batch_size, seq_length, input_size)
    # Initialize hidden state and cell state
    h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_size, device=x.device).requires_grad_()
    c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_size, device=x.device).requires_grad_()

    out, _ = self.lstm(x, (h0.detach(), c0.detach())) # out shape: (batch_size, seq_length, hidden_size)
    out = self.fc(out[:, -1, :]) # out shape: (batch_size, output_size)
    return out

# GRU Model Definition
class GRUmodel(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(GRUmodel, self).__init__()
    self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
  def forward(self, x):
    out, _ = self.gru(x)
    out = self.fc(out[:, -1, :])
    return out

# --- Global Cache for Loaded Models and Scalers ---
_loaded_models_cache: Dict[str, Any] = {}
_loaded_scalers_cache: Dict[str, MinMaxScaler] = {}

def _get_cache_key(ticker_key: str, object_type: str, sub_type: str = None) -> str:
    key = f"{ticker_key.upper()}_{object_type}"
    if sub_type:
        key += f"_{sub_type}"
    return key

# --- Model and Scaler Loading ---
def load_model(ticker_key: str, model_type: str, problem_type: str) -> Any:
    cache_key = _get_cache_key(ticker_key, model_type, sub_type=problem_type)
    if cache_key in _loaded_models_cache:
        return _loaded_models_cache[cache_key]

    model_path = get_model_path(ticker_key, model_type, problem_type)
    model = None
    if not os.path.exists(model_path):
        print(f"MODEL_UTILS: Model file not found at {model_path}")
        return None
        
    print(f"MODEL_UTILS: Loading model {model_type} for {ticker_key} from {model_path}...")

    try:
        if model_type == "xgboost":
            model = xgb.XGBRegressor()
            model.load_model(model_path)
        elif model_type in ["random_forest", "knn"]:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_type == "lstm":
            model = LSTMRegressor(input_size=LSTM_NUM_FEATURES, hidden_size=200)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
        elif model_type == "gru":
            if ticker_key == "^GSPC":
                model = GRUmodel(input_size=LSTM_NUM_FEATURES, hidden_size=1000, output_size=1)
            elif ticker_key == "IBM":
                model = GRUmodel(input_size=LSTM_NUM_FEATURES, hidden_size=500, output_size=1)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
        else:
            print(f"MODEL_UTILS: Unknown model type: {model_type}")
            return None
        
        _loaded_models_cache[cache_key] = model
        print(f"MODEL_UTILS: Successfully loaded model: {cache_key}")
        return model
    except Exception as e:
        print(f"MODEL_UTILS: Error loading model {model_type} for {ticker_key}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_lstm_scaler(ticker_key: str, scaler_name: str) -> MinMaxScaler | None:
    cache_key = _get_cache_key(ticker_key, "lstm_scaler", scaler_name)
    if cache_key in _loaded_scalers_cache:
        return _loaded_scalers_cache[cache_key]

    scaler_path = get_lstm_scaler_path(ticker_key, scaler_name)
    scaler = None
    print(f"MODEL_UTILS: Loading LSTM scaler '{scaler_name}' for {ticker_key} from {scaler_path}...")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    _loaded_scalers_cache[cache_key] = scaler
    print(f"MODEL_UTILS: Successfully loaded LSTM scaler: {cache_key}")
    return scaler

def load_gru_scaler(ticker_key: str, scaler_name: str) -> MinMaxScaler | None:
    cache_key = _get_cache_key(ticker_key, "gru_scaler", scaler_name)
    if cache_key in _loaded_scalers_cache:
        return _loaded_scalers_cache[cache_key]

    scaler_path = get_lstm_scaler_path(ticker_key, scaler_name)
    scaler = None
    print(f"MODEL_UTILS: Loading GRU scaler '{scaler_name}' for {ticker_key} from {scaler_path}...")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    _loaded_scalers_cache[cache_key] = scaler
    print(f"MODEL_UTILS: Successfully loaded GRU scaler: {cache_key}")
    return scaler

# --- Feature Preparation for Prediction ---
def prepare_features_for_ensemble_model(historical_data_df: pd.DataFrame) -> pd.DataFrame:
    required_rows = LAG_DAYS + 1

    features = {}
    latest_data_series = historical_data_df[ENSEMBLE_TARGET_COLUMN].sort_index(ascending=True).tail(required_rows)
    # print(latest_data_series)
    for i in range(1, LAG_DAYS + 1):
        feature_name = f'Close_lag_{i}'
        features[feature_name] = latest_data_series.iloc[-(i + 1)]

    pct_change_feature_name = 'Close_pct_change_1d'
    if pct_change_feature_name in EXPECTED_FEATURES_ENSEMBLE:
        if len(latest_data_series) >= 2:
            current_val = latest_data_series.iloc[-1]
            previous_val = latest_data_series.iloc[-2]
            features[pct_change_feature_name] = (current_val - previous_val) / previous_val if previous_val != 0 else 0.0
        else:
            features[pct_change_feature_name] = 0.0
    feature_df = pd.DataFrame([features], columns=EXPECTED_FEATURES_ENSEMBLE)
    return feature_df

def prepare_input_sequence_for_lstm(historical_data_df: pd.DataFrame, ticker_key: str) -> torch.Tensor | None:
    features_X_scaler = load_lstm_scaler(ticker_key, "feature")
    sequence_data_df = historical_data_df[LSTM_INPUT_FEATURE_COLUMNS].tail(LSTM_SEQUENCE_LENGTH)
    
    scaled_sequence_np = features_X_scaler.transform(sequence_data_df.values)
        
    input_tensor = torch.from_numpy(scaled_sequence_np).float().unsqueeze(0)
    return input_tensor

def prepare_features_for_knn(historical_data_series, transform_name=None):
    seq_length = KNN_PARAMS["seq_length"]
    sequence = historical_data_series.iloc[-seq_length:].values
    
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
    result_tensor = sequence_tensor.unsqueeze(0)
    
    result_tensor.base_close_price = historical_data_series.iloc[-1]
    
    return result_tensor

def prepare_features_for_rfc(historical_data_df):
    from .utils.tech_indicators import calculate_technical_indicators
    # Calculate all technical indicators
    features_df = calculate_technical_indicators(historical_data_df)
    # Drop NaN values
    features_df.dropna(inplace=True)    
    return features_df

def prepare_input_sequence_for_gru(historical_data_df: pd.DataFrame, ticker_key: str) -> torch.Tensor | None:
    features_X_scaler = load_gru_scaler(ticker_key, "feature")
    sequence_data_df = historical_data_df[GRU_INPUT_FEATURE_COLUMNS].tail(GRU_SEQUENCE_LENGTH)
    
    scaled_sequence_np = features_X_scaler.transform(sequence_data_df.values)
        
    input_tensor = torch.from_numpy(scaled_sequence_np).float().unsqueeze(0)
    return input_tensor

# --- Prediction Function ---
def make_prediction(model: any, model_type: str, problem_type: str, feature_input: any, ticker_key: str = None, as_of_date: pd.Timestamp = None) -> float | dict | None:
    # REGRESSION MODELS
    if problem_type == "regression":
        try:
            if model_type in ["xgboost", "random_forest_regression", "random_forest"]:
                prediction_pct_change = model.predict(feature_input)[0]
                from .db_utils import get_latest_ohlcv_prices
                
                if as_of_date is not None:
                    if isinstance(feature_input, pd.DataFrame) and 'Close_lag_1' in feature_input.columns:
                        last_price = feature_input['Close_lag_1'].iloc[0]
                    else:
                        historical_data = get_latest_ohlcv_prices(ticker_key, days=5, end_date=as_of_date)
                        if historical_data.empty:
                            print(f"ERROR: No historical data found for {ticker_key} as of {as_of_date}")
                            return None
                        last_price = historical_data['Close'].iloc[-1]
                else:
                    historical_data = get_latest_ohlcv_prices(ticker_key, days=5)
                    last_price = historical_data['Close'].iloc[-1]
                
                # Transform percentage change to actual price
                prediction_value = last_price * (1 + float(prediction_pct_change))
                print(f"Predicted pct change: {prediction_pct_change:.4f}, Last price: {last_price:.2f}, Predicted price: {prediction_value:.2f}")
                return float(prediction_value)                
            elif model_type == "lstm":
                target_y_scaler = load_lstm_scaler(ticker_key, "target")
                model.eval()
                with torch.no_grad():
                    predicted_scaled_tensor = model(feature_input)
                
                predicted_scaled_np = predicted_scaled_tensor.cpu().numpy()
                prediction_value = target_y_scaler.inverse_transform(predicted_scaled_np)[0,0]
                return float(prediction_value)
            elif model_type == "gru":
                target_y_scaler = load_gru_scaler(ticker_key, "target")
                model.eval()
                with torch.no_grad():
                    predicted_scaled_tensor = model(feature_input)
                
                predicted_scaled_np = predicted_scaled_tensor.cpu().numpy()
                prediction_value = target_y_scaler.inverse_transform(predicted_scaled_np)[0,0]
                return float(prediction_value)
        except Exception as e:
            print(f"ERROR making prediction with {model_type} for regression: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    # CLASSIFICATION MODELS
    elif problem_type == "classification":
        try:
            if model_type == "knn":
                input_tensor = torch.tensor(feature_input, dtype=torch.float32) if not isinstance(feature_input, torch.Tensor) else feature_input
                # Get prediction from KNN model - returns confidence score, predicted class
                prediction_result = model.predict(input_tensor, reduction="score")
                logit = prediction_result[1][0]  # The predicted class (0: down, 1: up)
                confidence = prediction_result[0][0]  # Confidence score
                
                # Return direction and confidence for 30-day prediction
                direction = "up" if logit == 1 else "down"
                return {
                    "direction": direction,
                    "confidence": float(confidence),
                    "window": 30
                }
            elif model_type == "random_forest":
                prediction_class = model.predict(feature_input)[0]
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(feature_input)[0]
                    confidence = probabilities.max()
                else:
                    confidence = 0.8
                
                direction = "up" if prediction_class > 0 else "down"
                return {
                    "direction": direction,
                    "confidence": float(confidence),
                    "window": 30
                }
            else:
                print(f"ERROR: Unsupported model type for classification: {model_type}")
                return None
                
        except Exception as e:
            print(f"ERROR making prediction with {model_type} for classification: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    print(f"ERROR: Unsupported problem_type: {problem_type}")
    return None

if __name__ == "__main__":
    print("--- Model Utils Test ---")
    tk_key = "^GSPC"
    print(f"\n--- Testing for Ticker: {tk_key} ---")
    
    from .db_utils import get_latest_ohlcv_prices
    historical_data_df = get_latest_ohlcv_prices("^GSPC", days=LSTM_SEQUENCE_LENGTH + 15)
    predict_input = prepare_features_for_knn(historical_data_df)
    print(historical_data_df.tail(15))
    print(predict_input.head())

    print("\n--- Testing KNN Prediction ---")
    model = load_model(tk_key, "knn", "classification")
    prediction = make_prediction(model, "knn", "classification", predict_input, tk_key)
    print(f"Prediction: {prediction}")
    print("--- End Model Utils Test ---")