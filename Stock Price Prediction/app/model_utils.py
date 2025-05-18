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
    ENSEMBLE_TARGET_COLUMN, KNN_PARAMS, LAG_DAYS, TICKERS, EXPECTED_FEATURES_ENSEMBLE,
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

# --- Global Cache for Loaded Models and Scalers ---
_loaded_models_cache: Dict[str, Any] = {}
_loaded_scalers_cache: Dict[str, MinMaxScaler] = {}

def _get_cache_key(ticker_key: str, object_type: str, sub_type: str = None) -> str:
    key = f"ticker_{ticker_key.upper()}_{object_type}"
    if sub_type:
        key += f"_{sub_type}"
    return key

# --- Model and Scaler Loading ---
def load_model(ticker_key: str, model_type: str, problem_type: str) -> Any:
    """Loads a specific model for a given ticker and caches it."""
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
            model = LSTMRegressor(input_size=LSTM_NUM_FEATURES, hidden_size=200) # hidden_size should match training
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
    cache_key = _get_cache_key(ticker_key, "lstm_scaler", scaler_name) # Use scaler_name as sub_type
    if cache_key in _loaded_scalers_cache:
        return _loaded_scalers_cache[cache_key]

    scaler_path = get_lstm_scaler_path(ticker_key, scaler_name) # From config
    scaler = None
    print(f"MODEL_UTILS: Loading LSTM scaler '{scaler_name}' for {ticker_key} from {scaler_path}...")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    _loaded_scalers_cache[cache_key] = scaler
    print(f"MODEL_UTILS: Successfully loaded LSTM scaler: {cache_key}")
    return scaler

# --- Feature Preparation for Prediction ---
def prepare_features_for_ensemble_model(historical_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features for XGBoost/RandomForest prediction from the latest historical data.
    historical_data_df: DataFrame with 'Date' as index, sorted, containing ENSEMBLE_TARGET_COLUMN.
                        Needs at least LAG_DAYS_ensemble + 1 rows.
    Returns a single-row DataFrame with features.
    """
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
    """
    Prepares the most recent sequence of OHL C data for LSTM prediction, scaling it.
    historical_data_df: DataFrame with 'Date' as index, sorted, containing LSTM_OHLC_FEATURE_COLUMNS.
                        Needs at least LSTM_SEQUENCE_LENGTH rows.
    ticker_key: To load the correct 'features_X_scaler'.
    Returns a PyTorch tensor for LSTM input or None on failure.
    """

    # Load the scaler used for X features during training
    features_X_scaler = load_lstm_scaler(ticker_key, "feature")
    sequence_data_df = historical_data_df[LSTM_INPUT_FEATURE_COLUMNS].tail(LSTM_SEQUENCE_LENGTH)
    
    # Ensure columns are in the same order as during scaler fitting (if scaler has feature_names_in_)
    if hasattr(features_X_scaler, 'feature_names_in_') and list(sequence_data_df.columns) != list(features_X_scaler.feature_names_in_):
        print(f"Warning (model_utils): Column order mismatch for LSTM scaling. Reordering. "
              f"Data has: {list(sequence_data_df.columns)}, Scaler expects: {list(features_X_scaler.feature_names_in_)}")
        sequence_data_df = sequence_data_df[features_X_scaler.feature_names_in_]
    
    scaled_sequence_np = features_X_scaler.transform(sequence_data_df.values) # Pass NumPy array
        
    # Reshape for LSTM: (batch_size=1, sequence_length, num_features)
    input_tensor = torch.from_numpy(scaled_sequence_np).float().unsqueeze(0)
    return input_tensor

def prepare_features_for_knn(historical_data_series, transform_name=None):
    """
    Prepare a sequence of historical data for KNN prediction
    
    Args:
        historical_data_series: pandas Series with historical Close prices
        transform_name: Transformation to apply ("standardize" or "min_max_scale")
    
    Returns:
        torch.Tensor: Prepared sequence ready for KNN prediction
    """
    try:
        # Ensure KNN_PARAMS is properly imported and contains seq_length
        if not KNN_PARAMS or not isinstance(KNN_PARAMS, dict) or "seq_length" not in KNN_PARAMS:
            print("ERROR: KNN_PARAMS not properly defined in config or missing seq_length")
            return None
        
        seq_length = KNN_PARAMS["seq_length"]
        
        # Check if historical_data_series has enough data
        if len(historical_data_series) < seq_length:
            print(f"ERROR: Not enough data points. Need {seq_length}, got {len(historical_data_series)}")
            return None
        
        # Get the most recent sequence of the required length
        sequence = historical_data_series.iloc[-seq_length:].values
        
        # Convert to tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        result_tensor = sequence_tensor.unsqueeze(0)  # Add batch dimension
        
        # Store the last price as an attribute on the tensor for later use in make_prediction
        result_tensor.base_close_price = historical_data_series.iloc[-1]
        
        return result_tensor
    except Exception as e:
        print(f"ERROR in prepare_features_for_knn: {str(e)}")
        return None

def prepare_features_for_rfc(historical_data_df):
    """
    Prepare features for Random Forest Classifier model
    
    Args:
        historical_data_df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators as features
    """
    try:
        from .utils.tech_indicators import calculate_technical_indicators
        
        # We need at least 30 days of data to calculate all indicators
        if len(historical_data_df) < 30:
            print("ERROR: Not enough data points for RFC features. Need at least 30 days.")
            return None
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in historical_data_df.columns]
        
        # If other columns are missing, we can't continue
        if missing_cols:
            print(f"ERROR: Missing required columns for RFC features: {missing_cols}")
            return None
        
        # Calculate all technical indicators
        features_df = calculate_technical_indicators(historical_data_df)
        
        # Drop NaN values (some indicators need several periods to initialize)
        features_df.dropna(inplace=True)
        
        if features_df.empty:
            print("ERROR: Features dataframe is empty after calculating indicators")
            return None
            
        return features_df
    except Exception as e:
        print(f"ERROR in prepare_features_for_rfc: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# --- Prediction Function ---
def make_prediction(model: any, model_type: str, problem_type: str, feature_input: any, ticker_key: str = None, as_of_date: pd.Timestamp = None) -> float | dict | None:
    """
    Makes a prediction using the loaded model.
    
    For regression models (problem_type="regression"), returns a predicted price.
    For classification models (problem_type="classification"), returns a dict with direction and confidence.
    
    Args:
        model: The loaded model to use for prediction
        model_type: Type of model ("xgboost", "random_forest", "lstm", "knn")
        problem_type: Type of prediction problem ("regression" or "classification")
        feature_input: Prepared features for the model
        ticker_key: Stock ticker symbol
        as_of_date: Optional timestamp to use for "as of" date calculations
    """
    # REGRESSION MODELS - Return a predicted price value
    if problem_type == "regression":
        try:
            # Models that predict percentage change (xgboost, random_forest)
            if model_type in ["xgboost", "random_forest"]:
                prediction_pct_change = model.predict(feature_input)[0]
                # Convert percentage change to actual price
                from .db_utils import get_latest_ohlcv_prices
                
                # If we're seeding historical predictions and as_of_date is provided
                if as_of_date is not None:
                    # Use the close price from the provided historical data
                    # This assumes the feature_input contains enough information
                    # to extract the last close price
                    if isinstance(feature_input, pd.DataFrame) and 'Close_lag_1' in feature_input.columns:
                        last_price = feature_input['Close_lag_1'].iloc[0]
                    else:
                        # Fall back to database query but with date limit
                        historical_data = get_latest_ohlcv_prices(ticker_key, days=5, end_date=as_of_date)
                        if historical_data.empty:
                            print(f"ERROR: No historical data found for {ticker_key} as of {as_of_date}")
                            return None
                        last_price = historical_data['Close'].iloc[-1]
                else:
                    # Normal real-time prediction - use latest data
                    historical_data = get_latest_ohlcv_prices(ticker_key, days=5)
                    if historical_data.empty:
                        print(f"ERROR: No historical data found for {ticker_key}")
                        return None
                    last_price = historical_data['Close'].iloc[-1]
                
                # Transform percentage change to actual price
                prediction_value = last_price * (1 + float(prediction_pct_change))
                print(f"Predicted pct change: {prediction_pct_change:.4f}, Last price: {last_price:.2f}, Predicted price: {prediction_value:.2f}")
                return float(prediction_value)
                
            # LSTM model (predicts actual price)
            elif model_type == "lstm":
                # LSTM predictions should be fine without modification since they're based
                # on the input sequence provided
                target_y_scaler = load_lstm_scaler(ticker_key, "target")
                model.eval()
                with torch.no_grad():
                    predicted_scaled_tensor = model(feature_input)
                
                predicted_scaled_np = predicted_scaled_tensor.cpu().numpy()
                prediction_value = target_y_scaler.inverse_transform(predicted_scaled_np)[0,0]
                return float(prediction_value)
                
            # KNN when used for regression
            elif model_type == "knn":
                # Convert input to tensor if needed
                input_tensor = torch.tensor(feature_input, dtype=torch.float32) if not isinstance(feature_input, torch.Tensor) else feature_input
                
                # For regression - direct price prediction
                prediction_result = model.predict(input_tensor, reduction="score")
                logit = prediction_result[1][0]
                confidence = prediction_result[0][0]
                
                # Get the last price either from provided data or database
                if as_of_date is not None:
                    # This assumes data_for_current_features was passed correctly in seeder.py
                    # Need to pass the data directly or extract it from feature_input
                    if hasattr(feature_input, 'base_close_price'):
                        last_price = feature_input.base_close_price
                    else:
                        historical_data = get_latest_ohlcv_prices(ticker_key, days=5, end_date=as_of_date)
                        if historical_data.empty:
                            return None
                        last_price = historical_data['Close'].iloc[-1]
                else:
                    from .db_utils import get_latest_ohlcv_prices
                    historical_data = get_latest_ohlcv_prices(ticker_key, days=5)
                    if historical_data.empty:
                        return None
                    last_price = historical_data['Close'].iloc[-1]
                
                # Use confidence to determine the strength of movement
                # Higher confidence = larger price move
                confidence_factor = min(max(confidence, 0.5), 0.95)  # Limit between 0.5 and 0.95
                move_size = 0.003 + (confidence_factor - 0.5) * 0.014  # Range from 0.3% to 1.0%
                
                if logit == 1:  # Up prediction
                    prediction_value = last_price * (1 + move_size)
                else:  # Down prediction
                    prediction_value = last_price * (1 - move_size)
                    
                return float(prediction_value)
        except Exception as e:
            print(f"ERROR making prediction with {model_type} for regression: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    # CLASSIFICATION MODELS - Return direction and confidence
    elif problem_type == "classification":
        try:
            # KNN when used for classification
            if model_type == "knn":
                # Convert input to tensor if needed
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
                    "window": 30  # KNN predicts 30 days ahead
                }
                
            # RFC (Random Forest Classifier) when used for classification
            # This is the native mode for RFC
            elif model_type == "random_forest":
                # For RFC, the prediction is a class (1 for up, -1 or 0 for down)
                prediction_class = model.predict(feature_input)[0]
                
                # If model has predict_proba, get the confidence level
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(feature_input)[0]
                    confidence = probabilities.max()  # Get the highest probability
                else:
                    confidence = 0.8  # Default confidence if not available
                
                # Return direction and confidence based on prediction window
                direction = "up" if prediction_class > 0 else "down"
                return {
                    "direction": direction,
                    "confidence": float(confidence),
                    "window": 30  # RFC predicts 30 days ahead by default
                }
            
            # Unsupported model type for classification
            else:
                print(f"ERROR: Unsupported model type for classification: {model_type}")
                return None
                
        except Exception as e:
            print(f"ERROR making prediction with {model_type} for classification: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    # If we reach here, something went wrong with the problem type
    print(f"ERROR: Unsupported problem_type: {problem_type}")
    return None

if __name__ == "__main__":
    print("--- Model Utils Direct Test (Loading Models and Scalers) ---")
    # This test assumes that training has been run and models/scalers exist.

    tk_key = "^GSPC" # Example ticker key
    print(f"\n--- Testing for Ticker: {tk_key} ---")
    
    from .db_utils import get_latest_ohlcv_prices
    historical_data_df = get_latest_ohlcv_prices("^GSPC", days=LSTM_SEQUENCE_LENGTH + 15)
    predict_input = prepare_features_for_knn(historical_data_df)
    print(historical_data_df.tail(15))
    # print(predict_input.head())

    print("\n--- Testing KNN Regression Prediction ---")
    model = load_model(tk_key, "knn", "regression")
    if model:
        prediction = make_prediction(model, "knn", "regression", predict_input, tk_key)
        print(f"Prediction: {prediction}")
    else:
        print("Failed to load Random Forest model for regression.")
    print("--- End Model Utils Direct Test ---")