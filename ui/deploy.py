# ui/streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time

# --- Configuration ---
DEFAULT_API_URL = "http://52.64.105.162:8000"
FASTAPI_URL = os.getenv("FASTAPI_URL_FOR_UI", DEFAULT_API_URL)

TICKERS_DISPLAY_CONFIG = {
    "^GSPC": {"display_name": "S&P 500 Index"},
    "IBM": {"display_name": "IBM Stock"}
}
AVAILABLE_TICKER_KEYS = list(TICKERS_DISPLAY_CONFIG.keys())
REGRESSION_MODELS_TO_DISPLAY = ["xgboost", "random_forest", "lstm", "gru"]
CLASSIFICATION_MODELS_TO_DISPLAY = ["random_forest", "knn"]

st.set_page_config(
    layout="wide", 
    page_title="Stock & Index Price Prediction",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)
st.title("📈 Stock & Index Price Prediction Dashboard")
st.markdown("---")

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# --- Sidebar for Ticker Selection ---
st.sidebar.header("🎯 Select Ticker")
selected_ticker_key_api = st.sidebar.selectbox(
    "Ticker:",
    options=AVAILABLE_TICKER_KEYS,
    format_func=lambda x: TICKERS_DISPLAY_CONFIG[x]["display_name"],
    key="ticker_selector_main_ui_v2"
)
selected_ticker_display_name = TICKERS_DISPLAY_CONFIG[selected_ticker_key_api]["display_name"]

# --- Section 1: Latest Predictions (Auto-fetched for all models) ---
st.header(f"🔮 Latest Predictions for {selected_ticker_display_name}")
st.caption(f"Predictions for the next trading day. Refresh page or use sidebar to update.")

# Function to fetch prediction for a specific model
@st.cache_data(ttl=120, show_spinner=False) # Cache for 2 minutes
def fetch_latest_prediction(ticker_key_for_api: str, model_type_for_api: str, problem_type="regression"):
    print(f"UI: Fetching latest prediction for {ticker_key_for_api} using {model_type_for_api} ({problem_type}) from {FASTAPI_URL}")
    try:
        predict_url = f"{FASTAPI_URL}/predict"
        params = {"ticker_key": ticker_key_for_api, "model_type": model_type_for_api, "problem_type": problem_type}
        response = requests.post(predict_url, params=params, timeout=25)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        error_detail = "Unknown API Error"
        if http_err.response is not None:
            try: error_detail = http_err.response.json().get('detail', http_err.response.text)
            except: error_detail = http_err.response.text
        print(f"UI API Error ({model_type_for_api}): {http_err.response.status_code} - {error_detail}")
        return {"error": f"API Error ({http_err.response.status_code}): {error_detail}"}
    except requests.exceptions.RequestException as req_err:
        print(f"UI Request Error ({model_type_for_api}): {req_err}")
        return {"error": f"Request Error: Could not connect to API. {req_err}"}
    except Exception as e:
        print(f"UI Error fetching prediction for {model_type_for_api}: {e}")
        return {"error": f"Unexpected error: {e}"}

# Add this function after existing fetch functions
@st.cache_data(ttl=120, show_spinner=False) 
def fetch_classification_window(ticker_key: str, model_type: str, days_limit: int = 30):
    """Fetch classification predictions for the next 30 days window"""
    try:
        url = f"{FASTAPI_URL}/classification-window"
        params = {
            "ticker_key": ticker_key,
            "model_type": model_type,
            "days_limit": days_limit
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"UI: Error fetching classification window for {model_type}: {str(e)}")
        return {"error": str(e)}

# Display regression predictions
if REGRESSION_MODELS_TO_DISPLAY:
    reg_cols = st.columns(len(REGRESSION_MODELS_TO_DISPLAY))
    for i, model_name in enumerate(REGRESSION_MODELS_TO_DISPLAY):
        with reg_cols[i]:
            st.subheader(f"{model_name.replace('_',' ').title()}")
            prediction_data = fetch_latest_prediction(selected_ticker_key_api, model_name)
            
            if prediction_data and "error" not in prediction_data:
                if prediction_data.get("predicted_close_price") is not None:
                    pred_date_obj = datetime.strptime(prediction_data.get("prediction_date"), '%Y-%m-%d')
                    st.metric(
                        label=f"Predicted Close on {pred_date_obj.strftime('%b %d, %Y')}",
                        value=f"${prediction_data.get('predicted_close_price'):,.2f}"
                    )
                    value_type = prediction_data.get("predicted_value_type", "absolute_close")
                    if "percentage_change" in value_type:
                        st.caption(f"Derived from predicted % change")
                    else:
                         st.caption(f"Direct price prediction")
                else:
                    st.warning("Prediction data received, but price is missing.")
            elif prediction_data and "error" in prediction_data:
                 st.error(prediction_data["error"], icon="🚨")
            else:
                st.error("Failed to fetch prediction data.", icon="❓")
else:
    st.info("No regression models configured for display.")


# Display Classification Models
if CLASSIFICATION_MODELS_TO_DISPLAY:
    st.subheader("📈 Price Direction (Classification)")
    class_cols = st.columns(len(CLASSIFICATION_MODELS_TO_DISPLAY))
    for i, model_name_class in enumerate(CLASSIFICATION_MODELS_TO_DISPLAY):
        with class_cols[i]:
            st.markdown(f"**{model_name_class.replace('_',' ').title()}**")
            
            classification_data = fetch_latest_prediction(selected_ticker_key_api, model_name_class, problem_type="classification")
            
            if classification_data and "error" not in classification_data:
                if classification_data.get("predicted_direction"):
                    direction = classification_data.get("predicted_direction")
                    confidence = classification_data.get("confidence_score", 0.5)
                    window = classification_data.get("prediction_window", 30)
                    pred_date_obj = datetime.strptime(classification_data.get("prediction_date"), '%Y-%m-%d')
                    
                    if direction == "up":
                        st.markdown("<h3 style='text-align: center; color: lightgreen;'>▲ UP</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h3 style='text-align: center; color: salmon;'>▼ DOWN</h3>", unsafe_allow_html=True)
                    
                    st.caption(f"Prediction for {pred_date_obj.strftime('%b %d, %Y')}")
                    st.caption(f"Confidence: {confidence:.2f}")
                    st.caption(f"{window}-day window")
                    
                    # Add classification history window
                    with st.expander(f"Show next {window} days predictions"):
                        window_data = fetch_classification_window(selected_ticker_key_api, model_name_class, window)
                        
                        if window_data and "error" not in window_data and window_data.get("data"):
                            days_data = window_data.get("data", [])
                            
                            if days_data:
                                # Create a mini-calendar style display
                                cols_per_row = 5
                                rows_needed = (len(days_data) + cols_per_row - 1) // cols_per_row
                                
                                for row in range(rows_needed):
                                    mini_cols = st.columns(cols_per_row)
                                    for col in range(cols_per_row):
                                        idx = row * cols_per_row + col
                                        if idx < len(days_data):
                                            day_data = days_data[idx]
                                            direction = day_data.get("predicted_direction")
                                            date_obj = datetime.strptime(day_data.get("prediction_date"), '%Y-%m-%d')
                                            
                                            with mini_cols[col]:
                                                st.markdown(f"**{date_obj.strftime('%b %d')}**")
                                                if direction == "up":
                                                    st.markdown("🟢 UP", help=f"Confidence: {day_data.get('confidence_score', 0.5):.2f}")
                                                else:
                                                    st.markdown("🔴 DOWN", help=f"Confidence: {day_data.get('confidence_score', 0.5):.2f}")
                            else:
                                st.info("No future predictions available yet")
                        else:
                            st.info("No future predictions available yet")
                else:
                    st.warning("Classification data received, but direction is missing.")
            elif classification_data and "error" in classification_data:
                st.error(classification_data["error"], icon="🚨")
            else:
                st.error("Failed to fetch classification data.", icon="❓")
else:
    st.caption("No classification models currently configured.")

st.markdown("---")

# --- Section 2: Historical Predictions Chart & Table (for Regression Models) ---
st.header(f"📊 Historical Predictions for {selected_ticker_display_name} (Regression)")

@st.cache_data(ttl=360, show_spinner="Loading historical data...")
def load_history_data_from_api(ticker_key_for_api: str, days_limit: int = 365):
    print(f"UI: Fetching history for {ticker_key_for_api}, limit {days_limit} days from {FASTAPI_URL}")
    try:
        history_url = f"{FASTAPI_URL}/history"
        params = {"ticker_key": ticker_key_for_api, "days_limit": days_limit}
        response = requests.get(history_url, params=params, timeout=30)
        response.raise_for_status()
        history_api_response = response.json()
        history_data = history_api_response.get("data", [])

        if not history_data:
            return pd.DataFrame(columns=['prediction_date', 'predicted_price', 'actual_price', 'model_used'])
        
        df = pd.DataFrame(history_data)
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df.set_index('prediction_date', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        if 'predicted_price' in df.columns:
            df['predicted_price'] = pd.to_numeric(df['predicted_price'], errors='coerce')
        if 'actual_price' in df.columns:
            df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
        return df
    except requests.exceptions.HTTPError as http_err:
        st.error(f"API Error loading history ({ticker_key_for_api}): {http_err.response.status_code}", icon="🚨")
        try: st.error(f"Details: {http_err.response.json().get('detail', http_err.response.text)}")
        except: st.error(f"Details: {http_err.response.text}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Request Error loading history ({ticker_key_for_api}): {req_err}", icon="🌐")
    except Exception as e:
        st.error(f"Error loading prediction history for {ticker_key_for_api}: {e}", icon="🔥")
    return pd.DataFrame(columns=['prediction_date', 'predicted_price', 'actual_price', 'model_used'])


days_of_history_slider = st.slider(
    "Days of History to Display:", min_value=7, max_value=365,
    value=180, step=30, key=f"history_days_slider_ui_v2_{selected_ticker_key_api}"
)
history_df_all_models = load_history_data_from_api(selected_ticker_key_api, days_limit=days_of_history_slider)

if not history_df_all_models.empty:
    history_df_regression = history_df_all_models[
        history_df_all_models['model_used'].isin(REGRESSION_MODELS_TO_DISPLAY)
    ].copy()

    if not history_df_regression.empty:
        unique_models_in_history_reg = sorted(history_df_regression['model_used'].dropna().unique())
        if not unique_models_in_history_reg: unique_models_in_history_reg = REGRESSION_MODELS_TO_DISPLAY

        selected_models_for_chart = st.multiselect(
            "Show predictions on chart from models:",
            options=unique_models_in_history_reg,
            default=unique_models_in_history_reg,
            key=f"hist_model_filter_ui_v2_{selected_ticker_key_api}"
        )

        chart_data_to_plot = history_df_regression.copy()

        fig = go.Figure()
        actual_prices_series = chart_data_to_plot['actual_price'].dropna().sort_index()
        if not actual_prices_series.empty:
            unique_actual_prices = actual_prices_series.groupby(actual_prices_series.index).first()
            fig.add_trace(go.Scatter(
                x=unique_actual_prices.index, y=unique_actual_prices,
                mode='lines', name='Actual Price', line=dict(color='deepskyblue', width=2),
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Actual: $%{y:,.2f}<extra></extra>'
            ))

        color_map = {"xgboost": "orange", "random_forest": "lightgreen", "lstm": "violet", "gru": "lightblue"}
        dash_map = {"xgboost": "solid", "random_forest": "dashdot", "lstm": "dot", "gru": "longdash"}

        for model_name_hist in selected_models_for_chart:
            model_pred_series = chart_data_to_plot[chart_data_to_plot['model_used'] == model_name_hist]['predicted_price'].dropna()
            if not model_pred_series.empty:
                fig.add_trace(go.Scatter(
                    x=model_pred_series.index, y=model_pred_series,
                    mode='lines', name=f'Pred. ({model_name_hist.replace("_"," ").title()})',
                    line=dict(color=color_map.get(model_name_hist, "grey"), dash=dash_map.get(model_name_hist, "longdash"), width=1.5),
                    hovertemplate=f'Date: %{{x|%Y-%m-%d}}<br>Pred. ({model_name_hist}): $ %{{y:,.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=f'Historical Comparison: Actual vs. Predicted Prices for {selected_ticker_display_name}',
            xaxis_title='Date', yaxis_title='Price (USD)', legend_title_text='Legend',
            height=550, hovermode="x unified", template="plotly_dark", # Using a dark theme
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display Table Data
        st.subheader("Historical Data Table")
        table_df_to_display = history_df_regression[
            history_df_regression['model_used'].isin(selected_models_for_chart)
        ][['model_used', 'predicted_price', 'actual_price']].copy()
        
        table_df_to_display.rename(columns={
            'model_used': 'Model Used',
            'predicted_price': 'Predicted Price',
            'actual_price': 'Actual Price'
        }, inplace=True)
        st.dataframe(table_df_to_display.sort_index(ascending=False).style.format({
            "Predicted Price": "${:,.2f}", "Actual Price": "${:,.2f}"}, na_rep="N/A"
        ), height=350)
    else:
        st.info(f"No regression model prediction history to display for {selected_ticker_display_name} for the selected period.")
else:
     st.warning(f"No prediction history available at all for {selected_ticker_display_name} for the last {days_of_history_slider} days. "
                "Data might still be populating or worker jobs need to run.")

st.markdown("---")

# Classification History and Accuracy Section
st.header(f"🔍 Classification Prediction History for {selected_ticker_display_name}")

@st.cache_data(ttl=360, show_spinner="Loading classification history...")
def load_classification_history_data_from_api(ticker_key_for_api: str, window: int = 30, days_limit: int = 180):
    print(f"UI: Fetching classification history for {ticker_key_for_api}, window {window} days from {FASTAPI_URL}")
    try:
        history_url = f"{FASTAPI_URL}/classification-history"
        params = {"ticker_key": ticker_key_for_api, "prediction_window": window, "days_limit": days_limit}
        response = requests.get(history_url, params=params, timeout=30)
        response.raise_for_status()
        history_api_response = response.json()
        history_data = history_api_response.get("data", [])

        if not history_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(history_data)
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df.set_index('prediction_date', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df
    except Exception as e:
        print(f"UI Error loading classification history: {e}")
        return pd.DataFrame()

st.markdown("---")
# --- Fetch Data Actions ---
# st.sidebar.header("⚙️ Fetch Data Actions")
# if st.sidebar.button("Trigger Full Data Update Pipeline", key="trigger_pipeline_button"):
#     with st.spinner("Requesting data pipeline trigger... This may take several minutes. Check server logs for progress."):
#         try:
#             trigger_url = f"{FASTAPI_URL}/trigger-data-update-all"
#             response = requests.post(trigger_url, timeout=20)
#             response.raise_for_status()
#             st.sidebar.success(response.json().get("message", "Data pipeline trigger request sent!"))
#             st.sidebar.info("Please wait a few minutes for data to update, then refresh the page or re-fetch history.")
#         except requests.exceptions.HTTPError as http_err:
#             detail = "Unknown API Error"
#             if http_err.response is not None:
#                 try: detail = http_err.response.json().get("detail", http_err.response.text)
#                 except: detail = http_err.response.text
#             st.sidebar.error(f"API Error triggering pipeline: {http_err.response.status_code} - {detail}", icon="🚨")
#         except Exception as e:
#             st.sidebar.error(f"Failed to trigger data pipeline: {e}", icon="🔥")

st.sidebar.markdown("---")