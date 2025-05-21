import numpy as np
import pandas as pd

def RSI(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period, adjust=False).mean()
    
    rs = avg_gain/avg_loss
    rsi = 100 - (100 / (1+rs))
    
    return rsi

def MACD(series):
    """Calculate Moving Average Convergence Divergence"""
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def ROC(series, period=14):
    """Calculate Rate of Change"""
    return (series - series.shift(period)) * 100 / series.shift(period)

def STO_OS(close, low, high, period=14):
    """Calculate Stochastic Oscillator %K"""
    low_14 = low.rolling(period).min()
    high_14 = high.rolling(period).max()
    return (close - low_14) * 100 / (high_14 - low_14)

def CCI(close, low, high, period=20):
    """Calculate Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mean_dev = tp.rolling(window=period).apply(lambda x: (abs(x - x.mean())).mean())
    
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    return cci

def DIX(close, period=14, method='SMA'):
    """Calculate Distance Index"""
    if method.upper() == 'SMA':
        ma = close.rolling(window=period).mean()
    elif method.upper() == 'EMA':
        ma = close.ewm(span=period, adjust=False).mean()
    else:
        raise ValueError('Method must be SMA or EMA')
    
    dix = (close - ma) * 100 / ma
    return dix

def ATR(close, low, high, period=14):
    """Calculate Average True Range"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

def OBV(close, volume):
    """Calculate On-Balance Volume"""
    if volume.sum() == 0:
        return pd.Series(0, index=close.index)
        
    price_change_sign = np.sign(close.diff().fillna(0))
    signed_volume = volume * price_change_sign
    obv = signed_volume.cumsum()
    return obv

def CMF(close, low, high, volume, period=20):
    """Calculate Chaikin Money Flow"""
    if volume.sum() == 0:
        return pd.Series(0, index=close.index)
        
    try:
        # Avoid division by zero
        range_diff = high - low
        range_diff = range_diff.replace(0, np.nan)  
        
        mfm = ((2 * close - high - low) / range_diff).fillna(0)
        mfv = mfm * volume
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum().replace(0, np.nan).fillna(1)
        return cmf.fillna(0)
    except Exception as e:
        print(f"Error calculating CMF: {e}")
        return pd.Series(0, index=close.index)

def calculate_technical_indicators(df):
    """Calculate all technical indicators for a given dataframe"""
    result = pd.DataFrame(index=df.index)
    
    # Basic indicators that don't need volume
    result['RSI'] = RSI(df['Close'])
    result['MACD'] = MACD(df['Close'])[0]
    result['ROC'] = ROC(df['Close'])
    result['%K'] = STO_OS(df['Close'], df['Low'], df['High'])
    result['CCI'] = CCI(df['Close'], df['Low'], df['High'])
    result['DIX'] = DIX(df['Close'])
    result['ATR'] = ATR(df['Close'], df['Low'], df['High'])
    
    # Volume-based indicators (handle possible zeros)
    result['OBV'] = OBV(df['Close'], df['Volume'])
    result['CMF'] = CMF(df['Close'], df['Low'], df['High'], df['Volume'])
    
    # Replace any NaNs or infinities with zeros
    result.replace([np.inf, -np.inf], 0, inplace=True)
    
    return result