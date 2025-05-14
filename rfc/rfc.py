import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period, adjust=False).mean()
    
    rs = avg_gain/avg_loss
    rsi = 100 - (100 / (1+rs))
    
    return rsi

def MACD(series):
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def ROC(series, period=14):
    return (series - series.shift(period)) * 100 / series.shift(period)

def STO_OS(close, low, high, period=14):
    low_14 = low.rolling(period).min()
    high_14 = high.rolling(period).max()
    return (close - low_14) * 100 / (high_14 - low_14)

def CCI(close, low, high, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mean_dev = tp.rolling(window=period).apply(lambda x: (abs(x - x.mean())).mean())
    
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    return cci

def DIX(close, period=14, method='SMA'):
    if method.upper() == 'SMA':
        ma = close.rolling(window=period).mean()
    elif method.upper() == 'EMA':
        ma = close.ewm(span=period, adjust=False).mean()
    else:
        raise ValueError('Method must be SMA or EMA')
    
    dix = (close - ma) * 100 / ma
    return dix

def ATR(close, low, high, period=14):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, adjust=False).mean()

    return atr

def OBV(close, volume):
    price_change_sign = np.sign(close.diff().fillna(0))
    signed_volume = volume * price_change_sign
    obv = signed_volume.cumsum()
    obv.name = 'OBV'
    return obv

def CMF(close, low, high, volume, period=20):
    mfm = ((2 * close - high - low) / (high - low)).fillna(0)  
    mfv = mfm * volume 

    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf

def main():
    stockdt = pd.read_csv('../data/sp500_stock_prices_2000_2025.csv', header=[0,1], index_col=0, parse_dates=True)
    
    alpha = 0.012
    stockdt['Smooth'] = stockdt['Close'].ewm(alpha=alpha, adjust=False).mean()
    
    stockdt['RSI'] = RSI(stockdt['Close'])
    stockdt['MACD'] = MACD(stockdt['Close'])[0]
    stockdt['ROC'] = ROC(stockdt['Close'])
    stockdt['%K'] = STO_OS(stockdt['Close'], stockdt['Low'], stockdt['High'])
    stockdt['CCI'] = CCI(stockdt['Close'], stockdt['Low'], stockdt['High'])
    stockdt['DIX'] = DIX(stockdt['Close'])

    stockdt['ATR'] = ATR(stockdt['Close'], stockdt['Low'], stockdt['High'])

    stockdt['OBV'] = OBV(stockdt['Close'], stockdt['Volume'])
    stockdt['CMF'] = CMF(stockdt['Close'], stockdt['Low'], stockdt['High'], stockdt['Volume'])

    stockdt.dropna(axis=0, inplace=True)
    
    trading_window = [10, 15, 30, 60, 90]
    for window in trading_window:
        stockdt[f'Smooth_{window}'] = stockdt['Close'].ewm(span=window, adjust=False).mean()
        stockdt[f'Target_{window}'] = np.sign(stockdt[f'Smooth_{window}'].shift(-window) - stockdt[f'Smooth_{window}'])
    stockdt.dropna(axis=0, inplace=True)
    
    
    test_size = 1000
    train_idx = stockdt.index[:len(stockdt)-test_size]
    test_idx = stockdt.index[len(stockdt)-test_size:]

    X = stockdt.loc[:, 'RSI':'CMF']
    y = stockdt.loc[:, [f'Target_{window}' for window in trading_window]]

    X_train = X.loc[train_idx, :]
    X_test = X.loc[test_idx, :]

    y_train = {}
    y_test = {}
    for window in trading_window:
        y_train[window] = y.loc[train_idx, f'Target_{window}']
        y_test[window] = y.loc[test_idx, f'Target_{window}']


    grid_rf = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [3, 5, 7, 9, None],  
    'max_features': [3, 4, 5]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    for window in trading_window:
        gscv = GridSearchCV(estimator=RandomForestClassifier(random_state=36, n_jobs=-1), param_grid=grid_rf, cv=tscv, scoring='accuracy')
        gscv_fit = gscv.fit(X_train, y_train[window])
        best_parameters = gscv_fit.best_params_
        print("Hyperparameter: ", best_parameters)
        model = RandomForestClassifier(**best_parameters, random_state=36)
        model.fit(X_train, y_train[window])
        models.append(model)
    
    print('************************************')
    print()
    y_pred = {}
    accuracy = {}
    precision = {}
    recall = {}
    f1 = {}
    confusion_mat = {}
    for i, window in enumerate(trading_window):
        value_counts = pd.Series(y_test[window]).value_counts()

        proportion_1 = value_counts.get(1, 0) / len(y_test[window])
        proportion_minus_1 = value_counts.get(-1, 0) / len(y_test[window])
        print("- Proportion of 1s:", proportion_1)
        print("- Proportion of -1s:", proportion_minus_1)
        
        
        y_pred[window] = models[i].predict(X_test)
        accuracy[window] = accuracy_score(y_test[window], y_pred[window])
        precision[window] = precision_score(y_test[window], y_pred[window], average='macro')
        recall[window] = recall_score(y_test[window], y_pred[window], average='macro')
        f1[window] = f1_score(y_test[window], y_pred[window], average='macro')
        confusion_mat[window] = confusion_matrix(y_test[window], y_pred[window])

        print(f"{window}_day Trading Window")
        print("- Accuracy:", accuracy[window])
        print("- Precision:", precision[window])
        print("- Recall:", recall[window])
        print("- F1 Score:", f1[window])
        print("- Confusion Matrix:")
        print(confusion_mat[window])
        
        print('----------------------------------')
        
    for i, window in enumerate(trading_window):
        retrain_model = models[i].fit(X, y.loc[:,f'Target_{window}'])
        with open(f'rfc_sp500_{window}.pkl', 'wb') as f:
            pickle.dump(retrain_model, f)
            
if __name__ == '__main__':
    main()