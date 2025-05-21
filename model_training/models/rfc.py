import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

class StockPriceRFC:
    """Random Forest Classifier for stock price direction prediction"""
    def __init__(self, n_estimators=200, max_depth=5, max_features=4, random_state=36):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_importance_ = None
        self.required_features = [
            'RSI', 'MACD', 'ROC', '%K', 'CCI', 'DIX', 'ATR', 'OBV', 'CMF'
        ]
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.feature_importance_ = self.model.feature_importances_
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)