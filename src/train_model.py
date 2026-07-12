import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, params=None):
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def tune_xgboost(X_train, y_train, param_grid=None, cv=3):
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
    
    xgb_base = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    
    grid_search = GridSearchCV(
        xgb_base,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f'Model saved to {filepath}')


def load_model(filepath):
    model = joblib.load(filepath)
    print(f'Model loaded from {filepath}')
    return model
