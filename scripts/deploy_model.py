import pandas as pd
import joblib

# Load the scaler and best model (XGBoost)
scaler = joblib.load('results/scaler.joblib')
model = joblib.load('results/xgboost_model.joblib')