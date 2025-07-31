import pandas as pd
import joblib

# Load the scaler and best model (XGBoost)
scaler = joblib.load('results/scaler.joblib')
model = joblib.load('results/xgboost_model.joblib')

# Simulate 2025 race data (example: Monaco Grand Prix)
new_data = pd.DataFrame({
    'qualifying_position': [1, 2, 3, 4, 5],
    'driver_podium_rate': [0.35, 0.28, 0.15, 0.10, 0.05],
    'constructor_podium_rate': [0.40, 0.35, 0.20, 0.15, 0.10],
    'is_home_race': [1, 0, 0, 0, 0],
    'driver': ['LEC', 'VER', 'NOR', 'HAM', 'SAI'],
    'constructor': ['Ferrari', 'Red Bull', 'McLaren', 'Mercedes', 'Ferrari'],
    'race_name': ['Monaco Grand Prix'] * 5,
    'season': [2025] * 5
})
# Prepare features
features = ['qualifying_position', 'driver_podium_rate', 'constructor_podium_rate', 'is_home_race']
X_new = new_data[features]

# Scale features
X_new_scaled = scaler.transform(X_new)
X_new_scaled = pd.DataFrame(X_new_scaled, columns=features)

# Predict
predictions = model.predict(X_new_scaled)
probabilities = model.predict_proba(X_new_scaled)[:, 1]

# Add predictions to DataFrame
new_data['podium_prediction'] = predictions
new_data['podium_probability'] = probabilities

# Save predictions
new_data.to_csv('results/2025_predictions.csv', index=False)
print("\n2025 Race Predictions:")
print(new_data[['driver', 'constructor', 'race_name', 'podium_prediction', 'podium_probability']])



