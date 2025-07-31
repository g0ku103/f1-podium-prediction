import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# Initialize SHAP JavaScript for force plots
shap.initjs()

# Load historical and 2025 data
historical_data = pd.read_csv('data/f1_cleaned_data.csv')
drivers = pd.read_csv('data/f1-data/drivers.csv')
circuits = pd.read_csv('data/f1-data/circuits.csv')
qualifying_2025 = pd.read_csv('data/F1_2025_Dataset/F1_2025_QualifyingResults.csv')

# Check unique tracks and set race_track
print("Available tracks in 2025 data:", qualifying_2025['Track'].unique())
race_track = 'Australia'  # Using Australia as requested
new_data = qualifying_2025[qualifying_2025['Track'] == race_track].copy()

# Check if new_data is empty
if new_data.empty:
    print(f"No data found for track '{race_track}'. Please check available tracks or data file.")
else:
    # --- Feature Engineering ---
    # Map columns: Position -> qualifying_position, Driver -> driver, Team -> constructor
    new_data = new_data.rename(columns={
        'Position': 'qualifying_position',
        'Driver': 'driver',
        'Team': 'constructor'
    })

    # Handle 'NC' in qualifying_position by excluding those drivers
    new_data['qualifying_position'] = pd.to_numeric(new_data['qualifying_position'], errors='coerce')
    new_data = new_data.dropna(subset=['qualifying_position'])  # Remove rows with NaN (from 'NC')

    # Merge with drivers to get nationality
    new_data = new_data.merge(drivers[['code', 'nationality']], left_on='driver', right_on='code', how='left')
    new_data['driver'] = new_data['driver'].fillna(new_data['code'])  # Use code if driver name missing

    # Merge with circuits to get country
    circuits_dict = circuits.set_index('circuitId')['country'].to_dict()
    track_to_circuit = {'Australia': '1'}  # Australia circuitId
    new_data['circuitId'] = new_data['Track'].map(track_to_circuit)
    new_data['circuitId'] = new_data['circuitId'].astype(int)
    new_data = new_data.merge(circuits[['circuitId', 'country']], on='circuitId', how='left')

    # Map nationalities to countries
    nationality_to_country = {
        'British': 'UK', 'Italian': 'Italy', 'German': 'Germany', 'French': 'France',
        'Spanish': 'Spain', 'Dutch': 'Netherlands', 'Finnish': 'Finland', 'Australian': 'Australia',
        'Brazilian': 'Brazil', 'Mexican': 'Mexico', 'Monegasque': 'Monaco', 'Canadian': 'Canada',
        'Austrian': 'Austria', 'Belgian': 'Belgium', 'Japanese': 'Japan', 'Russian': 'Russia',
        'Danish': 'Denmark', 'Swedish': 'Sweden', 'Polish': 'Poland', 'Chinese': 'China'
    }
    new_data['nationality_country'] = new_data['nationality'].map(nationality_to_country).fillna('Other')
    new_data['is_home_race'] = (new_data['nationality_country'] == new_data['country']).astype(int)

    # Compute driver_podium_rate from historical data
    driver_rates = historical_data.groupby('driver')['podium'].mean().reset_index()
    driver_rates.rename(columns={'podium': 'driver_podium_rate'}, inplace=True)
    new_data = new_data.merge(driver_rates, on='driver', how='left')
    new_data['driver_podium_rate'] = new_data['driver_podium_rate'].fillna(0)

    # Compute constructor_podium_rate from historical data
    constructor_rates = historical_data.groupby('constructor')['podium'].mean().reset_index()
    constructor_rates.rename(columns={'podium': 'constructor_podium_rate'}, inplace=True)
    new_data = new_data.merge(constructor_rates, on='constructor', how='left')
    new_data['constructor_podium_rate'] = new_data['constructor_podium_rate'].fillna(0)

    # --- Prepare Features ---
    features = ['qualifying_position', 'driver_podium_rate', 'constructor_podium_rate', 'is_home_race']
    X_new = new_data[features]

    # Load scaler and model
    scaler = joblib.load('results/scaler.joblib')
    model = joblib.load('results/xgboost_model.joblib')

    # Scale features and handle NaN
    X_new_scaled = scaler.transform(X_new)
    X_new_scaled = pd.DataFrame(X_new_scaled, columns=features, index=X_new.index)

    # --- Predict ---
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)[:, 1]

    # Add predictions to DataFrame
    new_data['podium_prediction'] = predictions
    new_data['podium_probability'] = probabilities

    # --- SHAP Explanations ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_new_scaled)

    # Debug Output
    print("Scaled data shape:", X_new_scaled.shape)
    print("First row of scaled data:", X_new_scaled.iloc[0].to_numpy() if not X_new_scaled.empty else "Empty scaled data")
    print("Number of SHAP values:", len(shap_values) if shap_values is not None and len(shap_values) > 0 else "No SHAP values")
    print("First SHAP value:", shap_values[0].tolist() if shap_values is not None and len(shap_values) > 0 else "No SHAP values or invalid")

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_new_scaled, feature_names=features, show=False)
    plt.title(f'SHAP Summary Plot for {race_track} 2025')
    plt.tight_layout()
    plt.savefig('results/shap_summary_plot.png')
    plt.close()

    # Bar plot for top 5 podium probabilities with Seaborn
    top_5 = new_data.nlargest(5, 'podium_probability')
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")  # Improved background
    bar_plot = sns.barplot(x='driver', y='podium_probability', data=top_5, color='skyblue', label='Predicted')
    # Hypothetical real podium (update with actual results)
    real_podium = {'VER': 0.9, 'HAM': 0.85, 'LEC': 0.8}  # Example values
    real_top_3 = [driver for driver in real_podium.keys() if driver in top_5['driver'].values]
    real_probs = [real_podium[driver] for driver in real_top_3]
    sns.barplot(x=real_top_3, y=real_probs, color='salmon', alpha=0.7, label='Real (Hypothetical)', ax=bar_plot)
    plt.xlabel('Driver')
    plt.ylabel('Podium Probability')
    plt.title(f'Top 5 Podium Probabilities for {race_track} 2025')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/top_5_podium_probabilities.png')
    plt.close()

    # --- Save Predictions ---
    new_data.to_csv('results/2025_predictions.csv', index=False)
    print(f"\n2025 {race_track} Predictions:")
    print(new_data[['driver', 'constructor', 'Track', 'podium_prediction', 'podium_probability']])
    print("\nPlots Saved:")
    print("- Summary Plot: results/shap_summary_plot.png")
    print("- Top 5 Podium Probabilities: results/top_5_podium_probabilities.png")