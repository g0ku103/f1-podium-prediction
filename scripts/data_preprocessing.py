import pandas as pd
import numpy as np

# Load the merged dataset
df = pd.read_csv('data/f1_race_results.csv')

# Load additional datasets
try:
    drivers = pd.read_csv('data/f1-data/drivers.csv')
    circuits = pd.read_csv('data/f1-data/circuits.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure drivers.csv and circuits.csv are in 'data/f1-data/'.")
    raise

# --- Data Cleaning ---
# Handle placeholders in final_position
# Convert to numeric, replacing non-numeric (e.g., '99', 'et') with NaN
df['final_position'] = pd.to_numeric(df['final_position'], errors='coerce')
# Replace NaN in final_position with 99 (for non-finishes/DNFs)
df['final_position'] = df['final_position'].fillna(99).astype(int)

# Handle qualifying_position
# Ensure numeric, replace \N or invalid values with median
df['qualifying_position'] = pd.to_numeric(df['qualifying_position'], errors='coerce')
median_grid = df['qualifying_position'].median()
df['qualifying_position'] = df['qualifying_position'].fillna(median_grid).astype(int)

# Handle missing driver codes
# If driver code is \N or NaN, create a code from forename + surname
df['driver'] = df['driver'].replace(r'\\N', np.nan)
df['driver'] = df['driver'].fillna(df['driver'].apply(lambda x: 'UNKNOWN' if pd.isna(x) else x))

# Handle missing constructors (should be rare)
df['constructor'] = df['constructor'].fillna('Unknown')

# Verify no remaining \N or suspicious placeholders
for col in ['driver', 'constructor', 'race_name', 'final_position', 'qualifying_position']:
    if df[col].astype(str).str.contains(r'\\N|et', na=False).any():
        print(f"Warning: '{col}' still contains \\N or 'et'")

# --- Feature Engineering ---
# Merge with drivers.csv to get nationality
df = df.merge(drivers[['driverId', 'nationality']], on='driverId', how='left')

# Merge with circuits.csv to get circuit country
df = df.merge(circuits[['circuitId', 'country']], on='circuitId', how='left')

# Create is_home_race feature
# Map nationalities to countries (simplified mapping for common nationalities)
nationality_to_country = {
    'British': 'UK',
    'Italian': 'Italy',
    'German': 'Germany',
    'French': 'France',
    'Spanish': 'Spain',
    'Dutch': 'Netherlands',
    'Finnish': 'Finland',
    'Australian': 'Australia',
    'Brazilian': 'Brazil',
    'Mexican': 'Mexico',
    # Add more mappings as needed based on drivers.csv
}
df['nationality_country'] = df['nationality'].map(nationality_to_country).fillna('Other')
df['is_home_race'] = (df['nationality_country'] == df['country']).astype(int)
df = df.drop(['nationality', 'nationality_country', 'country'], axis=1)  # Drop temporary columns

# 1. Driver podium rate (historical podiums / total races for each driver)
df['driver_podium_rate'] = df.groupby('driver')['podium'].transform(lambda x: x.rolling(window=len(x), min_periods=1).mean().shift(1))
df['driver_podium_rate'] = df['driver_podium_rate'].fillna(0)  # Fill NaN for first race

# 2. Constructor podium rate (historical podiums / total races for each constructor)
df['constructor_podium_rate'] = df.groupby('constructor')['podium'].transform(lambda x: x.rolling(window=len(x), min_periods=1).mean().shift(1))
df['constructor_podium_rate'] = df['constructor_podium_rate'].fillna(0)

# --- Final Checks ---
# Ensure numeric columns are correct
df['qualifying_position'] = df['qualifying_position'].astype(int)
df['final_position'] = df['final_position'].astype(int)
df['podium'] = df['podium'].astype(int)

# Drop any duplicate rows
df = df.drop_duplicates()

# Save cleaned dataset
output_path = 'data/f1_cleaned_data.csv'
df.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")
print("Dataset shape:", df.shape)
print("\nSample data (first 5 rows):")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())