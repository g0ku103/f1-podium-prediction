 # F1 Prediction Project
A machine learning project to predict Formula 1 podium finishes (top 3) based on qualifying position, driver, constructor, and track. Built with Python, pandas, scikit-learn, matplotlib, seaborn, and Streamlit for a web app interface.

## Folder Structure
- `data/`: Raw and processed datasets
- `scripts/`: Python scripts for data processing, modeling, and visualization
- `visualizations/`: Output plots and charts
- `app/`: Streamlit app code
- `notebooks/`: Jupyter notebooks for EDA and experimentation

## Setup Instructions
1. Clone the repository: `git clone https://github.com/your-username/f1-podium-prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the Kaggle dataset: [Formula 1 World Championship (1950�2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
4. Place the dataset in `data/formula-1-world-championship-1950-2020/`

## EDA Insights (Updated)
- **Dataset**: Formula 1 World Championship (1950–2024) in `data/f1-data-2024/`.
- **Key Files**: `results.csv`, `races.csv`, `drivers.csv`, `constructors.csv`.
- **Time Range**: 1950–2024, with focus on 2010–2024 for modeling.
- **Placeholders**:
  - `\N` in `results.position`/`positionText` (non-finishes), `results.grid` (missing qualifying data), and `drivers.code` (older drivers).
  - Possible `et` or other placeholders (needs investigation, may be typos).
  - `grid` may have zeros for missing qualifying data.
- **Next Steps**: Replace `\N` and `et` with appropriate values (e.g., high position for DNFs, median for grid), impute missing codes, and source weather data.

## Visualizations
- **Driver-Wise Podium Trends**: Bar chart showing top 10 drivers by podium finishes (2010–2024). Saved as `visualizations/driver_podiums.png`.
- **Qualifying vs. Final Position**: Line plot of average final position by qualifying position, showing correlation. Saved as `visualizations/qualifying_vs_final.png`.
- **Track-Wise Podium Distribution**: Bar chart of top 10 circuits by podium finishes. Saved as `visualizations/circuit_podiums.png`.
## Data Collection
- **Script**: `scripts/data_collection.py`
- **Input**: `results.csv`, `races.csv`, `drivers.csv`, `constructors.csv` from `data/f1-data-2024/`.
- **Output**: `data/f1_race_results.csv` with columns: `season`, `race_name`, `circuitId`, `driver`, `constructor`, `qualifying_position`, `final_position`, `podium`.
- **Details**:
  - Merged CSVs to create a unified dataset for 2010–2024 (hybrid era).
  - Handled `\N` and `et` placeholders minimally: `\N` and `et` in `final_position` replaced with '99', `\N` or 0 in `qualifying_position` replaced with median.
  - Created `podium` column (1 for top 3, 0 otherwise).
- **Next**: Clean remaining placeholders and engineer features in `scripts/data_preprocessing.py`.
## Data Collection
- **Script**: `scripts/data_collection.py`
- **Input**: `results.csv`, `races.csv`, `drivers.csv`, `constructors.csv` from `data/f1-data/`.
- **Output**: `data/f1_race_results.csv` with columns: `season`, `race_name`, `circuitId`, `driverId`, `constructorId`, `driver`, `constructor`, `qualifying_position`, `final_position`, `podium`.
- **Details**:
  - Merged CSVs for 2010–2024 (hybrid era).
  - Handled `\N` and `et` in `final_position` with '99', `\N` or 0 in `qualifying_position` with median.
  - Included `driverId` and `constructorId` for further merges.

## Data Cleaning & Feature Engineering
- **Script**: `scripts/data_preprocessing.py`
- **Input**: `data/f1_race_results.csv`, `drivers.csv`, `circuits.csv`
- **Output**: `data/f1_cleaned_data.csv` with columns: `season`, `race_name`, `circuitId`, `driverId`, `constructorId`, `driver`, `constructor`, `qualifying_position`, `final_position`, `podium`, `driver_podium_rate`, `constructor_podium_rate`, `is_home_race`.
- **Cleaning**:
  - Replaced `\N` and `et` in `final_position` with 99.
  - Filled NaN in `qualifying_position` with median.
  - Replaced `\N` or NaN in `driver` with 'UNKNOWN'.
- **Features**:
  - `driver_podium_rate`: Historical podium rate per driver.
  - `constructor_podium_rate`: Historical podium rate per constructor.
  - `is_home_race`: 1 if driver’s nationality matches circuit country.
- **Next**: Build ML models in `scripts/modeling.py`.