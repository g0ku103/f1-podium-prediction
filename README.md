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
## Modeling
- **Script**: `scripts/modeling.py`
- **Input**: `data/f1_cleaned_data.csv`
- **Output**:
  - `results/model_performance.csv`: Accuracy, precision, recall, F1-score, ROC-AUC for logistic regression, random forest, XGBoost.
  - `results/roc_curves.png`: ROC curves for all models.
  - `results/feature_importance_rf.png`: Feature importance for random forest.
  - `results/feature_importance_xgb.png`: Feature importance for XGBoost.
- **Details**:
  - Features: `qualifying_position`, `driver_podium_rate`, `constructor_podium_rate`, `is_home_race`.
  - Train: 2010–2020, Test: 2021–2024.
  - Models: Logistic regression (simple), random forest (robust), XGBoost (high performance).
  ## Modeling
- **Script**: `scripts/modeling.py`
- **Input**: `data/f1_cleaned_data.csv`
- **Output**:
  - `results/model_performance.csv`: Accuracy, precision, recall, F1-score, ROC-AUC.
  - `results/roc_curves.png`: ROC curves.
  - `results/feature_importance_rf.png`, `results/feature_importance_xgb.png`: Feature importance.
  - `results/scaler.joblib`, `results/*_model.joblib`: Saved scaler and models.
- **Details**:
  - Features: `qualifying_position`, `driver_podium_rate`, `constructor_podium_rate`, `is_home_race`.
  - Train: 2010–2020, Test: 2021–2024.
  - Models: Logistic regression (F1: 0.491), random forest (F1: 0.283), XGBoost (F1: 0.538, best).
- **Analysis**: XGBoost outperforms due to highest F1-Score and accuracy. `qualifying_position` and `driver_podium_rate` are likely key predictors (see feature importance plots).

## Deployment
- **Script**: `scripts/deploy_model.py`
- **Input**: `data/F1_2025_Dataset/F1_2025_QualifyingResults.csv`, historical data, `results/scaler.joblib`, `results/xgboost_model.joblib`
- **Output**:
  - `results/2025_predictions.csv`: Predicted podiums for 2025 Australia race.
  - `results/shap_summary_plot.png`: SHAP feature importance.
  - `results/top_5_podium_probabilities.png`: Seaborn bar plot of top 5 podium probabilities.
- **Details**: Uses XGBoost with 2025 qualifying data. Handles 'NC' by exclusion and adds Seaborn for better interpretability.
- **Insights**: Predictions should favor top qualifiers; compare with hypothetical real podium.
- **Future Work**: Add weather, refine model for better accuracy.
