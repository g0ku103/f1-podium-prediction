import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import os
import joblib

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

#Load cleaned dataset
df = pd.read_csv('data/f1_cleaned_data.csv')

#Data Preparation
#Select features and target
features = ['qualifying_position', 'driver_podium_rate', 'constructor_podium_rate', 'is_home_race']
X= df[features]
y=df['podium']

#Split data = train(2010-2020), test (2021-2024)
train_mask = df['season']<=2020
test_mask = df['season'] > 2020
X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

#Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames to preserve feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

# Save scaler for deployment
joblib.dump(scaler, 'results/scaler.joblib')


# --- Model Training ---
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

results = []
roc_data = {}

#Training Model

for name,model in models.items():
    #train
    model.fit(X_train,y_train)

    #predict
    y_pred=model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    
    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr)

# --- Save Metrics ---
results_df = pd.DataFrame(results)
results_df.to_csv('results/model_performance.csv', index=False)
print("\nModel Performance:")
print(results_df)

# --- Plot ROC Curves ---
plt.figure(figsize=(10, 6))
for name, (fpr, tpr) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, models[name].predict_proba(X_test_scaled)[:, 1]):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Podium Prediction Models')
plt.legend()
plt.grid(True)
plt.savefig('results/roc_curves.png')
plt.close()

# --- Plot Feature Importance (Random Forest and XGBoost) ---
# Random Forest
rf_model = models['Random Forest']
plt.figure(figsize=(10, 6))
plt.bar(features, rf_model.feature_importances_, color='blue')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/feature_importance_rf.png')
plt.close()


# XGBoost
xgb_model = models['XGBoost']
plt.figure(figsize=(10, 6))
plt.bar(features, xgb_model.feature_importances_, color='green')
plt.title('Feature Importance (XGBoost)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/feature_importance_xgb.png')
plt.close()

print("\nResults saved:")
print("- Metrics: results/model_performance.csv")
print("- ROC Curves: results/roc_curves.png")
print("- Feature Importance (RF): results/feature_importance_rf.png")
print("- Feature Importance (XGB): results/feature_importance_xgb.png")