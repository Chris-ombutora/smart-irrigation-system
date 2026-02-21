import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import os

# Check for XGBoost and CatBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
    print("XGBoost not installed. Skipping XGBoost.")

try:
    from catboost import CatBoostRegressor
    cat_available = True
except ImportError:
    cat_available = False
    print("CatBoost not installed. Skipping CatBoost.")

# Setup
data_path = r"C:\Users\Chris\Desktop\irrigation_prediction.csv"
output_dir = r"C:\Users\Chris\Desktop\irrigation system"
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

print("Loading data...")
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
    exit(1)

# Preprocessing
target_col = 'Irrigation_Need'

# Ordinal Encode Target (Low=0, Medium=1, High=2)
target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df[target_col] = df[target_col].map(target_mapping)
print(f"Target Encoded: {target_mapping}")

# Encode Categorical Features
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Feature Engineering (Adding the one we found useful)
df['Moisture_Temperature_Ratio'] = df['Soil_Moisture'] / df['Temperature_C']

# Split Data
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (Optional but often good)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}

# 1. Random Forest
print("\nTraining Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train) # Tree models don't strictly need scaling
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'MSE': mean_squared_error(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'R2': r2_score(y_test, y_pred_rf),
    'Model': rf
}

# 2. XGBoost
if xgb_available:
    print("Training XGBoost...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results['XGBoost'] = {
        'MSE': mean_squared_error(y_test, y_pred_xgb),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'R2': r2_score(y_test, y_pred_xgb),
        'Model': xgb
    }

# 3. CatBoost
if cat_available:
    print("Training CatBoost...")
    # CatBoost handles categoricals well internally, but we already encoded them.
    # We'll use the numeric features for now to keep comparison fair.
    cb = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=42)
    cb.fit(X_train, y_train)
    y_pred_cb = cb.predict(X_test)
    results['CatBoost'] = {
        'MSE': mean_squared_error(y_test, y_pred_cb),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_cb)),
        'R2': r2_score(y_test, y_pred_cb),
        'Model': cb
    }

# Compare Results
print("\nModel Comparison:")
metrics_df = pd.DataFrame(results).T
print(metrics_df[['MSE', 'RMSE', 'R2']])

# Save Comparison Plot
plt.figure(figsize=(10, 6))
metrics_df['R2'].sort_values().plot(kind='barh')
plt.title("Model R2 Score Comparison")
plt.xlabel("R2 Score")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "model_comparison_r2.png"))
plt.close()
print("Comparison plot saved.")

# Feature Importance (from best model)
best_model_name = metrics_df['R2'].idxmax()
best_model = results[best_model_name]['Model']
print(f"\nBest Model: {best_model_name}")

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Feature Importances ({best_model_name})")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"))
    plt.close()
    print("Feature importance plot saved.")

# Save results to text
results_txt_path = os.path.join(output_dir, "model_results.txt")
with open(results_txt_path, "w") as f:
    f.write("Model Performance Metrics:\n")
    f.write(metrics_df[['MSE', 'RMSE', 'R2']].to_string())
    f.write(f"\n\nBest Model: {best_model_name}\n")

print("Training script completed.")
