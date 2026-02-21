import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import os

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

# Encode Categorical Features
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Feature Engineering
df['Moisture_Temperature_Ratio'] = df['Soil_Moisture'] / df['Temperature_C']

X = df.drop(target_col, axis=1)
y = df[target_col]

# Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Cross-Validation (K-Fold)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

print(f"\nRunning {k}-Fold Cross-Validation...")
# R2 Scores
r2_scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')

# Negative MSE (sklearn uses negative so higher is better, we flip it back for readability)
neg_mse_scores = cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_squared_error')
mse_scores = -neg_mse_scores
rmse_scores = np.sqrt(mse_scores)

print("\nCross-Validation Results:")
print("-" * 30)
print(f"R2 Scores per fold: {r2_scores}")
print(f"Mean R2: {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
print("-" * 30)
print(f"RMSE Scores per fold: {rmse_scores}")
print(f"Mean RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
print("-" * 30)

# Save results to file
results_path = os.path.join(output_dir, "cv_results.txt")
with open(results_path, "w") as f:
    f.write(f"{k}-Fold Cross-Validation Results for Random Forest:\n\n")
    f.write(f"R2 Scores: {r2_scores}\n")
    f.write(f"Mean R2: {r2_scores.mean():.4f}\n")
    f.write(f"Std R2: {r2_scores.std():.4f}\n\n")
    f.write(f"RMSE Scores: {rmse_scores}\n")
    f.write(f"Mean RMSE: {rmse_scores.mean():.4f}\n")
    f.write(f"Std RMSE: {rmse_scores.std():.4f}\n")

# Visualization of Fold Performance
plt.figure(figsize=(10, 6))
plt.boxplot([r2_scores], patch_artist=True, labels=['R2 Score'])
plt.title(f'{k}-Fold Cross-Validation R2 Score Distribution')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "cv_r2_boxplot.png"))
plt.close()
print("Boxplot saved.")

print("Cross-Validation complete.")
