import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# Setup
data_path = r"C:\Users\Chris\Desktop\irrigation_prediction.csv"
output_dir = r"C:\Users\Chris\Desktop\irrigation system"
models_dir = os.path.join(output_dir, "models")
os.makedirs(models_dir, exist_ok=True)

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

# Encode Categorical Features and save encoders
categorical_cols = df.select_dtypes(include=['object']).columns
encoders = {}

print("Encoding categorical features...")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Feature Engineering
df['Moisture_Temperature_Ratio'] = df['Soil_Moisture'] / df['Temperature_C']

# Prepare Data
X = df.drop(target_col, axis=1)
y = df[target_col]

# Train Final Model
print("Training final Random Forest Regressor on full dataset...")
rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
rf_final.fit(X, y)

# Save Artifacts
model_path = os.path.join(models_dir, "rf_irrigation_model.pkl")
encoders_path = os.path.join(models_dir, "encoders.pkl")

print(f"Saving model to {model_path}...")
joblib.dump(rf_final, model_path)

print(f"Saving encoders to {encoders_path}...")
joblib.dump(encoders, encoders_path)

print("Model deployment artifacts created successfully.")
