# Smart Irrigation System — Water Conservation Overview
# Project: Optimize water usage in agriculture by predicting
#          irrigation need based on environmental conditions.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ───────────────────────────────────────────────
# Project Description
# ───────────────────────────────────────────────
# This is a Smart Irrigation Water Requirement Prediction Dataset.
# Goal: Predict whether a field needs Low / Medium / High irrigation
# by analysing soil, climate, crop, and environmental variables.

# ───────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────
data_path   = r"C:\Users\Chris\Desktop\irrigation_prediction.csv"
project_dir = r"C:\Users\Chris\Desktop\irrigation system"
plots_dir   = os.path.join(project_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ───────────────────────────────────────────────
# Load Dataset
# ───────────────────────────────────────────────
water_data = pd.read_csv(data_path)

print("=" * 55)
print("SMART IRRIGATION WATER CONSERVATION PROJECT")
print("=" * 55)
print(f"\nDataset Shape : {water_data.shape}")
print(f"Columns       : {list(water_data.columns)}\n")

print("─── Descriptive Statistics ───")
print(water_data.describe())

print("\n─── Missing Values ───")
print(water_data.isnull().sum())

print("\n─── Target Distribution (Irrigation_Need) ───")
print(water_data['Irrigation_Need'].value_counts())

# ───────────────────────────────────────────────
# Water Savings Estimate
# ───────────────────────────────────────────────
print("\n─── Water Conservation Potential ───")
total_fields   = len(water_data)
low_need       = (water_data['Irrigation_Need'] == 'Low').sum()
medium_need    = (water_data['Irrigation_Need'] == 'Medium').sum()
high_need      = (water_data['Irrigation_Need'] == 'High').sum()

pct_low        = 100 * low_need    / total_fields
pct_medium     = 100 * medium_need / total_fields
pct_high       = 100 * high_need   / total_fields

print(f"  Low    : {low_need:>5} fields ({pct_low:.1f}%) — No irrigation needed")
print(f"  Medium : {medium_need:>5} fields ({pct_medium:.1f}%) — Half-cycle irrigation")
print(f"  High   : {high_need:>5} fields ({pct_high:.1f}%) — Full-cycle irrigation")
print(f"\n  Potential water saving by avoiding over-irrigation: {pct_low + pct_medium:.1f}% of fields")

# ───────────────────────────────────────────────
# Visualisation — Pie Chart of Conservation Impact
# ───────────────────────────────────────────────
labels  = ['Low (No Irrigation)', 'Medium (Half)', 'High (Full)']
sizes   = [pct_low, pct_medium, pct_high]
colors  = ['#2ecc71', '#f39c12', '#e74c3c']
explode = (0.05, 0.03, 0)

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
        explode=explode, startangle=140, shadow=True,
        textprops={'fontsize': 12})
plt.title("Irrigation Need Distribution\n(Water Conservation Overview)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "water_conservation_pie.png"))
plt.close()
print(f"\nPie chart saved to plots/water_conservation_pie.png")

# ───────────────────────────────────────────────
# Key Features Summary
# ───────────────────────────────────────────────
print("\n─── Key Soil & Climate Features ───")
key_cols = ['Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 'Humidity', 'Sunlight_Hours']
for col in key_cols:
    if col in water_data.columns:
        print(f"  {col:<25} mean={water_data[col].mean():.2f}  std={water_data[col].std():.2f}")

print("\nWater conservation overview complete.")
print("Run eda_script.py          → Full EDA")
print("Run correlation_analysis.py → Feature correlation")
print("Run train_models.py        → Model training & comparison")
print("Run train_rf_cv.py         → Cross-validation")
print("Run deploy_model.py        → Save final model")
print("Run test_system.py         → Predict on scenarios")
