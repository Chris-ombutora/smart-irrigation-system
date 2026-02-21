# Smart Irrigation System — Feature Correlation Analysis
# Analyzes correlations between features and Irrigation_Need
# and engineers/validates the best predictors.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import os

# ───────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────
data_path   = r"C:\Users\Chris\Desktop\irrigation_prediction.csv"
project_dir = r"C:\Users\Chris\Desktop\irrigation system"
plots_dir   = os.path.join(project_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ───────────────────────────────────────────────
# Load & Encode Data
# ───────────────────────────────────────────────
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} rows.")

# Ordinal encode target
target_order = [['Low', 'Medium', 'High']]
oe = OrdinalEncoder(categories=target_order)
df['Irrigation_Need_Encoded'] = oe.fit_transform(df[['Irrigation_Need']])

# Label-encode all other categoricals
le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns.drop('Irrigation_Need').tolist()
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# ───────────────────────────────────────────────
# Feature Engineering
# ───────────────────────────────────────────────
df_encoded['Moisture_Temperature_Ratio'] = df_encoded['Soil_Moisture'] / df_encoded['Temperature_C']
df_encoded['Rain_Humidity_Index']        = df_encoded['Rainfall_mm'] * df_encoded['Humidity'] / 100
df_encoded['Sunlight_Wind_Ratio']        = df_encoded['Sunlight_Hours'] / (df_encoded['Wind_Speed_kmh'] + 1)

print("Engineered features: Moisture_Temperature_Ratio, Rain_Humidity_Index, Sunlight_Wind_Ratio")

# ───────────────────────────────────────────────
# Correlation with Target
# ───────────────────────────────────────────────
numeric_features = df_encoded.select_dtypes(include=np.number).columns.drop('Irrigation_Need_Encoded')
correlations = df_encoded[numeric_features].corrwith(df_encoded['Irrigation_Need_Encoded']).sort_values(ascending=False)

print("\nFeature Correlations with Irrigation_Need (Ordinal Encoded):")
print("=" * 55)
print(correlations.to_string())

# Save correlation results
corr_file = os.path.join(project_dir, "correlation_results.txt")
with open(corr_file, "w") as f:
    f.write("Feature Correlations with Irrigation_Need:\n")
    f.write("=" * 55 + "\n")
    f.write(correlations.to_string())
    f.write("\n\nEngineered Features:\n")
    f.write("  - Moisture_Temperature_Ratio = Soil_Moisture / Temperature_C\n")
    f.write("  - Rain_Humidity_Index        = Rainfall_mm * Humidity / 100\n")
    f.write("  - Sunlight_Wind_Ratio        = Sunlight_Hours / (Wind_Speed_kmh + 1)\n")

print(f"\nCorrelation results saved to: {corr_file}")

# ───────────────────────────────────────────────
# Correlation Bar Chart
# ───────────────────────────────────────────────
plt.figure(figsize=(12, 8))
colors = ['#e74c3c' if c > 0 else '#3498db' for c in correlations]
correlations.plot(kind='barh', color=colors, edgecolor='white')
plt.title("Feature Correlation with Irrigation Need", fontsize=14)
plt.xlabel("Pearson Correlation Coefficient")
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "feature_correlations.png"))
plt.close()
print("Correlation bar chart saved.")

# ───────────────────────────────────────────────
# Full Encoded Correlation Heatmap
# ───────────────────────────────────────────────
top_features = correlations.abs().nlargest(10).index.tolist() + ['Irrigation_Need_Encoded']
plt.figure(figsize=(12, 10))
sns.heatmap(df_encoded[top_features].corr(), annot=True, fmt=".2f", cmap='RdYlGn', linewidths=0.5)
plt.title("Top 10 Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "top_features_heatmap.png"))
plt.close()
print("Top features heatmap saved.")

print("\nCorrelation analysis completed successfully.")
