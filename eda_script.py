# Smart Irrigation Water Requirement Prediction
# Exploratory Data Analysis Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ───────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────
data_path   = r"C:\Users\Chris\Desktop\irrigation_prediction.csv"
project_dir = r"C:\Users\Chris\Desktop\irrigation system"
plots_dir   = os.path.join(project_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ───────────────────────────────────────────────
# Load Data
# ───────────────────────────────────────────────
try:
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
    exit(1)

# ───────────────────────────────────────────────
# Basic Statistics
# ───────────────────────────────────────────────
stats_file = os.path.join(project_dir, "eda_stats.txt")
with open(stats_file, "w") as f:
    f.write("=" * 50 + "\n")
    f.write("Dataset Information:\n")
    f.write("=" * 50 + "\n")
    df.info(buf=f)
    f.write("\n\nDescriptive Statistics:\n")
    f.write("=" * 50 + "\n")
    f.write(df.describe().to_string())
    f.write("\n\nClass Distribution (Irrigation_Need):\n")
    f.write("=" * 50 + "\n")
    f.write(df['Irrigation_Need'].value_counts().to_string())
    f.write("\n\nMissing Values:\n")
    f.write("=" * 50 + "\n")
    f.write(df.isnull().sum().to_string())

print(f"Stats saved to: {stats_file}")

# ───────────────────────────────────────────────
# Visualisations
# ───────────────────────────────────────────────
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 1. Target Distribution
plt.figure(figsize=(8, 5))
df['Irrigation_Need'].value_counts().plot(kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'])
plt.title("Irrigation Need Distribution", fontsize=14)
plt.xlabel("Irrigation Level")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "target_distribution.png"))
plt.close()
print("Target distribution plot saved.")

# 2. Histograms for Numerical Features
if numerical_cols:
    df[numerical_cols].hist(bins=20, figsize=(18, 12), color='steelblue', edgecolor='white')
    plt.suptitle("Histograms of Numerical Features", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(plots_dir, "histograms.png"))
    plt.close()
    print("Histograms saved.")

# 3. Correlation Heatmap
if len(numerical_cols) > 1:
    plt.figure(figsize=(14, 10))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"))
    plt.close()
    print("Correlation matrix saved.")

# 4. Boxplots — Numerical Features vs Irrigation Need
for col in ['Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 'Humidity']:
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        df.boxplot(column=col, by='Irrigation_Need')
        plt.title(f"{col} by Irrigation Need")
        plt.suptitle("")
        plt.xlabel("Irrigation Need")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"boxplot_{col}.png"))
        plt.close()

print("Boxplots saved.")

# 5. Pairplot (sample if large)
sample_df = df.sample(min(500, len(df)), random_state=42)
sns.pairplot(sample_df[numerical_cols[:5] + ['Irrigation_Need']], hue='Irrigation_Need', diag_kind='kde')
plt.suptitle("Pairplot (sample)", y=1.02, fontsize=13)
plt.savefig(os.path.join(plots_dir, "pairplot_sample.png"))
plt.close()
print("Pairplot saved.")

print("\nEDA script completed successfully.")
