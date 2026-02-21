# smart-irrigation-system
smart system used for automated irrigation and water conservation purposes using machine learning


# Step 1 — Overview
python water_conserve.py

# Step 2 — EDA
python eda_script.py

# Step 3 — Correlation Analysis
python correlation_analysis.py

# Step 4 — Train & Compare Models
python train_models.py

# Step 5 — Cross-Validate Random Forest
python train_rf_cv.py

# Step 6 — Deploy Final Model (saves .pkl files)
python deploy_model.py

# Step 7 — Test System with Real Scenarios
python test_system.py


Script Summaries
1. water_conserve.py
Loads dataset and prints shape, columns, statistics, class distributions
Estimates water savings potential based on Low/Medium/High need split
Generates plots/water_conservation_pie.png


3. eda_script.py
Generates descriptive stats → eda_stats.txt
Plots: histograms, correlation heatmap, pairplot, boxplots by irrigation level


5. correlation_analysis.py
Ordinal-encodes target (Low=0, Medium=1, High=2)
Engineers 3 new features:
Moisture_Temperature_Ratio = Soil_Moisture / Temperature_C
Rain_Humidity_Index = Rainfall_mm × Humidity / 100
Sunlight_Wind_Ratio = Sunlight_Hours / (Wind_Speed_kmh + 1)
Ranks all features by Pearson correlation with target
Generates correlation bar chart + top-10 feature heatmap



7. train_models.py
Trains Random Forest, XGBoost, and CatBoost regressors
Compares models on MSE, RMSE, and R² score
Plots model_comparison_r2.png and feature_importance.png

9. train_rf_cv.py
Runs 5-Fold Cross-Validation on Random Forest
Reports per-fold R² and RMSE, saves to cv_results.txt
Plots fold R² boxplot

11. deploy_model.py
Trains the final Random Forest on the full dataset
Saves models/rf_irrigation_model.pkl and models/encoders.pkl

13. smart_irrigation.py — SmartIrrigationSystem class
predict(input_data) → returns "Low", "Medium", or "High"
optimize(input_data) → returns human-readable irrigation recommendationScript Summaries

1. water_conserve.py
Loads dataset and prints shape, columns, statistics, class distributions
Estimates water savings potential based on Low/Medium/High need split
Generates plots/water_conservation_pie.png

3. eda_script.py
Generates descriptive stats → eda_stats.txt
Plots: histograms, correlation heatmap, pairplot, boxplots by irrigation level

5. correlation_analysis.py
Ordinal-encodes target (Low=0, Medium=1, High=2)
Engineers 3 new features:
Moisture_Temperature_Ratio = Soil_Moisture / Temperature_C
Rain_Humidity_Index = Rainfall_mm × Humidity / 100
Sunlight_Wind_Ratio = Sunlight_Hours / (Wind_Speed_kmh + 1)
Ranks all features by Pearson correlation with target
Generates correlation bar chart + top-10 feature heatmap

7. train_models.py
Trains Random Forest, XGBoost, and CatBoost regressors
Compares models on MSE, RMSE, and R² score
Plots model_comparison_r2.png and feature_importance.png


9. train_rf_cv.py
Runs 5-Fold Cross-Validation on Random Forest
Reports per-fold R² and RMSE, saves to cv_results.txt
Plots fold R² boxplot


11. deploy_model.py
Trains the final Random Forest on the full dataset
Saves models/rf_irrigation_model.pkl and models/encoders.pkl

13. smart_irrigation.py — SmartIrrigationSystem class
predict(input_data) → returns "Low", "Medium", or "High"
optimize(input_data) → returns human-readable irrigation recommendation
