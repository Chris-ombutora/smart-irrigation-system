import pandas as pd
import numpy as np
import joblib
import os

class SmartIrrigationSystem:
    def __init__(self, model_path, encoders_path):
        """
        Initialize the Smart Irrigation System.
        
        Args:
            model_path (str): Path to the trained model file (.pkl).
            encoders_path (str): Path to the encoders file (.pkl).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Encoders not found at {encoders_path}")
            
        print(f"Loading Smart Irrigation System from {model_path}...")
        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoders_path)
        print("System Loaded Successfully.")
        
        # Define expected feature columns 
        # (This must match the order during training in deploy_model.py)
        # deploy_model.py used:
        # X = df.drop(target_col, axis=1) where df had:
        # Soil_Type, Soil_pH, Soil_Moisture, Organic_Carbon, Electrical_Conductivity, 
        # Temperature_C, Humidity, Rainfall_mm, Sunlight_Hours, Wind_Speed_kmh, 
        # Crop_Type, Crop_Growth_Stage, Season, Irrigation_Type, Water_Source, 
        # Field_Area_hectare, Mulching_Used, Previous_Irrigation_mm, Region, 
        # Moisture_Temperature_Ratio
        
        self.feature_columns = [
            'Soil_Type', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon',
            'Electrical_Conductivity', 'Temperature_C', 'Humidity', 'Rainfall_mm',
            'Sunlight_Hours', 'Wind_Speed_kmh', 'Crop_Type', 'Crop_Growth_Stage',
            'Season', 'Irrigation_Type', 'Water_Source', 'Field_Area_hectare',
            'Mulching_Used', 'Previous_Irrigation_mm', 'Region',
            'Moisture_Temperature_Ratio'
        ]

    def _engineer_features(self, df):
        if 'Soil_Moisture' in df.columns and 'Temperature_C' in df.columns:
            df['Moisture_Temperature_Ratio'] = df['Soil_Moisture'] / df['Temperature_C']
        return df

    def predict(self, input_data):
        """
        Predict irrigation need (Low, Medium, High).
        input_data: dict of feature values.
        """
        # Convert dict to DataFrame
        df = pd.DataFrame([input_data])
        
        # Feature Engineering
        df = self._engineer_features(df)
        
        # Encode Categoricals
        for col, le in self.encoders.items():
            if col in df.columns:
                # Handle unknown labels? For now, we assume valid input or handle error
                # Ideally we try transform, if fail, map to unknown/mode
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    # simplistic fallback
                    df[col] = 0 
        
        # Ensure correct column order
        # Add missing columns with 0 if necessary? No, strict validation is better for now.
        # But let's check for missing cols
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
             raise ValueError(f"Missing columns in input: {missing}")
             
        X = df[self.feature_columns]
        
        # Predict
        # Model predicts float 0..2
        pred_val = self.model.predict(X)[0]
        # Round to nearest class
        pred_class_idx = int(round(pred_val))
        
        mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        return mapping.get(pred_class_idx, "Unknown")

    def optimize(self, input_data):
        """
        Returns a recommendation string based on prediction.
        """
        level = self.predict(input_data)
        
        if level == "High":
            return "Recommendation: TURN ON IRRIGATION (Full Cycle). Soil moisture is critical."
        elif level == "Medium":
            return "Recommendation: TURN ON IRRIGATION (Half Cycle). Moderate water need."
        elif level == "Low":
            return "Recommendation: DO NOT IRRIGATE. Moisture sufficient."
        else:
            return "Error: Could not determine irrigation need."
