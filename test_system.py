from smart_irrigation import SmartIrrigationSystem
import os

# Paths
base_dir = r"C:\Users\Chris\Desktop\irrigation system"
model_path    = os.path.join(base_dir, "models", "rf_irrigation_model.pkl")
encoders_path = os.path.join(base_dir, "models", "encoders.pkl")

# Initialize System
print("Initializing Smart Irrigation System...")
try:
    system = SmartIrrigationSystem(model_path, encoders_path)
except FileNotFoundError as e:
    print(e)
    exit(1)

# Define Scenarios
scenarios = [
    {
        "name": "Scenario 1: Hot & Dry (Critical)",
        "data": {
            'Soil_Type': 'Sandy',
            'Soil_pH': 6.5,
            'Soil_Moisture': 10.0, # Very Low
            'Organic_Carbon': 0.5,
            'Electrical_Conductivity': 1.5,
            'Temperature_C': 35.0, # High
            'Humidity': 30.0,
            'Rainfall_mm': 0.0,
            'Sunlight_Hours': 10.0,
            'Wind_Speed_kmh': 15.0,
            'Crop_Type': 'Wheat',
            'Crop_Growth_Stage': 'Flowering',
            'Season': 'Summer',
            'Irrigation_Type': 'Drip',
            'Water_Source': 'Ground Water',
            'Field_Area_hectare': 5.0,
            'Mulching_Used': 'No',
            'Previous_Irrigation_mm': 0.0,
            'Region': 'Desert' # Hypothetical region
            # 'Moisture_Temperature_Ratio' will be calculated by system: 10/35 = 0.28 (Low)
        }
    },
    {
        "name": "Scenario 2: Cool & Wet (No Need)",
        "data": {
            'Soil_Type': 'Loam',
            'Soil_pH': 7.0,
            'Soil_Moisture': 60.0, # High
            'Organic_Carbon': 1.2,
            'Electrical_Conductivity': 2.0,
            'Temperature_C': 20.0, # Low-ish
            'Humidity': 80.0,
            'Rainfall_mm': 500.0,
            'Sunlight_Hours': 5.0,
            'Wind_Speed_kmh': 5.0,
            'Crop_Type': 'Rice',
            'Crop_Growth_Stage': 'Vegetative',
            'Season': 'Monsoon',
            'Irrigation_Type': 'Canal',
            'Water_Source': 'River',
            'Field_Area_hectare': 2.0,
            'Mulching_Used': 'Yes',
            'Previous_Irrigation_mm': 50.0,
            'Region': 'Humid'
            # Ratio: 60/20 = 3.0 (High)
        }
    },
    {
        "name": "Scenario 3: Moderate Conditions (Medium)",
        "data": {
            'Soil_Type': 'Clay',
            'Soil_pH': 6.0,
            'Soil_Moisture': 30.0, # Medium
            'Organic_Carbon': 0.8,
            'Electrical_Conductivity': 1.8,
            'Temperature_C': 28.0,
            'Humidity': 50.0,
            'Rainfall_mm': 50.0,
            'Sunlight_Hours': 8.0,
            'Wind_Speed_kmh': 10.0,
            'Crop_Type': 'Maize',
            'Crop_Growth_Stage': 'Fruiting',
            'Season': 'Spring',
            'Irrigation_Type': 'Sprinkler',
            'Water_Source': 'Well',
            'Field_Area_hectare': 4.0,
            'Mulching_Used': 'No',
            'Previous_Irrigation_mm': 20.0,
            'Region': 'Semi-Arid'
            # Ratio: 30/28 = 1.07
        }
    }
]

# Run Simulations
print("\nRunning Simulations...\n")
for scenario in scenarios:
    print(f"--- {scenario['name']} ---")
    data = scenario['data']
    
    # Predict
    try:
        recommendation = system.optimize(data)
        print(f"Features: Moisture={data['Soil_Moisture']}, Temp={data['Temperature_C']}")
        print(f"System Output: {recommendation}")
    except Exception as e:
        print(f"Error: {e}")
    print()

print("Simulation Complete.")
