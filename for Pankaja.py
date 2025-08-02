## Important note

## This code will select the only one city (first pickle file) based on which city is inside your directory
### Save this file
### For me, the first city is Paro (model_Paro.pkl)
### Wait for few seconds and the visualization will pop-up

### RUN IT ON VSCODE

## Folder structure to be saved in your local system:

##📂 bhutan_app
##│
##├── for_Pankaja.py               
##│
##├── 📂 data
##│   └── bhutan_runoff_hourly_data.csv
##│
##└── 📂 model
##    └── model_Paro.pkl

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

BASE_DIR = os.getcwd() 
DATA_PATH = os.path.join(BASE_DIR, "data", "bhutan_runoff_hourly_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['Hour'] = df['DateTime'].dt.hour
    return df

df = load_data()

# select your city
available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
cities = sorted([f.replace("model_", "").replace(".pkl", "") for f in available_models])

selected_city = cities[0]  #selecting only one city (Paro)
print(f"Selected city: {selected_city}")

# Loading the model
model_path = os.path.join(MODEL_DIR, f"model_{selected_city}.pkl")
model = joblib.load(model_path)

future_dates = pd.date_range(start="2026-01-01", periods=10, freq="D")
city_df_full = df[df['City'] == selected_city].copy()
last_row = city_df_full.iloc[-1]

forecast_features = pd.DataFrame({
    "Temperature": [last_row['Temperature']] * 10,
    "Precipitation": [last_row['Precipitation']] * 10,
    "Month": future_dates.month,
    "Day": future_dates.day,
    "Hour": [12] * 10
})

forecast_values = model.predict(forecast_features)
forecast_df = pd.DataFrame({"Date": future_dates, "Forecast_Runoff": forecast_values})

### Visualization
plt.figure(figsize=(8, 5))
plt.plot(forecast_df['Date'], forecast_df['Forecast_Runoff'], marker='o', color='blue', label='Forecast')
plt.title(f"10-Day Forecast (2026) - {selected_city}")
plt.xlabel("Date")
plt.ylabel("Runoff")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()