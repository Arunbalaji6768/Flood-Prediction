import requests
import joblib
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

API_KEY = "S6CVFET6UM8AY64E9UHX7R8Q2"
MODEL_PATH = "C:/Users/manoj/Documents/Rescue/model/flood_model.pkl"  # Relative path

# Load trained models
try:
    model_package = joblib.load(MODEL_PATH)
    xgb_model = model_package['xgb_model']
    rf_model = model_package['rf_model']
    logistic_poly = model_package['logistic_poly']
    scaler = model_package['scaler']
    print("✅ Ensemble models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # Fallback to single model if ensemble fails
    model_package = None

def get_city_weather(city):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}?unitGroup=metric&key={API_KEY}&include=current"
    response = requests.get(url)
    data = response.json()

    return {
        "city": city.title(),
        "temp": data['currentConditions']['temp'],
        "cloud": data['currentConditions']['cloudcover'],
        "max_temp": data['days'][0]['tempmax'],
        "precip": data['days'][0]['precip'],
        "wind": data['currentConditions']['windspeed'],
        "humidity": data['currentConditions']['humidity']
    }

def predict_flood(weather_data):
    # Prepare input features in the correct order
    features = np.array([
        weather_data['temp'],
        weather_data['max_temp'],
        weather_data['wind'],
        weather_data['cloud'],
        weather_data['precip'],
        weather_data['humidity']
    ]).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Get predictions from all three models
    xgb_prob = xgb_model.predict_proba(features_scaled)[0][1]
    rf_prob = rf_model.predict_proba(features_scaled)[0][1]
    logistic_prob = logistic_poly.predict_proba(features_scaled)[0][1]
    
    # Ensemble prediction (weighted average)
    ensemble_prob = (xgb_prob * 0.4 + rf_prob * 0.4 + logistic_prob * 0.2)
    
    # Convert to final prediction
    prediction_label = 1 if ensemble_prob > 0.5 else 0
    confidence = round(max(ensemble_prob, 1 - ensemble_prob) * 100, 2)
    
    # Return label and confidence
    label = "Unsafe" if prediction_label == 1 else "Safe"
    
    # Also return individual model probabilities for display
    individual_probs = {
        'XGBoost': round(xgb_prob * 100, 1),
        'Random Forest': round(rf_prob * 100, 1),
        'Logistic Regression': round(logistic_prob * 100, 1)
    }
    
    return label, confidence, individual_probs