"""
This module provides API endpoints for weather forecasting and crop prediction
based on weather conditions using Flask, MongoDB, and Open-Meteo API.
"""

import os
import pickle
from flask import Flask, jsonify, request
from pymongo import MongoClient
import openmeteo_requests
import numpy as np
import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
CACHE_SESSION = requests_cache.CachedSession('.cache', expire_after=3600)
RETRY_SESSION = retry(CACHE_SESSION, retries=5, backoff_factor=0.2)
OPENMETEO = openmeteo_requests.Client(session=RETRY_SESSION)

URL = "https://api.open-meteo.com/v1/forecast"

app = Flask(__name__)

@app.route('/api/forecast', methods=['GET'])
def forecast():
    """
    Endpoint to get weather forecast for a given latitude and longitude.
    """
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')
    response = get_weather_forecast(latitude, longitude, True)
    return jsonify(response=response, timestamp=pd.Timestamp.now().isoformat(),
                   timeframe="hourly", latitude=latitude, longitude=longitude)

@app.route('/api/predict', methods=['GET'])
def predict():
    """
    Endpoint to predict crops based on weather conditions for a given latitude and longitude.
    """
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')
    conditions = get_weather_forecast(latitude, longitude)
    predicted_crops = predict_crops_for_conditions(conditions)
    return jsonify(conditions=conditions, predicted_crops=predicted_crops,
                   latitude=latitude, longitude=longitude)

def get_weather_forecast(latitude, longitude, raw=False):
    """
    Get weather forecast for a given latitude and longitude using Open-Meteo API.

    Args:
        latitude (str): Latitude of the location.
        longitude (str): Longitude of the location.
        raw (bool): Whether to include raw hourly weather data.

    Returns:
        dict: Weather conditions including temperature, humidity, and rainfall.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain"],
        "forecast_days": 16
    }
    responses = OPENMETEO.weather_api(URL, params=params)
    response = responses[0]

    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    hourly_data = extract_hourly_data(response)
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    print(hourly_dataframe.head())

    averages = calculate_averages(hourly_dataframe)
    print_averages(averages)

    if raw:
        data = averages
        data['weather_data'] = hourly_dataframe.to_dict(orient='records')
        return data

    return averages

def extract_hourly_data(response):
    """
    Extract hourly weather data from the API response.

    Args:
        response: API response object.

    Returns:
        dict: Hourly weather data.
    """
    hourly = response.Hourly()
    return {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy().tolist(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy().tolist(),
        "rain": hourly.Variables(2).ValuesAsNumpy().tolist()
    }

def calculate_averages(hourly_dataframe):
    """
    Calculate average temperature, humidity, and rainfall.

    Args:
        hourly_dataframe (pd.DataFrame): DataFrame containing hourly weather data.

    Returns:
        dict: Average weather conditions.
    """
    return {
        "temperature": hourly_dataframe['temperature_2m'].mean(),
        "humidity": hourly_dataframe['relative_humidity_2m'].mean(),
        "rainfall": hourly_dataframe['rain'].mean()
    }

def print_averages(averages):
    """
    Print the average weather conditions.

    Args:
        averages (dict): Average weather conditions.
    """
    print(f"Average Temperature: {averages['temperature']}")
    print(f"Average Relative Humidity: {averages['humidity']}")
    print(f"Average Rain: {averages['rainfall']}")

def insert_to_mongo(data):
    """
    Insert data into MongoDB collection.
    """
    client = MongoClient(os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017'))
    db = client['harvestpro']
    collection = db['prediction']
    collection.insert_one(data)

def predict_crops_for_conditions(conditions):
    """
    Predict crops based on weather conditions using a pre-trained model.

    Args:
        conditions (dict): Weather conditions including temperature, humidity, and rainfall.

    Returns:
        list: List of top 3 predicted crops with their probabilities.
    """
    with open('crop_model.pkl', 'rb') as file:
        model = pickle.load(file, encoding='latin1')

    conditions_df = pd.DataFrame(conditions, index=[0])
    probs = model.predict_proba(conditions_df)[0]
    top_indices = np.argsort(probs)[::-1]
    return [f"{model.classes_[idx]} ({probs[idx]*100:.2f}%)" for idx in top_indices[:3]]

if __name__ == '__main__':
    app.run(debug=True)
