from flask import Flask, jsonify, request
import openmeteo_requests
import numpy as np
import requests_cache
import pandas as pd
from retry_requests import retry
from flask import jsonify
from pymongo import MongoClient
import pickle

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://api.open-meteo.com/v1/forecast"

app = Flask(__name__)


@app.route('/api/forecast', methods=['POST'])
def forecast():
    # Get the latitude and longitude from the request JSON
    latitude = request.json.get('latitude')
    longitude = request.json.get('longitude')
    # Get the weather forecast for the given latitude and longitude
    response = get_weather_forecast(latitude, longitude, True)
    return jsonify(response=response, timestamp=pd.Timestamp.now().isoformat(), timeframe="hourly")


@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the latitude and longitude from the request JSON
    latitude = request.json.get('latitude')
    longitude = request.json.get('longitude')
    # Get the weather conditions for the given latitude and longitude
    conditions = get_weather_forecast(latitude, longitude)
    # Predict the crops based on the weather conditions
    predicted_crops = predict_crops_for_conditions(conditions)
    return jsonify(conditions=conditions, predicted_crops=predicted_crops)


def get_weather_forecast(latitude, longitude, raw=False):
    # Define the parameters for the weather forecast API request
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain"],
        "forecast_days": 16
    }
    # Get the weather forecast response from the Open-Meteo API
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    hourly_data["temperature_2m"] = hourly_temperature_2m.tolist()
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m.tolist()
    hourly_data["rain"] = hourly_rain.tolist()

    hourly_dataframe = pd.DataFrame(data=hourly_data)

    print(hourly_dataframe.head())
    average_temperature = hourly_dataframe['temperature_2m'].mean()
    average_relative_humidity = hourly_dataframe['relative_humidity_2m'].mean()
    average_rain = hourly_dataframe['rain'].mean()

    print(f"Average Temperature: {average_temperature}")
    print(f"Average Relative Humidity: {average_relative_humidity}")
    print(f"Average Rain: {average_rain}")

    data = {
        "temperature": average_temperature,
        "humidity": average_relative_humidity,
        "rainfall": average_rain
    }
    if raw:
        data['weather_data'] = hourly_dataframe.to_dict(orient='records')
        return data

    return data


def insert_to_mongo(data):
    # Connect to MongoDB and insert the data
    client = MongoClient('mongodb://localhost:27017')
    db = client['harvestpro']
    collection = db['prediction']
    collection.insert_one(data)


def predict_crops_for_conditions(conditions):
    print("Predicting crops for conditions")
    # Load the crop prediction model from a pickle file
    model = None
    with open('crop_model.pkl', 'rb') as file:
        model = pickle.load(file, encoding='latin1')

    conditions_df = pd.DataFrame(conditions, index=[0])
    probs = model.predict_proba(conditions_df)[0]
    top_indices = np.argsort(probs)[::-1]
    predicted_crops = []
    for idx in top_indices:
        if len(predicted_crops) < 3:
            predicted_crops.append(
                f"{model.classes_[idx]} ({probs[idx]*100:.2f}%)")
    return predicted_crops


if __name__ == '__main__':
    app.run(debug=True)
