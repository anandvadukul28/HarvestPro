from flask import Flask, jsonify, request
from pymongo import MongoClient
import openmeteo_requests
import numpy as np
import requests_cache
import pandas as pd
from retry_requests import retry
import pickle
import os

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://api.open-meteo.com/v1/forecast"

app = Flask(__name__)


@app.route('/api/forecast', methods=['GET'])
def forecast():
    # Get latitude and longitude from the request parameters
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')
    # Get weather forecast for the given coordinates
    response = get_weather_forecast(latitude, longitude, True)
    return jsonify(response=response, timestamp=pd.Timestamp.now().isoformat(), timeframe="hourly", latitude=latitude, longitude=longitude)


@app.route('/api/predict', methods=['GET'])
def predict():
    # Get latitude and longitude from the request parameters
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')
    # Get weather conditions for the given coordinates
    conditions = get_weather_forecast(latitude, longitude)
    # Predict crops based on the weather conditions
    predicted_crops = predict_crops_for_conditions(conditions)
    return jsonify(conditions=conditions, predicted_crops=predicted_crops, latitude=latitude, longitude=longitude)


def get_weather_forecast(latitude, longitude, raw=False):
    # Set parameters for the weather forecast API request
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain"],
        "forecast_days": 16
    }
    # Get weather forecast response from the Open-Meteo API
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    # Print some information about the weather forecast response
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Extract hourly weather data from the response
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()

    # Prepare the hourly weather data as a DataFrame
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

    # Print the first few rows of the hourly weather data
    print(hourly_dataframe.head())

    # Calculate average temperature, relative humidity, and rain
    average_temperature = hourly_dataframe['temperature_2m'].mean()
    average_relative_humidity = hourly_dataframe['relative_humidity_2m'].mean()
    average_rain = hourly_dataframe['rain'].mean()

    # Print the average weather conditions
    print(f"Average Temperature: {average_temperature}")
    print(f"Average Relative Humidity: {average_relative_humidity}")
    print(f"Average Rain: {average_rain}")

    # Prepare the weather conditions as a dictionary
    data = {
        "temperature": average_temperature,
        "humidity": average_relative_humidity,
        "rainfall": average_rain
    }
    if raw:
        # Include the raw hourly weather data if requested
        data['weather_data'] = hourly_dataframe.to_dict(orient='records')
        return data

    return data


def insert_to_mongo(data):
    # Insert data into MongoDB collection
    client = MongoClient(
        os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017'))
    db = client['harvestpro']
    collection = db['prediction']
    collection.insert_one(data)


def predict_crops_for_conditions(conditions):
    # Load the crop prediction model
    model = None
    with open('crop_model.pkl', 'rb') as file:
        model = pickle.load(file, encoding='latin1')

    # Prepare the weather conditions as a DataFrame
    conditions_df = pd.DataFrame(conditions, index=[0])
    # Predict crop probabilities based on the weather conditions
    probs = model.predict_proba(conditions_df)[0]
    top_indices = np.argsort(probs)[::-1]
    predicted_crops = []
    for idx in top_indices:
        if len(predicted_crops) < 3:
            # Include the top predicted crops with their probabilities
            predicted_crops.append(
                f"{model.classes_[idx]} ({probs[idx]*100:.2f}%)")
    return predicted_crops


if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
