# HarvestPro

HarvestPro is a project that helps farmers select the crops based on the current weather conditions. The project uses a Flask application to serve the predictions.

## Project Description

The main application logic is contained in the `app.py` file. This file contains the Flask application and routes for the API endpoints.

## Setup

To set up the Flask application, follow these steps:

1. Clone the repository.
2. Navigate to the project directory.
3. Create a virtual environment using venv:
   ```bash
   python3 -m venv env
   ```
4. Activate the virtual environment:
   ```
     env\bin\activate # On MacOS use `source env/bin/activate`
   ```
5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run application

To run the application, use the following command:

```

    python3 app.py

```

## API Endpoint

### /api/predict
This endpoint accepts POST requests with the following payload:

        {
            "latitude": 19.0728,
            "longitude": 72.8826
        }

The response will be:

        {
            "conditions": {
                "humidity": 63.809895833333336,
                "rainfall": 0.0,
                "temperature": 28.35866275926431
            },
            "predicted_crops": [
                "muskmelon (53.00%)",
                "mothbeans (24.00%)",
                "lentil (21.00%)"
            ]
        }

### /api/forecast
This endpoint accepts POST requests with the following payload:

        {
            "latitude": 19.0728,
            "longitude": 72.8826
        }

The response will be:

    {
    "response": {
        "humidity": 63.809895833333336,
        "rainfall": 0.0,
        "temperature": 28.35866275926431,
        "weather_data": []
        }
    }
