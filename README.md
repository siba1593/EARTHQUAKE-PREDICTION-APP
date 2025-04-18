# Earthquake Prediction App

A Streamlit application for predicting earthquake magnitudes using machine learning.

## Features

- Predict earthquake magnitude based on location and historical data
- Visualize prediction results with a gauge chart
- Explore feature importance and model performance metrics
- Upload custom datasets for prediction
- Select location from a map, address search, or direct coordinates input

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd earthquake_prediction_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at http://localhost:8501.

## Application Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies

## How It Works

1. The application uses a Random Forest regression model trained on synthetic earthquake data.
2. Users can select a location using a map, address search, or by inputting coordinates directly.
3. Additional parameters like depth can be adjusted.
4. Upon clicking "Predict", the model estimates the earthquake magnitude for the given location and parameters.
5. Results are visualized using a gauge chart, and an interpretation of the magnitude is provided.

## Model Details

The model uses the following features:
- Latitude and longitude coordinates
- Depth (km)
- Distance from equator
- Month and day of year
- Pacific Ring of Fire indicator

## Custom Data

Users can upload their own earthquake datasets in CSV format to use for analysis.
The required columns are:
- latitude
- longitude
- depth_km
- magnitude

## Note

This application is for educational purposes only and should not be used for actual earthquake prediction or emergency response planning. 