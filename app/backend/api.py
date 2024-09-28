from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import json
import requests
import pickle
from datetime import datetime, timedelta

app = FastAPI()

class SalesPredictionRequest(BaseModel):
    date: str
    store_id: int
    item_id: int

# Load DataFrames from JSON
def load_dataframes(json_file_path: str):
    try:
        # Load the JSON data from the file
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)

        # Extract train and test URLs into DataFrames
        train_urls = json_data['train']
        test_urls = json_data['test']

        # Create DataFrames
        train_url_df = pd.DataFrame(train_urls.items(), columns=['part', 'url'])
        test_url_df = pd.DataFrame(test_urls.items(), columns=['part', 'url'])

        # Update URLs to download files
        train_url_df['url'] = train_url_df['url'].str.replace('dl=0', 'dl=1')
        test_url_df['url'] = test_url_df['url'].str.replace('dl=0', 'dl=1')

        # Load pickle files into DataFrames
        def load_pickle_from_dropbox(dropbox_link):
            try:
                response = requests.get(dropbox_link, stream=True)
                response.raise_for_status()
                data = pickle.load(response.raw)
                return data
            except Exception as e:
                print(f"Error loading pickle: {e}")
                return None

        # Load training data
        train_data = {}
        for index, row in train_url_df.iterrows():
            part = row['part']
            url = row['url']
            train_data[part] = load_pickle_from_dropbox(url)

        # Concatenate all training DataFrames into a single DataFrame
        train_df = pd.concat([df for df in train_data.values() if df is not None], ignore_index=True)

        # Load testing data
        test_data = {}
        for index, row in test_url_df.iterrows():
            part = row['part']
            url = row['url']
            test_data[part] = load_pickle_from_dropbox(url)

        # Concatenate all testing DataFrames into a single DataFrame
        test_df = pd.concat([df for df in test_data.values() if df is not None], ignore_index=True)

        return train_df, test_df

    except Exception as e:
        print(f"An error occurred while loading DataFrames: {e}")
        return None, None

# Load DataFrames on startup 
train_df, test_df = load_dataframes('data.json')

@app.get("/")
def read_root() -> Dict[str, Dict[str, str]]:
    return {
        "project": "Sales Prediction API",
        "endpoints": {
            "/": "Project objectives, list of endpoints, etc.",
            "/health/": "Health check endpoint",
            "/sales/national/": "Forecast sales for the next 7 days",
            "/sales/stores/items/": "Predict sales for specific store and item"
        },
        "github": "https://github.com/buithehaiuts/at2_ml_as_a_service_api"
    }

@app.get("/health/")
def health_check() -> Dict[str, str]:
    return {"message": "Welcome to the Sales Prediction API!"}

@app.get("/sales/national/")
def forecast_sales(date: str) -> Dict[str, float]:
    try:
        # Validate the date format
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    # Generate forecasted sales for the next 7 days
    forecast_data = {}
    start_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)  # Exclude input date
    for i in range(7):
        forecast_date = start_date + timedelta(days=i)
        forecast_data[forecast_date.strftime('%Y-%m-%d')] = 10000 + i * 1.12  # Example increment logic

    return forecast_data

@app.get("/sales/stores/items/")
def predict_sales(date: str, store_id: int, item_id: int) -> Dict[str, float]:
    # Validate input parameters
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    if store_id <= 0 or item_id <= 0:
        raise HTTPException(status_code=400, detail="store_id and item_id must be positive integers")
    
    # Placeholder prediction logic
    prediction = {"prediction": 19.72}  # Replace with actual prediction logic
    return prediction
