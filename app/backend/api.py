from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import json
import requests
import pickle
from datetime import datetime, timedelta

class DataLoader:
    """Class responsible for loading DataFrames from JSON and Dropbox."""

    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.train_df, self.test_df = self.load_dataframes()

    def load_dataframes(self):
        """Load training and testing DataFrames from JSON and Dropbox links."""
        try:
            with open(self.json_file_path, 'r') as json_file:
                json_data = json.load(json_file)

            # Extract train and test URLs into DataFrames
            train_urls = json_data['train']
            test_urls = json_data['test']

            train_url_df = pd.DataFrame(train_urls.items(), columns=['part', 'url'])
            test_url_df = pd.DataFrame(test_urls.items(), columns=['part', 'url'])

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
            for _, row in train_url_df.iterrows():
                part = row['part']
                url = row['url']
                train_data[part] = load_pickle_from_dropbox(url)

            train_df = pd.concat([df for df in train_data.values() if df is not None], ignore_index=True)

            # Load testing data
            test_data = {}
            for _, row in test_url_df.iterrows():
                part = row['part']
                url = row['url']
                test_data[part] = load_pickle_from_dropbox(url)

            test_df = pd.concat([df for df in test_data.values() if df is not None], ignore_index=True)

            return train_df, test_df

        except Exception as e:
            print(f"An error occurred while loading DataFrames: {e}")
            return None, None

    def display_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Display basic information about the DataFrame."""
        return {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "head": df.head().to_dict(orient='records'),
            "tail": df.tail().to_dict(orient='records')
        }


class SalesAPI:
    """Class that defines API endpoints for sales prediction."""

    def __init__(self):
        self.data_loader = DataLoader('data.json')
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/")
        def read_root() -> Dict[str, Dict[str, str]]:
            return {
                "project": "Sales Prediction API",
                "endpoints": {
                    "/": "Project objectives, list of endpoints, etc.",
                    "/health/": "Health check endpoint",
                    "/sales/national/": "Forecast sales for the next 7 days",
                    "/sales/stores/items/": "Query sales for specific store and item",
                    "/data/display/train/": "Display training DataFrame",
                    "/data/display/test/": "Display testing DataFrame"
                },
                "github": "https://github.com/buithehaiuts/at2_ml_as_a_service_api"
            }

        @self.app.get("/health/")
        def health_check() -> Dict[str, str]:
            return {"message": "Welcome to the Sales Prediction API!"}

        @self.app.get("/sales/national/")
        def forecast_sales(date: str) -> Dict[str, float]:
            self.validate_date(date)
            forecast_data = self.generate_forecast(date)
            return forecast_data

        @self.app.get("/sales/stores/items/")
        def predict_sales(date: str, store_id: int, item_id: int) -> Dict[str, float]:
            self.validate_date(date)
            self.validate_store_item(store_id, item_id)

            # Placeholder prediction logic (replace with actual model logic)
            prediction = {"prediction": 19.72}  # Replace with actual prediction logic
            return prediction

        @self.app.get("/data/display/train/")
        def display_train_data() -> Dict[str, Any]:
            """Display the training DataFrame information."""
            return self.data_loader.display_dataframe(self.data_loader.train_df)

        @self.app.get("/data/display/test/")
        def display_test_data() -> Dict[str, Any]:
            """Display the testing DataFrame information."""
            return self.data_loader.display_dataframe(self.data_loader.test_df)

    @staticmethod
    def validate_date(date: str):
        """Validate the date format."""
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    @staticmethod
    def validate_store_item(store_id: int, item_id: int):
        """Validate store and item IDs."""
        if store_id <= 0 or item_id <= 0:
            raise HTTPException(status_code=400, detail="store_id and item_id must be positive integers")

    @staticmethod
    def generate_forecast(date: str) -> Dict[str, float]:
        """Generate forecasted sales for the next 7 days."""
        forecast_data = {}
        start_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
        for i in range(7):
            forecast_date = start_date + timedelta(days=i)
            forecast_data[forecast_date.strftime('%Y-%m-%d')] = 10000 + i * 1.12  # Example increment logic
        return forecast_data
