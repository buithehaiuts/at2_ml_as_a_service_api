from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import json
import requests
import pickle
from datetime import datetime
import logging
import os
import uvicorn
import streamlit as st
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoading:
    """Class responsible for loading DataFrames from parts specified in the data.json file."""

    def __init__(self, json_file_path='data.json', cache_dir: str = './cache/'):
        self.json_file_path = json_file_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Load DataFrames here, and they will be cached
        self.train_df, self.test_df = self.load_dataframes()

    def load_dataframes(self) -> (pd.DataFrame, pd.DataFrame):
        """Load training and testing DataFrames from parts specified in the data.json file.
        
        By default, it loads 4 parts of training and 2 parts of testing data.
        
        Returns:
            tuple: A tuple containing the training DataFrame and testing DataFrame.
        """
        try:
            with open(self.json_file_path, 'r') as json_file:
                json_data = json.load(json_file)

            # Default to 4 parts of training and 2 parts of testing data
            train_urls = {k: v for k, v in json_data.get('train', {}).items() if k in ['part1', 'part2', 'part3', 'part4']}
            test_urls = {k: v for k, v in json_data.get('test', {}).items() if k in ['part1', 'part2']}
            
            logger.info(f"Loading 4 parts of training data and 2 parts of testing data by default.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading JSON file: {e}")
            raise

        # Load DataFrame parts
        train_df = self.load_parts_from_github(train_urls)
        test_df = self.load_parts_from_github(test_urls)

        return train_df, test_df

    def load_parts_from_github(self, urls: dict) -> pd.DataFrame:
        """Load parts of DataFrame from GitHub and cache them.
        
        Args:
            urls (dict): A dictionary mapping part names to GitHub URLs.
        
        Returns:
            pd.DataFrame: The concatenated DataFrame loaded from GitHub.
        """
        dfs = []
        for part, url in urls.items():
            df_part = self.load_pickle_with_cache(url, part)
            if df_part is not None:
                dfs.append(df_part)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            logger.warning("No DataFrames were loaded.")
            return pd.DataFrame()  # Return an empty DataFrame if none are loaded

    def load_pickle_with_cache(self, url: str, part: str) -> pd.DataFrame:
        """Load a DataFrame part from a GitHub URL and cache it.
        
        Args:
            url (str): The URL to load the DataFrame from.
            part (str): The name of the part being loaded for caching purposes.
        
        Returns:
            pd.DataFrame: The loaded DataFrame, or None if loading failed.
        """
        cache_file_path = os.path.join(self.cache_dir, f"{part}.pkl")
        # Check if the cached file exists
        if os.path.exists(cache_file_path):
            logger.info(f"Loading {part} from cache.")
            return pd.read_pickle(cache_file_path)

        try:
            logger.info(f"Downloading {part} from {url}.")
            df = pd.read_pickle(url)
            df.to_pickle(cache_file_path)
            return df
        except Exception as e:
            logger.error(f"Error loading {part} from {url}: {e}")
            return None 
    
    def display_dataframe(self, df: pd.DataFrame, num_rows: int = 5, show_columns: bool = True):
        """Display a preview of the DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to display.
            num_rows (int): Number of rows to display. Default is 5.
            show_columns (bool): Whether to display column names. Default is True.
        
        """
        if df.empty:
            logger.warning("The DataFrame is empty.")
            print("The DataFrame is empty.")
            return

        print(f"DataFrame Shape: {df.shape}")
        
        if show_columns:
            print("\nColumns in DataFrame:")
            print(df.columns.tolist())

        print(f"\nDisplaying the first {num_rows} rows of the DataFrame:")
        print(df.head(num_rows))

    
class SalesAPI:
    """Class that defines API endpoints for sales prediction."""

    def __init__(self):
        self.data_loader = DataLoading()
        self.app = FastAPI()
        self.setup_routes()
        # Display the DataFrame immediately after loading
        self.display_initial_data()

    def setup_routes(self):
        self.app.get("/")(self.read_root)
        self.app.get("/health/")(self.health_check)
        self.app.get("/data/display/train/")(self.display_train_data)
        self.app.get("/data/display/test/")(self.display_test_data)

    async def read_root(self) -> Dict[str, Any]:
        return {
            "project": "Sales Prediction API",
            "endpoints": {
                "/": "Project objectives, list of endpoints, etc.",
                "/health/": "Health check endpoint",
                "/data/display/train/": "Display training DataFrame",
                "/data/display/test/": "Display testing DataFrame",
            },
            "github": "https://github.com/buithehaiuts/at2_ml_as_a_service_api"
        }

    async def health_check(self) -> Dict[str, str]:
        return {"message": "Sales Prediction API is running!"}

    async def display_train_data(self) -> Dict[str, Any]:
        """Display the training DataFrame information."""
        return self.data_loader.display_dataframe(self.data_loader.train_df)

    async def display_test_data(self) -> Dict[str, Any]:
        """Display the testing DataFrame information."""
        return self.data_loader.display_dataframe(self.data_loader.test_df)
    
    def display_initial_data(self):
        """Display initial loaded DataFrames to check if load was successful."""
        train_info = self.data_loader.display_dataframe(self.data_loader.train_df)
        test_info = self.data_loader.display_dataframe(self.data_loader.test_df)

        logger.info("Initial Training DataFrame Info: %s", train_info)
        logger.info("Initial Testing DataFrame Info: %s", test_info)       


# class SalesAPI:
#     """Class that defines API endpoints for sales prediction."""

#     def __init__(self):
#         self.data_loader = DataLoader('data.json')
#         self.app = FastAPI()
#         self.setup_routes()

#     def setup_routes(self):
#         self.app.get("/")(self.read_root)
#         self.app.get("/health/")(self.health_check)
#         self.app.get("/sales/national/")(self.forecast_sales)
#         self.app.get("/sales/stores/items/")(self.predict_sales)
#         self.app.get("/data/display/train/")(self.display_train_data)
#         self.app.get("/data/display/test/")(self.display_test_data)
#         self.app.get("/data/ids/")(self.get_ids)

#     async def read_root(self) -> Dict[str, Any]:
#         return {
#             "project": "Sales Prediction API",
#             "endpoints": {
#                 "/": "Project objectives, list of endpoints, etc.",
#                 "/health/": "Health check endpoint",
#                 "/sales/national/": "Forecast sales for the next 7 days",
#                 "/sales/stores/items/": "Query sales for specific store and item",
#                 "/data/display/train/": "Display training DataFrame",
#                 "/data/display/test/": "Display testing DataFrame",
#                 "/data/ids/": "Get available store and item IDs"
#             },
#             "github": "https://github.com/buithehaiuts/at2_ml_as_a_service_api"
#         }

#     async def health_check(self) -> Dict[str, str]:
#         return {"message": "Sales Prediction API is running!"}

#     async def forecast_sales(self, date: str) -> Dict[str, float]:
#         self.validate_date(date)
#         try:
#             forecast_data = self.generate_forecast(date)
#             return forecast_data
#         except Exception as e:
#             logger.error(f"Error generating forecast: {e}")
#             raise HTTPException(status_code=500, detail="Internal server error while forecasting sales")

#     async def predict_sales(self, date: str, store_id: int, item_id: int) -> Dict[str, float]:
#         self.validate_date(date)
#         self.validate_store_item(store_id, item_id)

#         try:
#             # Placeholder prediction logic
#             prediction = {"prediction": 19.72}  # Replace with actual prediction logic
#             return prediction
#         except Exception as e:
#             logger.error(f"Error predicting sales: {e}")
#             raise HTTPException(status_code=500, detail="Internal server error while predicting sales")

#     async def display_train_data(self) -> Dict[str, Any]:
#         """Display the training DataFrame information."""
#         return self.data_loader.display_dataframe(self.data_loader.train_df)

#     async def display_test_data(self) -> Dict[str, Any]:
#         """Display the testing DataFrame information."""
#         return self.data_loader.display_dataframe(self.data_loader.test_df)

#     async def get_ids(self) -> Dict[str, List[int]]:
#         """Return available store and item IDs."""
#         if not self.data_loader.store_ids or not self.data_loader.item_ids:
#             raise HTTPException(status_code=404, detail="Store or item data not available")
#         return {
#             "store_ids": self.data_loader.store_ids,
#             "item_ids": self.data_loader.item_ids
#         }

#     @staticmethod
#     def validate_date(date_str: str):
#         """Validate the date format (YYYY-MM-DD)."""
#         try:
#             datetime.strptime(date_str, '%Y-%m-%d')
#         except ValueError:
#             raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

#     def validate_store_item(self, store_id: int, item_id: int):
#         """Validate store_id and item_id."""
#         if store_id not in self.data_loader.store_ids:
#             raise HTTPException(status_code=404, detail=f"Store ID {store_id} not found.")
#         if item_id not in self.data_loader.item_ids:
#             raise HTTPException(status_code=404, detail=f"Item ID {item_id} not found.")

#     def generate_forecast(self, start_date: str) -> Dict[str, Any]:
#         """Generate a sales forecast."""
#         # Implement your forecasting logic here
#         return {"date": start_date, "forecast": 1000.00}  # Dummy return