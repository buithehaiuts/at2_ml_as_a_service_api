# backend/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import requests
import pickle

app = FastAPI()

# Function to download a file from GitHub and load it
def load_data_from_github(file_url: str) -> pd.DataFrame:
    response = requests.get(file_url)
    if response.status_code == 200:
        return pickle.loads(response.content)
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to download file")

# Function to load and merge multiple pickle files
def load_and_merge_data(urls: list) -> pd.DataFrame:
    dataframes = []
    for url in urls:
        try:
            df = load_data_from_github(url)
            dataframes.append(df)
        except HTTPException as e:
            # Handle specific URL load failure
            print(f"Error loading data from {url}: {e.detail}")
    return pd.concat(dataframes, ignore_index=True)

# URLs of the pickle files in your GitHub repository
train_urls = [
    f"https://raw.githubusercontent.com/buithehaiuts/at2_ml_as_a_service_experiments/main/data/processed/train_final_merged_part{i}.pkl" 
    for i in range(1, 21)
]
test_urls = [
    f"https://raw.githubusercontent.com/buithehaiuts/at2_ml_as_a_service_experiments/main/data/processed/test_final_merged_part{i}.pkl" 
    for i in range(1, 11)
]

# Load and merge the train and test datasets when the API starts
try:
    train_data = load_and_merge_data(train_urls)
    test_data = load_and_merge_data(test_urls)
except Exception as e:
    print(f"Error loading datasets: {e}")
    train_data = pd.DataFrame()  # Set to empty DataFrame if loading fails
    test_data = pd.DataFrame()    # Set to empty DataFrame if loading fails

@app.get("/train")
async def get_train_data():
    if train_data.empty:
        raise HTTPException(status_code=500, detail="Train data is not available.")
    return train_data.to_dict(orient="records")  # Return as a list of dictionaries

@app.get("/test")
async def get_test_data():
    if test_data.empty:
        raise HTTPException(status_code=500, detail="Test data is not available.")
    return test_data.to_dict(orient="records")  # Return as a list of dictionaries
