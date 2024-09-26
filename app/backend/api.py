# backend/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import requests
import pickle
import os

app = FastAPI()

# Function to download a file from GitHub and load it
def load_data_from_github(file_url):
    response = requests.get(file_url)
    if response.status_code == 200:
        return pickle.loads(response.content)
    else:
        raise Exception("Failed to download file")

# Function to load and merge multiple pickle files
def load_and_merge_data(urls):
    dataframes = []
    for url in urls:
        df = load_data_from_github(url)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# URLs of the pickle files in your GitHub repository
train_urls = [f"https://raw.githubusercontent.com/buithehaiuts/at2_ml_as_a_service_experiments/main/data/processed/train_final_merged_part{i}.pkl" for i in range(1, 21)]
test_urls = [f"https://raw.githubusercontent.com/buithehaiuts/at2_ml_as_a_service_experiments/main/data/processed/test_final_merged_part{i}.pkl" for i in range(1, 11)]

# Load and merge the train and test datasets when the API starts
train_data = load_and_merge_data(train_urls)
test_data = load_and_merge_data(test_urls)

@app.get("/train")
async def get_train_data():
    return train_data.to_dict(orient="records")  # Return as a list of dictionaries

@app.get("/test")
async def get_test_data():
    return test_data.to_dict(orient="records")  # Return as a list of dictionaries
