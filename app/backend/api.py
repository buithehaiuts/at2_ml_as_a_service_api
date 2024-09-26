from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import requests
import pickle
from typing import Optional, List

app = FastAPI()

# Function to download a file from GitHub and load it
def load_data_from_github(file_url: str) -> pd.DataFrame:
    response = requests.get(file_url)
    if response.status_code == 200:
        return pickle.loads(response.content)
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to download file")

# Function to load and merge multiple pickle files
def load_and_merge_data(urls: List[str]) -> pd.DataFrame:
    dataframes = []
    for url in urls:
        try:
            df = load_data_from_github(url)
            dataframes.append(df)
        except HTTPException as e:
            print(f"Error loading data from {url}: {e.detail}")
    if not dataframes:
        raise HTTPException(status_code=500, detail="No data loaded from any URL")
    return pd.concat(dataframes, ignore_index=True)

# URLs of the pickle files in your GitHub repository
def get_train_urls() -> List[str]:
    return [
        f"https://raw.githubusercontent.com/buithehaiuts/at2_ml_as_a_service_experiments/main/data/processed/train_final_merged_part{i}.pkl" 
        for i in range(1, 21)
    ]

def get_test_urls() -> List[str]:
    return [
        f"https://raw.githubusercontent.com/buithehaiuts/at2_ml_as_a_service_experiments/main/data/processed/test_final_merged_part{i}.pkl" 
        for i in range(1, 11)
    ]

# Data loading on demand (lazy loading)
train_data: Optional[pd.DataFrame] = None
test_data: Optional[pd.DataFrame] = None

@app.get("/")
async def root():
    return {"message": "Welcome to the Sales Revenue Forecasting and Prediction API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/train")
async def get_train_data():
    global train_data
    if train_data is None:
        try:
            train_data = load_and_merge_data(get_train_urls())
        except HTTPException as e:
            raise HTTPException(status_code=500, detail="Failed to load train data: " + str(e.detail))
    return train_data.to_dict(orient="records")

@app.get("/test")
async def get_test_data():
    global test_data
    if test_data is None:
        try:
            test_data = load_and_merge_data(get_test_urls())
        except HTTPException as e:
            raise HTTPException(status_code=500, detail="Failed to load test data: " + str(e.detail))
    return test_data.to_dict(orient="records")

# Endpoint for sales prediction
class SalesPredictionRequest(BaseModel):
    date: str
    store_id: int
    item_id: int

@app.post("/sales/stores/items/")
async def predict_sales(request: SalesPredictionRequest):
    # Placeholder for model prediction logic
    prediction = 19.72  # Dummy prediction
    # Here you would call your trained model to get the actual prediction based on request data
    return {"prediction": prediction}
