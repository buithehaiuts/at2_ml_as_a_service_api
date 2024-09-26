from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "project": "Sales Prediction API",
        "endpoints": {
            "/": "Project objectives, list of endpoints, etc.",
            "/health/": "Health check endpoint",
            "/sales/national/": "Forecast sales for the next 7 days",
            "/sales/stores/items/": "Predict sales for specific store and item"
        },
        "github": "https://github.com/your_repo_here"
    }

@app.get("/health/")
def health_check():
    return {"message": "Welcome to the Sales Prediction API!"}

class SalesPredictionRequest(BaseModel):
    date: str
    store_id: int
    item_id: int

@app.get("/sales/national/")
def forecast_sales(date: str):
    # Replace with your forecasting logic
    return {
        "2016-01-01": 10000.01,
        "2016-01-02": 10001.12,
        "2016-01-03": 10002.22,
        "2016-01-04": 10003.30,
        "2016-01-05": 10004.46,
        "2016-01-06": 10005.12,
        "2016-01-07": 10006.55,
    }

@app.get("/sales/stores/items/")
def predict_sales(date: str, store_id: int, item_id: int):
    # Replace with your prediction logic
    return {"prediction": 19.72}
