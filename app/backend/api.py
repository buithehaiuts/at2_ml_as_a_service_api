from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

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
        "github": "https://github.com/your_repo_here"
    }

@app.get("/health/")
def health_check() -> Dict[str, str]:
    return {"message": "Welcome to the Sales Prediction API!"}

class SalesPredictionRequest(BaseModel):
    date: str
    store_id: int
    item_id: int

@app.get("/sales/national/")
def forecast_sales(date: str) -> Dict[str, float]:
    # Replace with your forecasting logic
    # Example placeholder data; in a real application, this would call a model or service
    forecast_data = {
        "2016-01-01": 10000.01,
        "2016-01-02": 10001.12,
        "2016-01-03": 10002.22,
        "2016-01-04": 10003.30,
        "2016-01-05": 10004.46,
        "2016-01-06": 10005.12,
        "2016-01-07": 10006.55,
    }
    return forecast_data

@app.get("/sales/stores/items/")
def predict_sales(date: str, store_id: int, item_id: int) -> Dict[str, float]:
    # Replace with your prediction logic
    # Example placeholder prediction; in a real application, this would call a model or service
    if store_id <= 0 or item_id <= 0:
        raise HTTPException(status_code=400, detail="store_id and item_id must be positive integers")
    
    prediction = {"prediction": 19.72}  # Placeholder prediction
    return prediction
