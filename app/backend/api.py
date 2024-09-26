from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import requests
import logging
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production use
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample data for demonstration
dummy_sales_forecast = {
    "2016-01-01": 10000.01,
    "2016-01-02": 10001.12,
    "2016-01-03": 10002.22,
    "2016-01-04": 10003.30,
    "2016-01-05": 10004.46,
    "2016-01-06": 10005.12,
    "2016-01-07": 10006.55,
}

# Brief project description
@app.get("/", status_code=200)
async def root():
    return {
        "description": "This API predicts sales revenue based on historical data.",
        "endpoints": {
            "/": "Brief description of the project.",
            "/health/": "Check the health of the API.",
            "/sales/national/": "Get forecasted sales for the next 7 days.",
            "/sales/stores/items/": "Get predicted sales for a specific store and item."
        },
        "expected_input": {
            "sales/national/": {"date": "YYYY-MM-DD"},
            "sales/stores/items/": {"date": "YYYY-MM-DD", "store_id": "int", "item_id": "int"}
        },
        "expected_output": {
            "sales/national/": {
                "2016-01-01": 10000.01,
                "2016-01-02": 10001.12,
                "2016-01-03": 10002.22,
                "2016-01-04": 10003.30,
                "2016-01-05": 10004.46,
                "2016-01-06": 10005.12,
                "2016-01-07": 10006.55,
            },
            "sales/stores/items/": {"prediction": 19.72}
        },
        "github_repo": "https://github.com/your_github_repo_here"
    }

@app.get("/health/", status_code=200)
async def health_check():
    return {"status": "healthy"}

@app.get("/sales/national/", status_code=200)
async def forecast_sales(date: str):
    # Validate the date format
    try:
        start_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Generate next 7 days of forecasted sales
    forecast = {}
    for i in range(1, 8):
        forecast[(start_date + timedelta(days=i)).strftime("%Y-%m-%d")] = dummy_sales_forecast[
            "2016-01-01"] + (i * 1.11)  # Example logic for demonstration

    return forecast

class SalesPredictionRequest(BaseModel):
    date: str  # Expected date format YYYY-MM-DD
    store_id: int
    item_id: int

@app.get("/sales/stores/items/")
async def predict_sales(request: SalesPredictionRequest):
    # Example prediction logic (dummy prediction)
    # You can replace this with actual model prediction logic later
    return {"prediction": 19.72}  # Placeholder value for demonstration

