# backend/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import uvicorn

# Create FastAPI instance
api_app = FastAPI()

# Sample data for demonstration
sales_data = {
    "2024-01-01": 10000.01,
    "2024-01-02": 10001.12,
    "2024-01-03": 10002.22,
    "2024-01-04": 10003.30,
    "2024-01-05": 10004.46,
    "2024-01-06": 10005.12,
    "2024-01-07": 10006.55,
}

class SalesPredictionRequest(BaseModel):
    date: str
    store_id: int
    item_id: int

@api_app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Prediction API"}

@api_app.get("/health/")
def health_check():
    return {"status": "healthy"}

@api_app.get("/sales/national/")
def get_national_sales_forecast(date: str):
    # Validate date format and return sales forecasts for the next 7 days
    try:
        input_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    # Generate forecast for the next 7 days
    forecast = {}
    for i in range(1, 8):
        forecast_date = (input_date + timedelta(days=i)).strftime("%Y-%m-%d")
        forecast[forecast_date] = sales_data.get(forecast_date, 0.0)

    return forecast

@api_app.post("/sales/stores/items/")
def get_store_item_sales_prediction(request: SalesPredictionRequest):
    # Logic to predict sales for a specific store and item
    # Replace with your actual prediction logic
    return {"prediction": 19.72}

if __name__ == "__main__":  
    # This block will be ignored when running inside Docker
    uvicorn.run(api_app, host="0.0.0.0", port=8000, log_level="info")
