from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel
from typing import List, Dict, Any
import pickle  
import pandas as pd  
import os  
from pathlib import Path
import uvicorn
import logging
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

# Get the current working directory
current_directory = Path(os.getcwd())

# Define the models
class HealthCheck(BaseModel):
    """Response model for health check."""
    status: str

class Sale(BaseModel):
    """Model representing a sale."""
    id: int
    amount: float
    date: str

class SalesResponse(BaseModel):
    """Response model containing a list of sales."""
    sales: List[Sale]

# Create FastAPI instance with versioning
app = FastAPI(
    title="Sales Forecast and Prediction API",
    description="API for forecasting and predicting sales revenue using machine learning and time-series models.",
    version="1.0.0"
)

# Load models into a centralized dictionary
app.state.models = {}

def load_model(model_path: str):
    """Load a prediction model from a file."""
    model_path = Path(model_path).resolve()
    try:
        with open(model_path, 'rb') as f:  # Open the file in binary read mode
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def validate_date(date_str: str) -> bool:
    """Validate the date format to ensure it follows YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

# On startup, load all models
@app.on_event("startup")
async def startup_event():
    # Define model file paths using Path
    model_files = {
        'prophet': 'models/prophet.pkl', # Forecasting model
        'prophet_event': 'models/prophet_event.pkl', # Forecasting model
        'prophet_holiday': 'models/prophet_holiday.pkl',# Forecasting model
        'prophet_month': 'models/prophet_month.pkl', # Forecasting model
        'prophet_predictive_model': 'models/prophet_predictive_model.pkl'  # Used for predicting specific sales
    }
    for model_name, model_path in model_files.items():
        logger.info(f"Attempting to load model from: {model_path}")  # Log the model loading path
        
        model_path = Path(model_path).resolve()
        
        if model_path.exists():  # Check if the model file exists
            try:
                app.state.models[model_name] = load_model(str(model_path))  # Ensure the path is a string
                logger.info(f"{model_name} model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading {model_name} model from {model_path}: {str(e)}")
        else:
            logger.warning(f"Model file does not exist: {model_path}")

@app.get("/")
async def read_root():
    """
    Return project objectives, list of endpoints, input/output format, and GitHub repo link.
    """
    project_info = {
        "project_objectives": "This API provides sales forecasting and store-item sales prediction. "
                              "It allows users to forecast national sales for the next 7 days and predict sales "
                              "for specific store items on a given date.",
        "endpoints": {
            "/": "Displays project objectives, API details, and GitHub repo link. Nelow is the list of expected API endpoints",
            "/health/": {
                "description": "Checks the API health.",
                "method": "GET",
                "response": {"status": "API is healthy and running!"}
            },
            "/sales/national/": {
                "description": "Forecasts next 7 days of national sales starting from the input date.",
                "method": "GET",
                "input_parameters": {"date": "YYYY-MM-DD"},
                "output_format": {
                    "2016-01-01": 10000.01,
                    "2016-01-02": 10001.12,
                    "2016-01-03": 10002.22,
                }
            },
            "/sales/stores/items/": {
                "description": "Predicts sales for a specific store and item on a given date.",
                "method": "GET",
                "input_parameters": {
                    "date": "YYYY-MM-DD",
                    "store_id": "Store ID",
                    "item_id": "Item ID"
                },
                "output_format": {"prediction": 19.72}
            }
        },
        "github_repo": "https://github.com/buithehaiuts/at2_ml_as_a_service_api"
    }
    return project_info
# Health check endpoint
@app.get(
    "/health/",
    tags=["Healthcheck"],
    response_model=HealthCheck,
    summary="Health Check",
    response_description="Returns 200 if healthy",
    status_code=status.HTTP_200_OK,
)
async def health_check():
    """Health Check endpoint to ensure the API is up."""
    return HealthCheck(status="healthy")

# Prediction function for sales using the predictive model
def predict_sales(model, input_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Make a prediction using the Prophet predictive model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Perform prediction using the input data
        predictions = model.predict(input_data)
        
        # Extract relevant columns for output (e.g., ds = date, yhat = forecasted value)
        output = predictions[['ds', 'yhat']].to_dict(orient='records')
        
        return output
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Forecast function for total sales across all stores and items
def forecast_sales(model, start_date: str, period: int = 7) -> List[Dict[str, Any]]:
    """Make a sales forecast using the selected time-series model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert the input start date to a datetime object
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")

        # Forecast starts from the next day
        forecast_start_date = start_date_dt + timedelta(days=1)

        # Create a dataframe for future dates starting from the specified start date
        future_dates = model.make_future_dataframe(periods=period)  # This creates dates for forecasting

        # Replace the last 'period' entries in the 'ds' column with the new future dates
        future_dates['ds'][-period:] = [forecast_start_date + timedelta(days=i) for i in range(period)]

        # Forecast the total revenue for future dates
        train_forecast = model.predict(future_dates)

        # Filter the forecast to only include the new future dates
        output = train_forecast[train_forecast['ds'] >= forecast_start_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')

        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {str(e)}")
        
# Endpoint for predicting sales for a specific item in a store (uses get request)
@app.get("/sales/stores/items/")
async def predict_item_sales(
    ds: str = Query(..., description="Date for prediction in YYYY-MM-DD format"),
    item_id: str = Query(..., description="Item ID for the product"),
    store_id: str = Query(..., description="Store ID for the store"),
):
    """Predict sales for a specific item in a specific store."""
    if not validate_date(ds):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Prepare input for prediction
    input_data = pd.DataFrame({
        'ds': [ds],  # Date
        'item_id': [item_id],  # Item ID
        'store_id': [store_id]  # Store ID
    })

    # Use the prophet_predictive_model for predictions
    predictive_model = app.state.models['prophet_predictive_model']
    prediction = predict_sales(predictive_model, input_data)
    return {"model": "prophet_predictive_model", "prediction": prediction}

# FastAPI endpoint for forecasting total sales across all stores and items
@app.get("/sales/national/")
async def forecast_total_sales(start_date: str, models: str = "prophet"):
    """
    Forecast total sales across all stores and items for the next 7 days from the given start date.

    Args:
        start_date (str): The starting date for the forecast in YYYY-MM-DD format.
        models (str, optional): Comma-separated list of model names to use for forecasting. 
                                Defaults to "prophet". Available models are:
                                - prophet
                                - prophet_event
                                - prophet_holiday
                                - prophet_month

    Returns:
        dict: A dictionary containing the forecasts from the specified models.
    """
    # Split the models string into a list
    selected_models = models.split(",")
    
    # Initialize a dictionary to hold forecasts
    forecasts = {}
    
    # Iterate through selected models and validate them
    for model_name in selected_models:
        model_name = model_name.strip()  # Remove any leading/trailing whitespace
        if model_name not in app.state.models:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not available.")
        
        # Retrieve the model
        forecasting_model = app.state.models[model_name]
        
        # Generate the forecast
        forecast = forecast_sales(forecasting_model, start_date=start_date, period=7)
        forecasts[model_name] = forecast  # Store the forecast in the dictionary

    return {"forecasts": forecasts}
    
# Run the application if executed directly
if __name__ == "__main__":
    # Configure the port and host dynamically
    port = int(os.getenv("PORT", 8000))  # Get the port from environment variables or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
