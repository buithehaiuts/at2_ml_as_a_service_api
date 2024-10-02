from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel
from typing import List, Dict, Any
import pickle  
import pandas as pd  
import os  
from pathlib import Path
import uvicorn
import logging
from datetime import datetime

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
    title="Sales Forecast API",
    description="API for forecasting sales using various models.",
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
        'prophet': 'models/prophet.pkl',
        'prophet_event': 'models/prophet_event.pkl',
        'prophet_holiday': 'models/prophet_holiday.pkl',
        'prophet_month': 'models/prophet_month.pkl',
        'prophet_predictive_model':'models/prophet_predictive_model.pkl'
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
    """Return a welcome message at the root endpoint and the project root path."""
    logger.info(f"Current Working Directory: {current_directory}")
    
    # Navigate to the project root (two levels up)
    root = current_directory.parent.parent
    
    # Print the resolved root path for debugging
    logger.info(f"Resolved Root Path: {root}")

    return {
        "message": "Welcome to the Sales Forecast API!",
        "root": str(root)  # Convert Path to string for the response
    }

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

# Prediction function
def predict(model, input: pd.DataFrame) -> List[Dict[str, Any]]:
    """Make a prediction using the selected model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        predictions = model.predict(input)
        
        # Check if predictions are a Series or DataFrame
        if isinstance(predictions, pd.Series):
            return predictions.tolist()  # This works on Series
        elif isinstance(predictions, pd.DataFrame):
            return predictions.values.tolist()  # Use .values for DataFrame, then convert to list
        else:
            raise HTTPException(status_code=500, detail="Unknown prediction output format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Endpoint for national sales forecast (uses query parameters)
@app.get("/v1/sales/national/")
async def national_sales_forecast(
    ds: str = Query(..., description="Date for prediction in YYYY-MM-DD format"),
    item_id: str = Query(..., description="Item ID for the product"),
    store_id: str = Query(..., description="Store ID for the store"),
    model_type: str = Query('prophet', description="Model type (default: prophet)")
):
    """Get national sales forecast using the specified model."""
    if not validate_date(ds):  # Corrected the variable name from 'date' to 'ds'
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if model_type not in app.state.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found.")

    # Prepare input for prediction
    forecast_input = pd.DataFrame({
        'ds': [ds],
        'item_id': [item_id],
        'store_id': [store_id]
    })

    pred = predict(app.state.models[model_type], forecast_input)
    return {"model": model_type, "prediction": pred}

# Endpoint for predicting sales based on store and item (POST request for input data)
@app.post("/v1/sales/stores/items/")
async def predict_sales(
    ds: str = Query(..., description="Date for prediction in YYYY-MM-DD format"),
    item_id: str = Query(..., description="Item ID for the product"),
    store_id: str = Query(..., description="Store ID for the store"),
    model_type: str = Query('prophet', description="Model type (default: prophet)")
):
    """Predict sales based on store and item using a POST request."""
    if not validate_date(ds):  # Ensure correct date validation
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if model_type not in app.state.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found.")
    
    # Prepare input for prediction
    predictive_input = pd.DataFrame({
        'ds': [ds],
        'item_id': [item_id],
        'store_id': [store_id]
    })

    pred = predict(app.state.models[model_type], predictive_input)
    return {"model": model_type, "prediction": pred}

# Run the application if executed directly
if __name__ == "__main__":
    # Configure the port and host dynamically
    port = int(os.getenv("PORT", 8000))  # Get the port from environment variables or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
