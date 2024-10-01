from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib  
import pandas as pd  
import os  
from pathlib import Path
import uvicorn
import pickle

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
models = {}

def load_model(model_name: str, model_path: str):
    """Load a prediction model from a file."""
    try:
        model = pickle.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        return None

# On startup, load all models
from pathlib import Path
import os

models = {}

# On startup, load all models
@app.on_event("startup")
async def startup_event():
    global models

    # Get the root path by going two levels up
    root = Path(os.getcwd()).parent.parent
    # Use os.path.join to create the path to the "models" directory
    dataset_path = root / "models"  # Convert to Path object

    # List of model filenames
    model_files = {
        'prophet': 'prophet.pkl',
        'prophet_event': 'prophet_event.pkl',
        'prophet_holiday': 'prophet_holiday.pkl',
        'prophet_month': 'prophet_month.pkl'
    }

    # Load the models from the 'models' directory using Path objects
    for model_name, filename in model_files.items():
        model_path = dataset_path / filename
        if os.path.exists(model_path):
            models[model_name] = load_model(model_name, model_path)
            print(f"{model_name} model loaded successfully.")
        else:
            print(f"Error: Model file {model_path} does not exist.")
            models[model_name] = None  # Set to None if loading fails

    # Check if all models are loaded correctly
    for model_name, model in models.items():
        if model is None:
            print(f"Warning: {model_name} model failed to load.")
            
@app.get("/")
async def read_root():
    """Return a welcome message at the root endpoint and the project root path."""
    
    # Get the current working directory
    current_directory = Path(os.getcwd())
    print(f"Current Working Directory: {current_directory}")

    # Navigate to the project root (two levels up)
    root = current_directory.parent.parent
    
    # Print the resolved root path for debugging
    print(f"Resolved Root Path: {root}")

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
        return predictions.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Endpoint for national sales forecast (uses query parameters)
@app.get("/v1/sales/national/")
async def national_sales_forecast(
    date: str = Query(..., description="Date for prediction in YYYY-MM-DD format"),
    item_id: str = Query(..., description="Item ID for the product"),
    store_id: str = Query(..., description="Store ID for the store"),
    model_type: str = Query('prophet', description="Model type (default: prophet)")
):
    """Get national sales forecast using the specified model."""
    if model_type not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found.")

    # Prepare input for prediction
    forecast_input = pd.DataFrame({
        'date': [date],
        'item_id': [item_id],
        'store_id': [store_id]
    })

    pred = predict(models[model_type], forecast_input)
    return {"model": model_type, "prediction": pred}

# Endpoint for predicting sales based on store and item (POST request for input data)
@app.post("/v1/sales/stores/items/")
async def predict_sales(
    date: str = Query(..., description="Date for prediction in YYYY-MM-DD format"),
    item_id: str = Query(..., description="Item ID for the product"),
    store_id: str = Query(..., description="Store ID for the store"),
    model_type: str = Query('prophet', description="Model type (default: prophet)")
):
    """Predict sales based on store and item using a POST request."""
    if model_type not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found.")
    
    # Prepare input for prediction
    predictive_input = pd.DataFrame({
        'date': [date],
        'item_id': [item_id],
        'store_id': [store_id]
    })

    pred = predict(models[model_type], predictive_input)
    return {"model": model_type, "prediction": pred}

# Run the application if executed directly
if __name__ == "__main__":
    # Configure the port and host dynamically
    port = int(os.getenv("PORT", 8000))  # Get the port from environment variables or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
