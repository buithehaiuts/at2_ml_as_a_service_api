from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List
import joblib  # or you can use pickle
import pandas as pd  # Assuming you'll use pandas for data handling
import os  # For constructing file paths

# Define the models
class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str

class Sale(BaseModel):
    id: int
    amount: float
    date: str

class SalesResponse(BaseModel):
    sales: List[Sale]

# Create FastAPI instance
app = FastAPI()

# Load models
def load_model(model_path: str):
    """Load the prediction model from a file."""
    try:
        model = joblib.load(model_path)  # Load the model
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

# Load the models
prophet_model = load_model(os.path.join('models', 'prophet.pkl'))  # Load the model
prophet_event_model = load_model(os.path.join('models', 'prophet_event.pkl'))  # Load event model
prophet_holiday_model = load_model(os.path.join('models', 'prophet_holiday.pkl'))  # Load holiday model
prophet_month_model = load_model(os.path.join('models', 'prophet_month.pkl'))  # Load month model

# Health check endpoint
@app.get(
    "/health/",
    tags=["healthcheck"],
    response_model=HealthCheck,
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
async def health_check():
    """Health Check endpoint."""
    return HealthCheck(status="OK")

# Endpoint for national sales forecast
@app.get("/sales/national/")
async def national_sales_forecast(date: str, item_id: str, store_id: str):
    """Get national sales forecast."""
    # Prepare input for prediction
    format_data = {
        'date': [date],
        'item_id': [item_id],
        'store_id': [store_id]
    }
    forecast_input = pd.DataFrame(format_data)  # Convert to DataFrame for prediction
    pred = predict(prophet_model, forecast_input)  # Call predict method with model and input
    return {"prediction": pred}

# Endpoint for predicting sales based on store and item
@app.post("/sales/stores/items/")
async def predict_sales(date: str, item_id: str, store_id: str):
    """Predict sales based on store and item."""
    predictive_input = {
        'date': [date],
        'item_id': [item_id],
        'store_id': [store_id]
    }
    predictive_df = pd.DataFrame(predictive_input)  # Convert to DataFrame for prediction
    pred = predict(prophet_model, predictive_df) 
    return {"prediction": pred}

def predict(model, input):
    """Ensure that you have the logic to predict using the model."""
    if model is not None:
        return model.predict(input)  # Use the model to make a prediction
    else:
        return {"error": "Model not loaded"}

# Run the application if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
