from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib  # or you can use pickle
import pandas as pd  # Assuming you'll use pandas for data handling
import os  # For constructing file paths

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

# Load the models from the 'models' directory
prophet_model = load_model(os.path.join('models', 'prophet.pkl'))
prophet_event_model = load_model(os.path.join('models', 'prophet_event.pkl'))
prophet_holiday_model = load_model(os.path.join('models', 'prophet_holiday.pkl'))
prophet_month_model = load_model(os.path.join('models', 'prophet_month.pkl'))

# Startup event to print registered routes
@app.on_event("startup")
async def startup_event():
    print("Registered routes:")
    for route in app.routes:
        print(route)

# Root endpoint
@app.get("/")
async def read_root():
    """Return a welcome message at the root endpoint."""
    return {"message": "Welcome to the Sales Forecast API!"}

# Health check endpoint
@app.get(
    "/health/",
    tags=["healthcheck"],
    response_model=HealthCheck,
    summary="Health Check Performed",
    response_description="Returns HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
async def health_check():
    """Health Check endpoint."""
    return HealthCheck(status="OK")

# Prediction function
def predict(model, input: pd.DataFrame) -> Dict[str, Any]:
    """Make a prediction using the model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        predictions = model.predict(input)
        return predictions.tolist()  # Convert predictions to list for JSON response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

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

# Run the application if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Get the port from environment variable or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
