from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib  # or you can use pickle
import pandas as pd  # Assuming you'll use pandas for data handling
import os  # For constructing file paths

# Pydantic model for a Sale
class Sale(BaseModel):
    id: int
    amount: float
    date: str

# Pydantic model for the response containing multiple sales
class SalesResponse(BaseModel):
    sales: List[Sale]

# Main FastAPI application class
class SalesAPI:
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()
        self.prophet_model = self.load_model(os.path.join('models', 'prophet.pkl'))  # Load the model
        self.prophet_event_model = self.load_model(os.path.join('models', 'prophet_event.pkl'))  # Load event model
        self.prophet_holiday_model = self.load_model(os.path.join('models', 'prophet_holiday.pkl'))  # Load holiday model
        self.prophet_month_model = self.load_model(os.path.join('models', 'prophet_month.pkl'))  # Load month model

    def load_model(self, model_path: str):
        """Load the prediction model from a file."""
        try:
            model = joblib.load(model_path)  # Load the model
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None

    def setup_routes(self):
        # Health check endpoint
        @self.app.get("/health/")
        def health_check():
            return {"status": "healthy"}

        # Endpoint for national sales forecast
        @self.app.get("/sales/national/")
        async def national_sales_forecast(date: str, item_id: str, store_id: str):
            # Prepare input for prediction
            format_data = {
                'date': [date],
                'item_id': [item_id],
                'store_id': [store_id]
            }
            forecast_input = pd.DataFrame(format_data)  # Convert to DataFrame for prediction
            pred = self.predict(self.prophet_model, forecast_input)  # Call predict method with model and input
            return {"prediction": pred}

        # Endpoint for predicting sales based on store and item
        @self.app.post("/sales/stores/items/")
        async def predict_sales(date: str, item_id: str, store_id: str):
            # Implement your logic for predicting sales here
            predictive_input = {
                'date': [date],
                'item_id': [item_id],
                'store_id': [store_id]
            }
            predictive_df = pd.DataFrame(predictive_input)  # Convert to DataFrame for prediction
            pred = self.predict(self.prophet_model, predictive_df) 
            return {"prediction": pred}

    def predict(self, model, input):
        """Ensure that you have the logic to predict using the model."""
        if model is not None:
            return model.predict(input)  # Use the model to make a prediction
        else:
            return {"error": "Model not loaded"}

# This allows the FastAPI app to be imported and run directly
api = SalesAPI()

# Run the application if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api.app, host="0.0.0.0", port=8000)
