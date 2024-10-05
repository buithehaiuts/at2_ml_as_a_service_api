from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel
from typing import List, Dict, Any
import pickle  
import pandas as pd  
import os  
from pathlib import Path
import uvicorn
import json
import logging
from datetime import datetime, timedelta
from lightgbm import LGBMRegressor
import numpy as np

# Set up logging for better traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the FastAPI application
app = FastAPI(
    title="Sales Forecast and Prediction API",
    description="API for forecasting and predicting sales revenue using machine learning and time-series models.",
    version="1.0.0"
)

# Load models and encoders into a centralized state dictionary
app.state.models = {}
app.state.encoders = {}

# Load the id_values.json file
with open('app/backend/id_values.json', 'r') as f:
    id_values = json.load(f)

# Extracting values for dropdowns from JSON
item_ids = id_values.get("item_id", [])
store_ids = id_values.get("store_id", [])
state_ids = id_values.get("state_id", [])
cat_ids = id_values.get("cat_id", [])
dept_ids = id_values.get("dept_id", [])

# Define Pydantic models for input and output
class HealthCheck(BaseModel):
    """Model for health check response."""
    status: str

def load_model(model_path: str):
    """Load a prediction model from a file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)
        
def validate_date(date_str: str) -> bool:
    """Validate the date format to ensure it follows YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def safe_transform(encoder, data):
    try:
        return encoder.transform(data)
    except ValueError as e:
        print(f"Warning: {e}")
        # Optionally handle unseen categories here (e.g., return a default value or np.nan)
        return np.full(data.shape, -1)


@app.on_event("startup")        
async def startup_event():
    """Load models and encoders during startup."""
    
    # Initialize app state
    app.state.models = {}
    app.state.encoders = {}

    # Define model file paths using Path
    model_files = {
        'prophet': Path('models/prophet.pkl').resolve(),
        'prophet_event': Path('models/prophet_event.pkl').resolve(),
        'prophet_holiday': Path('models/prophet_holiday.pkl').resolve(),
        'prophet_month': Path('models/prophet_month.pkl').resolve(),
        'predictive_lgbm': Path('models/predictive_lgbm.pkl').resolve()
    }
    
    for model_name, model_path in model_files.items():
        if model_path.exists():
            try:
                if model_name == 'predictive_lgbm':
                    with open(model_path, 'rb') as f:
                        model, scaler = pickle.load(f)  
                        app.state.models[model_name] = {'model': model, 'scaler': scaler}
                    logger.info(f"{model_name} model and scaler loaded successfully from {model_path}.")
                else:
                    app.state.models[model_name] = load_model(str(model_path))
                    logger.info(f"{model_name} model loaded successfully from {model_path}.")
            except FileNotFoundError:
                logger.error(f"Model file not found: {model_path}.")
            except pickle.UnpicklingError:
                logger.error(f"Failed to unpickle model: {model_name} from {model_path}.")
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
        else:
            logger.warning(f"Model file does not exist: {model_path}")

    # Load encoders
    encoder_files = {
        'item_encoder': Path('app/backend/item_encoder.pkl').resolve(),
        'store_encoder': Path('app/backend/store_encoder.pkl').resolve(),
        'state_encoder': Path('app/backend/state_encoder.pkl').resolve(),
        'cat_encoder': Path('app/backend/cat_id.pkl').resolve(),
        'dept_encoder': Path('app/backend/dept_id_encoder.pkl').resolve()
    }

    for encoder_name, encoder_path in encoder_files.items():
        try:
            with open(encoder_path, 'rb') as f:
                app.state.encoders[encoder_name] = pickle.load(f)
            logger.info(f"{encoder_name} loaded successfully from {encoder_path}.")
        except FileNotFoundError:
            logger.error(f"Encoder file not found: {encoder_path}.")
        except pickle.UnpicklingError:
            logger.error(f"Failed to unpickle encoder: {encoder_name} from {encoder_path}.")
        except Exception as e:
            logger.error(f"Error loading {encoder_name}: {str(e)}")
            
@app.get("/")
async def read_root():
    """Return project objectives and API details."""
    project_info = {
        "project_objectives": "This API provides sales forecasting and store-item sales prediction. "
                              "It allows users to forecast national sales for the next 7 days and predict sales "
                              "for specific store items on a given date.",
        "endpoints": {
            "/": "Displays project objectives, API details, and GitHub repo link.",
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
                    "item_id": "Item ID",
                    "state_id": "State ID",
                    "cat_id": "Category ID",
                    "dept_id": "Department ID"
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
def predict_sales(scaler,model, input_data):
    input_data_scaled = scaler.transform(input_data)
    # Prepare input data (make sure to preprocess it similarly to the training data)
    predicted_sales = model.predict(input_data_scaled)
    return predicted_sales if isinstance(predicted_sales, np.float64) else predicted_sales.tolist()

def prepare_input_data(item_id, store_id, state_id, cat_id, dept_id, date):
    # Parse the date and extract day, month, year
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    day = date_obj.day
    month = date_obj.month
    year = date_obj.year
    
    # Create a DataFrame for input
    input_data = pd.DataFrame({
        'item_id': [item_id],
        'store_id': [store_id],
        'state_id': [state_id],
        'cat_id': [cat_id],
        'dept_id': [dept_id],
        'day': [day],
        'month': [month],
        'year': [year]
    })
    
    # Transform categorical columns using encoders from app state
    input_data['item_id'] = safe_transform(app.state.encoders['item_encoder'], input_data['item_id'])
    input_data['store_id'] = safe_transform(app.state.encoders['store_encoder'], input_data['store_id'])
    input_data['state_id'] = safe_transform(app.state.encoders['state_encoder'], input_data['state_id'])
    input_data['cat_id'] = safe_transform(app.state.encoders['cat_encoder'], input_data['cat_id'])
    input_data['dept_id'] = safe_transform(app.state.encoders['dept_encoder'], input_data['dept_id'])

    expected_columns = ['item_id', 'store_id', 'dept_id', 'cat_id', 'state_id', 'day', 'month', 'year']
    input_data=input_data[expected_columns]
    
    return input_data

# Forecast function for total sales across all stores and items
def forecast_sales(model, start_date: str, period: int = 7) -> List[Dict[str, Any]]:
    """Make a sales forecast using the selected time-series model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        forecast_start_date = start_date_dt + timedelta(days=1)

        future_dates = model.make_future_dataframe(periods=period)
        future_dates['ds'][-period:] = [forecast_start_date + timedelta(days=i) for i in range(period)]
        
        train_forecast = model.predict(future_dates)
        output = train_forecast[train_forecast['ds'] >= forecast_start_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')

        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {str(e)}")

# Endpoint for predicting sales for a specific item in a store (GET request)
@app.get("/sales/stores/items/")
async def predict_item_sales(
    date: str = Query(..., description="Date for prediction in YYYY-MM-DD format"),
    item_id: str = Query(..., description="Item ID for the product", enum=item_ids),
    store_id: str = Query(..., description="Store ID for the specific location", enum=store_ids),
    state_id: str = Query(..., description="State ID for the location", enum=state_ids),
    cat_id: str = Query(..., description="Category ID for the product", enum=cat_ids),
    dept_id: str = Query(..., description="Department ID for the product", enum=dept_ids)
    ):

    # Validate the input date
    if not validate_date(date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    # Create a DataFrame from the input data
    input_data = prepare_input_data(item_id, store_id, state_id, cat_id, dept_id, date)

    try:
        # Predict sales using the loaded LightGBM model
        model_info = app.state.models['predictive_lgbm']
        model=model_info['model']
        scaler=model_info['scaler']
        predictions = predict_sales(scaler,model, input_data)

        # Prepare response
        return predictions

    except Exception as e:
        logger.error(f"Error predicting sales for item: {str(e)}")
        raise HTTPException(status_code=500, detail="Error predicting sales.")

# Endpoint for forecasting total sales for the next 7 days (GET request)
@app.get("/sales/national/")
async def forecast_national_sales(
    date: str = Query(..., description="Start date for forecasting in YYYY-MM-DD format"),
    model_type: str = Query("prophet", description="Type of model to use for forecasting. Options: 'prophet', 'prophet_event', 'prophet_holiday', 'prophet_month'.")
):
    """Forecasts the total sales for the next 7 days starting from the input date."""
    # Validate the input date
    if not validate_date(date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Validate the model_type
    if model_type not in app.state.models:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Available options are: {', '.join(app.state.models.keys())}.")
        
    # Load the Prophet model for national sales
    prophet_model = app.state.models[model_type]

    try:
        forecast_data = forecast_sales(prophet_model, date)
        return {"forecasts": forecast_data}

    except Exception as e:
        logger.error(f"Error forecasting national sales: {str(e)}")
        raise HTTPException(status_code=500, detail="Error forecasting national sales.")

# Entry point for running the FastAPI application
if __name__ == "__main__":
    # Start the server with UVICORN
    uvicorn.run(app, host="0.0.0.0", port=8000)
