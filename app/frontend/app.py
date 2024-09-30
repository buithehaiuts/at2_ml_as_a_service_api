import streamlit as st
import requests
from datetime import date

# Base URL for the FastAPI service
BASE_URL = "https://fastapi-backend.onrender.com"

st.title("Sales Prediction Application")

# Helper function for API requests
def fetch_api(url, method='get', json_data=None, params=None):
    try:
        if method == 'get':
            response = requests.get(url, params=params)  # Use params for GET requests
        else:
            response = requests.post(url, json=json_data)
        
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
        return None

# Health Check
if st.button("Check API Health"):
    health_data = fetch_api(f"{BASE_URL}/health/")
    if health_data and health_data.get("status") == "healthy":
        st.success("API is healthy!")
    else:
        st.error("API is not responding!")

# National Sales Forecast
st.header("National Sales Forecast")
date_input = st.date_input("Select a date for forecast", min_value=date.today())
item_id = st.number_input("Enter Item ID", min_value=1)
store_id = st.number_input("Enter Store ID", min_value=1)

if st.button("Get National Sales Forecast"):
    if store_id <= 0 or item_id <= 0:
        st.error("Store ID and Item ID must be greater than zero.")
    else:
        with st.spinner("Fetching forecast..."):
            # Fetch forecast using the provided date, item_id, and store_id
            forecast_data = fetch_api(
                f"{BASE_URL}/sales/national/",
                method='get',
                params={"date": str(date_input), "item_id": str(item_id), "store_id": str(store_id)}
            )
            if forecast_data:
                st.write("Sales Forecast:")
                st.json(forecast_data)

# Store and Item Sales Prediction
st.header("Store Item Sales Prediction")
date_input = st.date_input("Select date for prediction", min_value=date.today())
store_id = st.number_input("Enter Store ID", min_value=1)
item_id = st.number_input("Enter Item ID", min_value=1)

if st.button("Predict Sales"):
    if store_id <= 0 or item_id <= 0:
        st.error("Store ID and Item ID must be greater than zero.")
    else:
        # Prepare the payload for prediction
        payload = {
            "date": str(date_input),  # Date for prediction
            "item_id": str(item_id),  # Item ID
            "store_id": str(store_id)  # Store ID
        }
        with st.spinner("Predicting sales..."):
            prediction_data = fetch_api(f"{BASE_URL}/sales/stores/items/", method='post', json_data=payload)
            if prediction_data:
                st.write("Sales Prediction:")
                st.json(prediction_data)
