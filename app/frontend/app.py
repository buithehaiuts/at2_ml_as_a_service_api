import streamlit as st
import requests
from datetime import date

# Update BASE_URL to the service name defined in your Docker Compose
BASE_URL = "http://fastapi-backend:8000"  # Use the service name for Docker networking

st.title("Sales Prediction Application")

# Helper function for API requests
def fetch_api(url, method='get', json_data=None):
    try:
        if method == 'get':
            response = requests.get(url)
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
if st.button("Get National Sales Forecast"):
    with st.spinner("Fetching forecast..."):
        forecast_data = fetch_api(f"{BASE_URL}/sales/national/?date={date_input}")
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
        payload = {
            "date": str(date_input),
            "store_id": store_id,
            "item_id": item_id
        }
        with st.spinner("Predicting sales..."):
            prediction_data = fetch_api(f"{BASE_URL}/sales/stores/items/", method='post', json_data=payload)
            if prediction_data:
                st.write("Sales Prediction:")
                st.json(prediction_data)
