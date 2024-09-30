import streamlit as st
import requests
from datetime import date

# Base URL for the FastAPI service
BASE_URL = "https://fastapi-backend-mslg.onrender.com"

st.title("Sales Prediction Application")

# Helper function for API requests
def fetch_api(url, method='get', json_data=None, params=None):
    try:
        if method == 'get':
            response = requests.get(url, params=params)
        else:
            response = requests.post(url, json=json_data)
        
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Request error occurred: {req_err}")
    except Exception as err:
        st.error(f"An error occurred: {err}")
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
item_id = st.number_input("Enter Item ID", min_value=1, key="national_item_id")  # Unique key
store_id = st.number_input("Enter Store ID", min_value=1, key="national_store_id")  # Unique key

if st.button("Get National Sales Forecast"):
    if store_id <= 0 or item_id <= 0:
        st.error("Store ID and Item ID must be greater than zero.")
    else:
        with st.spinner("Fetching forecast..."):
            forecast_data = fetch_api(
                f"{BASE_URL}/sales/national/",
                method='get',
                params={"date": str(date_input), "item_id": str(item_id), "store_id": str(store_id)}
            )
            if forecast_data:
                st.write("Sales Forecast:")
                st.json(forecast_data)

# Store Item Sales Prediction
st.header("Store Item Sales Prediction")
date_input = st.date_input("Select date for prediction", min_value=date.today())
store_id = st.number_input("Enter Store ID", min_value=1, key="store_item_store_id")  # Unique key
item_id = st.number_input("Enter Item ID", min_value=1, key="store_item_item_id")  # Unique key

if st.button("Predict Sales"):
    if store_id <= 0 or item_id <= 0:
        st.error("Store ID and Item ID must be greater than zero.")
    else:
        payload = {
            "date": str(date_input),
            "item_id": str(item_id),
            "store_id": str(store_id)
        }
        with st.spinner("Predicting sales..."):
            prediction_data = fetch_api(f"{BASE_URL}/sales/stores/items/", method='post', json_data=payload)
            if prediction_data:
                st.write("Sales Prediction:")
                st.json(prediction_data)
