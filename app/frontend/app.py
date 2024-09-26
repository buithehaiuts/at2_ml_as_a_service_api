import streamlit as st
import requests
import os

# Base URL of the FastAPI backend
BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")  # Default to localhost for local testing

# Streamlit app layout
st.title("Sales Revenue Forecasting and Prediction")

# Input fields for prediction
date_input = st.date_input("Select a date")
store_id_input = st.number_input("Store ID", min_value=1)
item_id_input = st.number_input("Item ID", min_value=1)

# Button to make prediction
if st.button("Predict Sales"):
    # Prepare the request payload
    payload = {
        "date": date_input.isoformat(),  # Ensure date is in the correct format
        "store_id": store_id_input,
        "item_id": item_id_input
    }
    
    # Make a request to the FastAPI backend
    try:
        response = requests.post(f"{BASE_URL}/sales/stores/items/", json=payload)
        response.raise_for_status()  # Raise an error for bad responses

        prediction = response.json().get("prediction")
        if prediction is not None:
            st.success(f"Predicted Sales: {prediction}")
        else:
            st.warning("No prediction available.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error making request to backend: {e}")

# Optionally add more components, visualizations, etc.
