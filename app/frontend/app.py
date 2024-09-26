import os
import streamlit as st
import requests

# Use the FASTAPI_URL environment variable
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")  # Default for local testing

# Example Streamlit code to interact with the FastAPI backend
st.title("Sales Prediction App")

# Example form for input data
with st.form(key='prediction_form'):
    date = st.text_input("Date (YYYY-MM-DD)")
    store_id = st.number_input("Store ID", min_value=1)
    item_id = st.number_input("Item ID", min_value=1)
    submit_button = st.form_submit_button("Predict Sales")

    if submit_button:
        # Make a request to the FastAPI backend
        try:
            response = requests.get(f"{FASTAPI_URL}/sales/stores/items/", params={
                "date": date,
                "store_id": store_id,
                "item_id": item_id
            })
            response.raise_for_status()  # Raise an error for bad responses
            prediction = response.json()
            st.success(f"Predicted sales: {prediction['prediction']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error making request to backend: {e}")
