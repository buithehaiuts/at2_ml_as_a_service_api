import os
import streamlit as st
import requests
from datetime import datetime

# Use the FASTAPI_URL environment variable, default to FastAPI service in Docker
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")  # Change 'localhost' to 'fastapi' for Docker

# Streamlit app title
st.title("Sales Prediction App")

# Example form for input data
with st.form(key='prediction_form'):
    date = st.text_input("Date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))
    store_id = st.number_input("Store ID", min_value=1)
    item_id = st.number_input("Item ID", min_value=1)
    submit_button = st.form_submit_button("Predict Sales")

    if submit_button:
        # Validate the date format
        try:
            # Validate date format
            datetime.strptime(date, "%Y-%m-%d")
            st.write("Validating input...")
            
            # Show loading indicator
            with st.spinner("Fetching data..."):
                # Make a request to the FastAPI backend
                response = requests.get(f"{FASTAPI_URL}/sales/stores/items/", params={
                    "date": date,
                    "store_id": store_id,
                    "item_id": item_id
                })
                response.raise_for_status()  # Raise an error for bad responses
                
                # Display the queried data
                prediction = response.json()
                st.success(f"Predicted sales: ${prediction['prediction']:.2f}")
        
        except ValueError as ve:
            st.error(f"Invalid date format. Please enter a date in YYYY-MM-DD format.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error making request to backend: {e}")
