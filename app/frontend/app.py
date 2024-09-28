import os
import streamlit as st
import requests
from datetime import datetime
import logging

# API URL for the FastAPI backend
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")  # Use localhost for local access


# # API URL for the FastAPI backend
# FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi-backend:8000")  # Use the service name for internal communication

st.title("Sales Prediction App")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to fetch store and item data from the backend
def fetch_store_item_data():
    try:
        with st.spinner("Fetching store and item IDs..."):
            logging.info(f"Fetching store and item data from {FASTAPI_URL}/data/ids/")
            response = requests.get(f"{FASTAPI_URL}/data/ids/")
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching store and item data: {e}")
        logging.error(f"Request failed: {e}")
        return None

# Function to fetch training data
def fetch_training_data():
    try:
        with st.spinner("Fetching training data..."):
            response = requests.get(f"{FASTAPI_URL}/data/display/train/")
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching training data: {e}")
        return None

# Load store and item data when the app runs
store_item_data = fetch_store_item_data()
training_data = fetch_training_data()

if store_item_data is not None:
    store_ids = sorted(store_item_data['store_ids'])
    item_ids = sorted(store_item_data['item_ids'])

    with st.form(key='data_check_form'):
        # User input for date with a date picker
        date = st.date_input("Select Date", value=datetime.now())

        # Dropdown for Store ID
        store_id = st.selectbox("Select Store ID", options=store_ids)

        # Dropdown for Item ID
        item_id = st.selectbox("Select Item ID", options=item_ids)

        check_button = st.form_submit_button("Check Data")

        if check_button:
            # Display selected store and item information
            st.write(f"Checking data for Store ID: {store_id}, Item ID: {item_id} on date: {date}")

            # Call the sales prediction endpoint
            try:
                prediction_response = requests.get(f"{FASTAPI_URL}/sales/stores/items/", params={
                    "date": date.strftime('%Y-%m-%d'),
                    "store_id": store_id,
                    "item_id": item_id
                })
                prediction_response.raise_for_status()
                prediction_data = prediction_response.json()

                # Check if prediction data has the expected structure
                if 'prediction' in prediction_data:
                    st.write(f"Predicted Sales: {prediction_data['prediction']}")
                else:
                    st.error("Prediction data format is unexpected.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching prediction data: {e}")

else:
    st.error("Failed to load store and item data.")
