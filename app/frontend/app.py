import os
import streamlit as st
import requests
from datetime import datetime

# API URL for the FastAPI backend
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")  # Adjust as needed

st.title("Sales Prediction App")

# Function to fetch store and item data from the backend
def fetch_store_item_data():
    try:
        with st.spinner("Fetching store and item data..."):
            response = requests.get(f"{FASTAPI_URL}/data/display/train/")
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching store and item data: {e}")
        return None

# Load store and item data when the app runs
store_item_data = fetch_store_item_data()

if store_item_data is not None:
    st.write("Store and Item Data:")
    st.json(store_item_data)  # Display the fetched data in a JSON format
    
    with st.form(key='data_check_form'):
        # User input for date with a text box
        date = st.date_input("Select Date", value=datetime.now())

        # Dropdown for Store ID
        store_ids = [1, 2, 3]  # Replace with actual store IDs fetched from your data
        store_id = st.selectbox("Select Store ID", options=store_ids)

        # Dropdown for Item ID
        item_ids = [101, 102, 103]  # Replace with actual item IDs fetched from your data
        item_id = st.selectbox("Select Item ID", options=item_ids)

        check_button = st.form_submit_button("Check Data")

        if check_button:
            # Display selected store and item information (mockup)
            st.write(f"Checking data for Store ID: {store_id}, Item ID: {item_id} on date: {date}")
            # You can add additional logic here to retrieve and display specific data for the given store and item
else:
    st.error("Failed to load store and item data.")
