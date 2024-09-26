import os
import streamlit as st
import requests

# Base URL of the FastAPI backend
BASE_URL = os.getenv("FASTAPI_URL", "http://fastapi-backend:8000/api")  # Ensure it points to /api

# Example function to call the backend
def fetch_sales_prediction(date, store_id, item_id):
    try:
        response = requests.post(f"{BASE_URL}/sales/stores/items/", json={
            "date": date,
            "store_id": store_id,
            "item_id": item_id
        })
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error making request to backend: {e}")
