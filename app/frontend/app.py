import os
import streamlit as st
import requests
from datetime import datetime

# API URL for the FastAPI backend
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")  # Adjust as needed

st.title("Sales Prediction App")

with st.form(key='prediction_form'):
    date = st.text_input("Date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))
    store_id = st.number_input("Store ID", min_value=1)
    item_id = st.number_input("Item ID", min_value=1)
    submit_button = st.form_submit_button("Predict Sales")

    if submit_button:
        try:
            datetime.strptime(date, "%Y-%m-%d")

            with st.spinner("Fetching data..."):
                response = requests.get(f"{FASTAPI_URL}/sales/stores/items/", params={
                    "date": date,
                    "store_id": store_id,
                    "item_id": item_id
                })
                response.raise_for_status()

                prediction = response.json()
                st.success(f"Predicted sales: ${prediction['prediction']:.2f}")

        except ValueError:
            st.error("Invalid date format. Please enter a date in YYYY-MM-DD format.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error making request to backend: {e}")
