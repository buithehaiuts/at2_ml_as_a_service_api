import streamlit as st
import requests

# Streamlit app layout
st.title("Sales Revenue Forecasting and Prediction")

# Section 1: Introduction
st.markdown("""
## Welcome to the Sales Revenue Prediction and Forecasting App

This app provides two key functionalities:
- Predict sales revenue for a specific item at a given store and date.
- Forecast total sales revenue across all stores for the next 7 days.

### API Endpoints:
- `/`: Project objectives, input parameters, and output formats.
- `/health/`: API health check.
- `/sales/national/`: Forecast sales for the next 7 days (national level).
- `/sales/stores/items/`: Predict sales for specific stores and items.

Expected input for both endpoints:
- **Date (YYYY-MM-DD)**
- **Store ID and Item ID** (for store-specific predictions)
""")

# Section 2: Health Check
if st.button("Check API Health"):
    url = "http://127.0.0.1:8000/health"  # FastAPI health endpoint
    response = requests.get(url)
    
    if response.status_code == 200:
        st.success(f"API is healthy: {response.text}")
    else:
        st.error(f"Error: {response.status_code}")

# Section 3: National Sales Forecast
st.header("National Sales Forecast")
date_forecast = st.date_input("Select date for 7-day forecast (YYYY-MM-DD)")

if st.button("Get National Forecast"):
    url = f"http://127.0.0.1:8000/sales/national?date={date_forecast}"
    response = requests.get(url)
    
    if response.status_code == 200:
        forecast = response.json()
        st.success("7-day Sales Forecast:")
        st.json(forecast)
    else:
        st.error(f"Error: {response.status_code}")

# Section 4: Store and Item Sales Prediction
st.header("Store and Item Sales Prediction")
store_id = st.number_input("Store ID", value=1)
item_id = st.number_input("Item ID", value=1)
date_prediction = st.date_input("Select date for prediction (YYYY-MM-DD)")

input_data = {
    'store_id': store_id,
    'item_id': item_id,
    'date': str(date_prediction)
}

if st.button("Get Item Prediction"):
    # Call FastAPI for store-item sales prediction
    with st.spinner("Calling FastAPI..."):
        url = "http://127.0.0.1:8000/sales/stores/items"
        response = requests.get(url, params=input_data)
        
        # Show prediction
        if response.status_code == 200:
            prediction = response.json().get('prediction', 'No prediction found.')
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"Error: {response.status_code}")

# Section 5: Error Handling and Feedback
st.markdown("""
### Notes:
- Ensure that the API is running before making requests.
- Input the correct store ID, item ID, and date formats.
""")
