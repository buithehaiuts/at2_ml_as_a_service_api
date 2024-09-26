import streamlit as st
import requests

# Set the API URL for the FastAPI backend
API_URL = "https://at2-ml-as-a-service-api.onrender.com"

# Streamlit app layout
st.title("Sales Revenue Forecasting and Prediction")

# Organize sections into tabs
selected_tab = st.sidebar.radio("Navigation", ["API Health Check", "National Sales Forecast", "Store & Item Prediction", "Instructions"])

# Tab 1: API Health Check
if selected_tab == "API Health Check":
    st.header("API Health Check")
    
    if st.button("Check API Health"):
        url = f"{API_URL}/health"  # FastAPI health endpoint
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            st.success(f"API is healthy: {response.json()}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

# Tab 2: National Sales Forecast
if selected_tab == "National Sales Forecast":
    st.header("National Sales Forecast")
    
    # Input field for date
    date_forecast = st.date_input("Select date for 7-day forecast (YYYY-MM-DD)")
    
    # Button to trigger forecast
    if st.button("Get National Forecast"):
        url = f"{API_URL}/sales/national?date={date_forecast}"  # Ensure the endpoint is correct
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            
            # Check for empty response
            if response.text:
                forecast = response.json()
                st.success("7-day Sales Forecast:")
                st.json(forecast)
            else:
                st.error("Received an empty response from the API.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
            st.write("Response Status Code:", response.status_code if 'response' in locals() else "N/A")
            st.write("Response Text:", response.text if 'response' in locals() else "N/A")

# Tab 3: Store & Item Sales Prediction
if selected_tab == "Store & Item Prediction":
    st.header("Store and Item Sales Prediction")
    
    # Input fields for store ID, item ID, and date
    store_id = st.number_input("Store ID", value=1, min_value=1)
    item_id = st.number_input("Item ID", value=1, min_value=1)
    date_prediction = st.date_input("Select date for prediction (YYYY-MM-DD)")
    
    # Prepare input data as a dictionary
    input_data = {
        'store_id': store_id,
        'item_id': item_id,
        'date': str(date_prediction)
    }
    
    # Button to trigger prediction
    if st.button("Get Item Prediction"):
        with st.spinner("Calling FastAPI..."):
            url = f"{API_URL}/sales/stores/items/"  # Ensure this endpoint is correct
            try:
                response = requests.post(url, json=input_data)  # Use POST instead of GET
                response.raise_for_status()  # Raise an error for bad responses
                
                # Check for empty response
                if response.text:
                    prediction = response.json().get('prediction', 'No prediction found.')
                    st.success(f"Prediction: {prediction}")
                else:
                    st.error("Received an empty response from the API.")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")
                st.write("Response Status Code:", response.status_code if 'response' in locals() else "N/A")
                st.write("Response Text:", response.text if 'response' in locals() else "N/A")

# Tab 4: Instructions
if selected_tab == "Instructions":
    st.header("Instructions and Notes")
    
    st.markdown(""" 
    ### How to use the app:
    1. **API Health Check**: Verify if the FastAPI backend is running and accessible.
    2. **National Sales Forecast**: Get a 7-day sales forecast for all stores based on the selected date.
    3. **Store & Item Sales Prediction**: Predict sales for a specific item at a specific store on the selected date.
    
    ### API Endpoints:
    - `/`: Project objectives, input parameters, and output formats.
    - `/health/`: API health check.
    - `/sales/national/`: Forecast sales for the next 7 days (national level).
    - `/sales/stores/items/`: Predict sales for specific stores and items.
    
    ### Important Notes:
    - Ensure that the API is running before making requests.
    - Input the correct store ID, item ID, and date formats.
    - For testing purposes, the app is configured to connect to `https://at2-ml-as-a-service-api.onrender.com`.
    """)
