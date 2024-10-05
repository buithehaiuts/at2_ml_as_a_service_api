import streamlit as st
import requests
from datetime import datetime
import json
import pandas as pd
import altair as alt

# Set the title of the application
st.title("Sales Prediction and Forecasting App")

# Sidebar for navigation
selected_tab = st.sidebar.radio("Select a Tab", 
                                 ["API Health Check", 
                                  "Sales Prediction", 
                                  "Sales Forecasting", 
                                  "About"])

# API Base URL
base_url = "https://at2-ml-as-a-service-api-update-8ac8.onrender.com"

# --- API Health Check Tab ---
if selected_tab == "API Health Check":
    st.header("API Health Check")
    
    # Button to check API health status
    if st.button("Check API Health Status"):
        try:
            response = requests.get(f"{base_url}/health/")
            if response.status_code == 200:
                st.success("API Status: **Healthy**")
            else:
                st.error("API Status: **Unhealthy**")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

# --- Sales Prediction Tab ---
elif selected_tab == "Sales Prediction":
    st.header("Sales Prediction")

    # Input fields for sales prediction
    date_input = st.date_input("Select a date for prediction", value=datetime.today())
    date_str = date_input.strftime("%Y-%m-%d")

    # Fetch dropdown options from the API
    try:
        item_response = requests.get(f"{base_url}/sales/stores/items/")
        dropdowns = item_response.json()
        
        store_ids = dropdowns.get("store_id", [])
        state_ids = dropdowns.get("state_id", [])
        cat_ids = dropdowns.get("cat_id", [])
        dept_ids = dropdowns.get("dept_id", [])
        item_ids = dropdowns.get("item_id", [])  # Assuming item_id also comes from the API

        # Dropdown for store ID selection
        store_id = st.selectbox("Select Store ID", store_ids)

        # Dropdown for item ID selection
        item_id = st.selectbox("Select Item ID", item_ids)

        # Dropdown for state ID selection
        state_id = st.selectbox("Select State ID", state_ids)

        # Dropdown for category ID selection
        cat_id = st.selectbox("Select Category ID", cat_ids)

        # Dropdown for department ID selection
        dept_id = st.selectbox("Select Department ID", dept_ids)

        # Button to trigger sales prediction
        if st.button("Predict Sales"):
            # Send a GET request to the sales prediction API endpoint
            try:
                prediction_url = f"{base_url}/sales/stores/items/"
                # Prepare the parameters for the API call
                params = {
                    "date": date_str,
                    "item_id": item_id,
                    "store_id": store_id,
                    "state_id": state_id,
                    "cat_id": cat_id,
                    "dept_id": dept_id
                }
                
                # Send the GET request with parameters
                prediction_response = requests.get(prediction_url, params=params)
                
                if prediction_response.status_code == 200:
                    prediction_data = prediction_response.json()
                    st.write("Predicted Sales Data:")
                    st.write(prediction_data)
                else:
                    st.error("Error in fetching sales prediction.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching dropdown data: {e}")

# --- Sales Forecasting Tab ---
elif selected_tab == "Sales Forecasting":
    st.header("Sales Forecasting")
    
    # Input for forecast start date
    forecast_start_date = st.date_input("Select a forecast start date", value=datetime.today())
    forecast_date_str = forecast_start_date.strftime("%Y-%m-%d")

    # Dropdown for model type selection
    model_type = st.selectbox("Select Forecasting Model Type", 
                               ['prophet', 'prophet_event', 'prophet_holiday', 'prophet_month'])

    # Button to trigger sales forecasting
    if st.button("Forecast Sales"):
        # Send a request to the sales forecasting API endpoint
        try:
            forecast_url = f"{base_url}/sales/national/"
            params = {"date": forecast_date_str, "model_type": model_type}
            forecast_response = requests.get(forecast_url, params=params)

            # Handle response
            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()
                
                # Assuming the response structure is in the correct format
                if 'forecast' in forecast_data:
                    forecast_df = pd.DataFrame(forecast_data['forecast'])
                    forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date

                    # Display a success message
                    st.success("Forecasted Sales for the Next 7 Days:")

                    # Display table
                    st.table(forecast_df)

                    # Create the base chart
                    base = alt.Chart(forecast_df).encode(x='date:T')
                    
                    # Line for predicted sales
                    line = base.mark_line(color='blue').encode(y='sales:Q')

                    # Points for predicted sales
                    points = base.mark_circle(color='blue').encode(y='sales:Q')

                    # Combine all layers
                    chart = (line + points).properties(
                        title='Sales Forecast for the Next 7 Days',
                        width=700,
                        height=400
                    )

                    # Display the chart in Streamlit
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.error("No forecast data available.")
            else:
                st.error("Failed to retrieve forecast data.")
                st.json(forecast_response.json())
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

# --- About Tab ---
elif selected_tab == "About":
    st.header("About This Project")
    st.markdown(""" 
    This project utilizes machine learning algorithms to predict and forecast sales revenue for an American retailer with stores across California, Texas, and Wisconsin. 
    The API built serves as a backend to provide sales predictions and forecasts.
    """)

# --- Footer ---
st.header("Footer")
st.markdown(""" 
For more information, visit the [GitHub Repository](https://github.com/buithehaiuts/at2_ml_as_a_service_api).
Contact: thehai.bui@student.uts.edu.au
""")

# --- Instructions and Documentation ---
st.header("Instructions")
st.markdown(""" 
To use this application, simply enter the required fields for sales prediction and forecasting, 
then click the respective buttons to see the results. 
Make sure the API is up and running for successful predictions.
""")
