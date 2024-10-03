import streamlit as st
import requests
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

# Set the title of the application
st.title("Sales Prediction and Forecasting App")

# Sidebar for navigation
selected_tab = st.sidebar.radio("Select a Tab", 
                                ["API Health Check", 
                                 "Sales Prediction", 
                                 "Sales Forecasting", 
                                 "About"])

# Load item IDs from the JSON file
with open('app/frontend/list_item.json') as f:
    item_ids = json.load(f)

# API Base URL
base_url = "https://at2-ml-as-a-service-api.onrender.com"

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

# # --- Sales Prediction Tab ---
# elif selected_tab == "Sales Prediction":
#     st.header("Sales Prediction")

#     # Input fields for sales prediction
#     date_input = st.date_input("Select a date for prediction", value=datetime.today())

#     # Dropdown for store ID selection
#     store_id = st.selectbox("Select Store ID", 
#                              ['CA_1', 'CA_2', 'CA_3', 'CA_4', 
#                               'TX_1', 'TX_2', 'TX_3', 
#                               'WI_1', 'WI_2', 'WI_3'])

#     # Dropdown for item ID selection using loaded item IDs
#     item_id = st.selectbox("Select Item ID", item_ids)

#     # Model type:
#     model_list = ['prophet', 'prophet_event', 'prophet_holiday', 'prophet_month']
#     model_type = st.selectbox("Select Model Type", model_list)

#     # Button to trigger sales prediction
#     if st.button("Predict Sales"):
#         # Send a GET request to the sales prediction API endpoint
#         try:
#             # Correctly format the GET request with query parameters
#             prediction_url = f"{base_url}/sales/stores/items/?ds={date_input}&item_id={item_id}&store_id={store_id}&model_type={model_type}"
            
#             # Send the GET request
#             prediction_response = requests.get(prediction_url)
            
#             # Handle response
#             if prediction_response.status_code == 200:
#                 prediction_data = prediction_response.json()
#                 st.write("API Response:", prediction_data)  # Debug line
                
#                 # Check if prediction data is valid
#                 if 'prediction' in prediction_data and prediction_data['prediction']:
#                     # Access the predicted value correctly
#                     predicted_value = prediction_data['prediction'][0]['yhat']  # Accessing 'yhat'
#                     st.success(f"Predicted Sales: ${predicted_value:.2f}")
#                 else:
#                     st.error("No prediction data available.")
#                     if 'error' in prediction_data:
#                         st.error(f"API Error: {prediction_data['error']}")
#             else:
#                 st.error("Failed to retrieve prediction data.")
#                 st.json(prediction_response.json())
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error: {e}")

# # --- Sales Forecasting Tab ---
# elif selected_tab == "Sales Forecasting":
#     st.header("Sales Forecasting")
    
#     # Input field for sales forecasting
#     forecast_start_date = st.date_input("Select a forecast start date", value=datetime.today())

#     # Button to trigger sales forecasting
#     if st.button("Forecast Sales"):
#         # Send a request to the sales forecasting API endpoint
#         try:
#             # Corrected URL for the forecast
#             forecast_url = f"{base_url}/sales/national/"
#             params = {"start_date": forecast_start_date.strftime("%Y-%m-%d")}  # Updated key
#             forecast_response = requests.get(forecast_url, params=params)
            
#             # Handle response
#             if forecast_response.status_code == 200:
#                 forecast_data = forecast_response.json()
#                 st.success("Forecasted Sales for the Next 7 Days:")
#                 st.json(forecast_data)  # Display the JSON response
#             else:
#                 st.error("Failed to retrieve forecast data.")
#                 st.json(forecast_response.json())
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error: {e}")


# --- Sales Prediction Tab ---
elif selected_tab == "Sales Prediction":
    st.header("Sales Prediction")

    # Input fields for sales prediction
    date_input = st.date_input("Select a date for prediction", value=datetime.today())

    # Dropdown for store ID selection
    store_id = st.selectbox("Select Store ID", 
                             ['CA_1', 'CA_2', 'CA_3', 'CA_4', 
                              'TX_1', 'TX_2', 'TX_3', 
                              'WI_1', 'WI_2', 'WI_3'])

    # Dropdown for item ID selection using loaded item IDs
    item_id = st.selectbox("Select Item ID", item_ids)

    # Model type:
    model_list = ['prophet', 'prophet_event', 'prophet_holiday', 'prophet_month']
    model_type = st.selectbox("Select Model Type", model_list)

    # Button to trigger sales prediction
    if st.button("Predict Sales"):
        # Send a GET request to the sales prediction API endpoint
        try:
            prediction_url = f"{base_url}/sales/stores/items/?ds={date_input}&item_id={item_id}&store_id={store_id}&model_type={model_type}"
            
            # Send the GET request
            prediction_response = requests.get(prediction_url)
            
            # Handle response
            if prediction_response.status_code == 200:
                prediction_data = prediction_response.json()

                # Check if prediction data is valid
                if 'prediction' in prediction_data and prediction_data['prediction']:
                    # Access the predicted value
                    predicted_value = prediction_data['prediction'][0]['yhat']
                    st.success(f"Predicted Sales: ${predicted_value:.2f}")

                    # Create a DataFrame for display
                    prediction_df = pd.DataFrame(prediction_data['prediction'])
                    prediction_df['ds'] = pd.to_datetime(prediction_df['ds']).dt.date  # Convert to date

                    # Display the prediction in a table
                    st.table(prediction_df[['ds', 'yhat']])

                else:
                    st.error("No prediction data available.")
                    if 'error' in prediction_data:
                        st.error(f"API Error: {prediction_data['error']}")
            else:
                st.error("Failed to retrieve prediction data.")
                st.json(prediction_response.json())
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

# --- Sales Forecasting Tab ---
elif selected_tab == "Sales Forecasting":
    forecast_start_date = st.date_input("Select a forecast start date", value=datetime.today())

    # Button to trigger sales forecasting
    if st.button("Forecast Sales"):
        # Send a request to the sales forecasting API endpoint
        try:
            forecast_url = f"{base_url}/sales/national/"
            params = {"start_date": forecast_start_date.strftime("%Y-%m-%d")}
            forecast_response = requests.get(forecast_url, params=params)

            # Handle response
            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()

                # Check if forecast data is valid
                if 'forecasts' in forecast_data and 'prophet' in forecast_data['forecasts']:
                    forecast_df = pd.DataFrame(forecast_data['forecasts']['prophet'])
                    forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.date

                    # Rename columns for better understanding
                    forecast_df.rename(columns={
                        'yhat': 'Predicted Sales',
                        'yhat_lower': 'Lower Estimate',
                        'yhat_upper': 'Upper Estimate'
                    }, inplace=True)

                    # Display a success message
                    st.success("Forecasted Sales for the Next 7 Days:")

                    # Display table
                    st.table(forecast_df[['ds', 'Predicted Sales', 'Lower Estimate', 'Upper Estimate']])

                    # Create the base chart

                    base = alt.Chart(forecast_df).encode(x='ds:T')

                    # Line for predicted sales
                    line = base.mark_line(color='blue').encode(y='Predicted Sales:Q')

                    # Points for predicted sales
                    points = base.mark_circle(color='blue').encode(y='Predicted Sales:Q')

                    # Confidence interval
                    confidence_interval = base.mark_area(opacity=0.5, color='lightblue').encode(
                        y='Lower Estimate:Q',
                        y2='Upper Estimate:Q'
                    )

                    # Combine all layers
                    chart = (confidence_interval + line + points).properties(
                        title='Sales Forecast for the Next 7 Days',
                        width=700,
                        height=400
                    ).configure_axis(
                        labelAngle=45
                    ).configure_view(
                        strokeWidth=0  # Remove border
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
