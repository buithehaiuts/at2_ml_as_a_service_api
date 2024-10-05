import streamlit as st
import requests
from datetime import datetime
import pandas as pd
import altair as alt
import json
import numpy as np

# Load the id_values.json file
with open('app/backend/id_values.json', 'r') as f:
    id_values = json.load(f)

# Extracting values for dropdowns from JSON
item_ids = id_values.get("item_id", [])
store_ids = id_values.get("store_id", [])
state_ids = id_values.get("state_id", [])
cat_ids = id_values.get("cat_id", [])
dept_ids = id_values.get("dept_id", [])

# Set the title of the application
st.title("Sales Prediction and Forecasting App")
st.write("This application allows you to forecast sales using different forecasting models. "
         "Select a tab below to get started.")

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

# Sales Prediction Tab 
elif selected_tab == "Sales Prediction":
    st.header("Sales Prediction")

    # Create sub-tabs for introduction and prediction in the sidebar
    subtab = st.sidebar.radio("Select Subtab", ["Introduction", "Prediction"])

    if subtab == "Introduction":
        # Description and Instructions
        st.write("""
        Welcome to the Sales Prediction section of the application! This tool allows you to predict sales for a specific item in a store on a chosen date.

        **Overview:**
        - In this section, you can select various parameters including store ID, item ID, state ID, category ID, and department ID to generate a sales prediction.

        **Instructions:**
        
        1. **Select Parameters for Prediction:**
        - Use the dropdowns to select the following:
            - **Store ID:** Choose the store for which you want to predict sales.
            - **Item ID:** Select the item you want to predict sales for.
            - **State ID:** Pick the state where the store is located.
            - **Category ID:** Choose the category of the item.
            - **Department ID:** Select the department the item belongs to.

        2. **Choose a Date for Prediction:**
        - Use the date picker to select the specific date for which you want to predict sales.

        3. **Predict Sales:**
        - Click the **"Predict Sales"** button to initiate the prediction process. The application will use the selected parameters to fetch the sales prediction.

       4. **View the Results:**
        - After generating the forecast, the results will be displayed in a table format, showing the predicted sales and lower and upper confidence intervals (`yhat_lower` and `yhat_upper`). 
        - These intervals provide a range of uncertainty around the predictions:
            - **`yhat`:** This is the predicted value (sales) at a given point in time.
            - **`yhat_lower`:** This indicates the lower bound of the confidence interval for the prediction, representing the minimum expected sales value within a specified level of certainty.
            - **`yhat_upper`:** This represents the upper bound of the confidence interval for the prediction, indicating the maximum expected sales value within the same level of certainty.
            
        5. **Visualize the Forecast:**
        - A chart will be generated to visualize the forecasted sales alongside the confidence intervals, allowing you to easily assess the predicted range of sales values.

        """)

    if subtab == "Prediction":
        # Input fields for sales prediction
        date_input = st.date_input("Select a date for prediction", value=datetime.today())
        date_str = date_input.strftime("%Y-%m-%d")

        # Check if dropdown lists are populated
        if not (store_ids and item_ids and state_ids and cat_ids and dept_ids):
            st.error("Dropdown data is incomplete. Please check your id_values.json file.")
        else:
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
                try:
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
                    prediction_url = f"{base_url}/sales/stores/items/"
                    prediction_response = requests.get(prediction_url, params=params)
                    
                    if prediction_response.status_code == 200:
                        prediction_data = prediction_response.json()

                        # Check if prediction_data is a list and contains a float value
                        if isinstance(prediction_data, list) and len(prediction_data) > 0:
                            raw_value = prediction_data[0]  # Assuming the first element is the prediction

                            # Check if raw_value is a float (or int) before applying expm1
                            if isinstance(raw_value, (float, int)):
                                # Applying expm1 to reverse the log1p transformation
                                sales_value = np.expm1(raw_value)
                                formatted_sales_value = f"{sales_value:,.2f}"  # Format with commas and 2 decimal places

                                # Display the predicted sales value as a metric
                                st.metric(label="Predicted Sales", value=f"${formatted_sales_value}")
                            else:
                                st.error("Prediction value is not numeric.")
                        else:
                            st.error("Unexpected data format received from the API.")
                    else:
                        st.error(f"Error in fetching sales prediction. Status code: {prediction_response.status_code}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {e}")

# --- Sales Forecasting Tab ---
elif selected_tab == "Sales Forecasting":
    st.header("Sales Forecasting")

    # Create sub-tabs for introduction and forecasting in the sidebar
    subtab = st.sidebar.radio("Select Subtab", ["Introduction", "Forecasting"])

    if subtab == "Introduction":
        st.write("""
        Welcome to the Sales Prediction section of the application! Here you can forecast future sales using various predictive models. 

        **Instructions:**

        1. **Select a Forecast Start Date:** 
        - Use the date picker to select the start date for your sales prediction. This date marks the beginning of the forecast period.

        2. **Choose a Forecasting Model Type:** 
        - From the dropdown menu, select the desired forecasting model. Options may include:
        - **Prophet:** A forecasting tool by Facebook that is robust to missing data and shifts in the trend.
        - **Prophet with Events:** Utilizes external events that may influence sales.
        - **Prophet with Holidays:** Considers holidays that could affect sales patterns.
        - **Prophet with Monthly Seasonality:** Takes into account monthly sales trends.

        3. **Generate the Sales Forecast:**
        - Click the **"Forecast Sales"** button to initiate the prediction process. The application will call the relevant forecasting model and return the predicted sales for the next seven days.

        4. **View the Results:**
        - After generating the forecast, the results will be displayed in a table, showing the predicted sales and lower and upper confidence intervals. 
         - These intervals provide a range of uncertainty around the predictions:
            - **`sales`:** This is the predicted value (sales) at a given point in time.
            - **`yhat_lower`:** This indicates the lower bound of the confidence interval for the prediction, representing the minimum expected sales value within a specified level of certainty.
            - **`yhat_upper`:** This represents the upper bound of the confidence interval for the prediction, indicating the maximum expected sales value within the same level of certainty.
        
        5. **Visualize the Forecast:**
        - A chart will be generated to visualize the forecasted sales alongside the confidence intervals, allowing you to assess the predicted range of sales values easily.

        **Important Notes:**
        - Please ensure you have a stable internet connection, as the forecast data is retrieved from an external API.
        - If you encounter any errors or unexpected results, please check the console for any messages and try again.

        Happy forecasting!
        """)

    elif subtab == "Forecasting":
        # Input for forecast start date
        forecast_start_date = st.date_input("Select a forecast start date", value=datetime.today())
        forecast_date_str = forecast_start_date.strftime("%Y-%m-%d")

        # Dropdown for model type selection
        model_type = st.selectbox("Select Forecasting Model Type", 
                                   ['prophet', 'prophet_event', 'prophet_holiday', 'prophet_month'])

        # Button to trigger sales forecasting
        if st.button("Forecast Sales"):
            try:
                forecast_url = f"{base_url}/sales/national/"
                params = {"date": forecast_date_str, "model_type": model_type}
                forecast_response = requests.get(forecast_url, params=params)

                if forecast_response.status_code == 200:
                    forecast_data = forecast_response.json()

                    # Check if the 'forecasts' key is in the response
                    if 'forecasts' in forecast_data:
                        # Convert the forecast data to a DataFrame
                        forecast_df = pd.DataFrame(forecast_data['forecasts'])

                        # Convert 'ds' to datetime
                        forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.date
                        # Rename the 'ds' column to 'date'
                        forecast_df.rename(columns={'ds': 'date', 'yhat': 'sales'}, inplace=True)

                        st.success("Forecasted Sales for the Next 7 Days:")
                        st.table(forecast_df)

                        # Create the chart
                        base = alt.Chart(forecast_df).encode(x='date:T')

                        # Line for sales forecast
                        line = base.mark_line(color='blue').encode(y='sales:Q', tooltip=['date', 'sales'])

                        # Area for confidence interval
                        lower_bound = base.mark_area(opacity=0.2, color='lightblue').encode(
                            y='yhat_lower:Q',
                            y2='yhat_upper:Q'
                        )

                        # Points for actual sales
                        points = base.mark_circle(color='blue').encode(
                            y='sales:Q', 
                            tooltip=['date', 'sales', 'yhat_lower', 'yhat_upper']
                        )

                        # Combine all layers
                        chart = (lower_bound + line + points).properties(
                            title='Sales Forecast for the Next 7 Days with Confidence Intervals',
                            width=700,
                            height=400
                        )

                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.error("No forecast key found in the response.")
                        st.json(forecast_data)  # Log the entire response for debugging
                else:
                    st.error("Failed to retrieve forecast data.")
                    st.json(forecast_response.json())  # Log error response for debugging

            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")

# About Tab
elif selected_tab == "About":
    st.header("About This Project")
    st.markdown(""" 
    This project utilizes machine learning algorithms to predict and forecast sales revenue for an American retailer with stores across California, Texas, and Wisconsin. 
    The API built serves as a backend to provide sales predictions and forecasts.
    """)

# Footer 
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
