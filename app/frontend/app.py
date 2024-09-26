import streamlit as st
import requests

# Set the FastAPI URL from environment variable or use a default
FASTAPI_URL = st.secrets.get("FASTAPI_URL", "http://localhost:8000")  # Adjust if necessary

# Function to fetch train data
def fetch_train_data():
    try:
        response = requests.get(f"{FASTAPI_URL}/train")
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching train data: {e}")
        return []

# Function to fetch test data
def fetch_test_data():
    try:
        response = requests.get(f"{FASTAPI_URL}/test")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching test data: {e}")
        return []

# Function to make sales prediction
def predict_sales(date, store_id, item_id):
    try:
        response = requests.post(f"{FASTAPI_URL}/sales/stores/items/", json={"date": date, "store_id": store_id, "item_id": item_id})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error making prediction: {e}")
        return None

# Streamlit application layout
st.title("Sales Revenue Prediction")

# Fetch and display train data
if st.button("Fetch Train Data"):
    train_data = fetch_train_data()
    st.write(train_data)

# Fetch and display test data
if st.button("Fetch Test Data"):
    test_data = fetch_test_data()
    st.write(test_data)

# Sales prediction form
st.header("Predict Sales")

with st.form(key="sales_prediction_form"):
    date = st.text_input("Date (YYYY-MM-DD)")
    store_id = st.number_input("Store ID", min_value=0)
    item_id = st.number_input("Item ID", min_value=0)
    
    submit_button = st.form_submit_button("Predict")

    if submit_button:
        prediction = predict_sales(date, store_id, item_id)
        if prediction:
            st.success(f"Predicted Sales: {prediction['prediction']}")
