import os
import sys
import threading
import time
import requests  # Import requests to make API calls
import streamlit as st
import uvicorn  # Import uvicorn to run FastAPI

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (which contains the backend folder)
parent_dir = os.path.join(current_dir, '..')

# Add the parent directory to the system path
sys.path.append(parent_dir)

# # Import the SalesAPI from the backend
# from backend.api import SalesAPI  # Import the SalesAPI class from api.py

# # Function to run FastAPI in a separate thread
# def run_fastapi():
#     sales_api = SalesAPI()  # Create an instance of SalesAPI
#     uvicorn.run(sales_api.app, host="0.0.0.0", port=8000, log_level="info")  # Change to 0.0.0.0 for broader access

# # Function to wait for FastAPI to be ready
# def wait_for_fastapi(host="127.0.0.1", port=8000, timeout=60):
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         try:
#             response = requests.get(f"http://{host}:{port}/docs")  # Check FastAPI docs endpoint
#             if response.status_code == 200:
#                 return True
#         except requests.ConnectionError:
#             time.sleep(1)  # Wait 1 second before retrying
#     return False

# Function to run Streamlit
# def run_streamlit():
#     st.title("Sales Prediction Dashboard")
#     st.write("This is the Streamlit application running alongside FastAPI.")

    # # Example: Adding a simple button to call an API endpoint
    # if st.button('Get Training Data'):
    #     try:
    #         response = requests.get('http://127.0.0.1:8000/data/display/train/')  # Call your FastAPI endpoint
    #         if response.status_code == 200:
    #             train_data = response.json()
    #             st.write("Training Data:")
    #             st.json(train_data)
    #         else:
    #             st.error(f"Failed to fetch training data: {response.status_code}")
    #     except requests.ConnectionError:
    #         st.error("FastAPI server is not running. Please ensure it's started.")

# if __name__ == "__main__":
#     # Start FastAPI in a separate thread
#     fastapi_thread = threading.Thread(target=run_streamlit)
#     fastapi_thread.start()
#     run_streamlit()  # Run Streamlit app once FastAPI is ready


import streamlit as st
import pandas as pd
import json
from backend.api import DataLoader

# Set up Streamlit app title and description
st.title("Data Loader App")
st.write("This app loads data from GitHub and displays it.")

# Define a cache directory
cache_dir = "data_cache"

# Initialize the DataLoader
data_loader = DataLoader(cache_dir)

# Load URLs from data.json
with open('data.json') as f:
    data = json.load(f)

# Extract train URLs
train_urls = data['train']

# Create a list of training URLs for display
train_urls_list = list(train_urls.values())

# Button to load data
if st.button("Load Training Data"):
    # Loading parts from GitHub
    train_df = data_loader.load_parts_from_github(train_urls_list)

    # Display the loaded DataFrame
    st.subheader("Training DataFrame")
    if not train_df.empty:
        data_loader.display_df(train_df)
    else:
        st.write("No training data loaded.")

# Optional: Add functionality to load testing data if needed
if st.button("Load Testing Data"):
    # You can add testing URLs in a similar manner if you have them
    st.write("Testing data loading is not yet implemented.")
# Run the Streamlit app with `streamlit run app.py` in the terminal


