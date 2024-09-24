import streamlit as st
import requests

# Streamlit app layout
st.title("Streamlit Frontend for FastAPI")

# Input fields for mock data
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)

# Prepare input data as a dictionary
input_data = {
    'feature1': feature1,
    'feature2': feature2
}

# Button to trigger prediction
if st.button("Get Prediction"):
    # Call FastAPI for prediction
    with st.spinner("Calling FastAPI..."):
        url = "http://127.0.0.1:8000/predict"  # FastAPI predict endpoint
        response = requests.post(url, json=input_data)
        
        # Show prediction
        if response.status_code == 200:
            prediction = response.json().get('prediction', 'No prediction found.')
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"Error: {response.status_code}")
