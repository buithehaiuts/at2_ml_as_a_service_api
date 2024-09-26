# backend/api.py

from fastapi import APIRouter, HTTPException
import requests

router = APIRouter()

# Base URL for the GitHub repository (replace 'your_username' and 'your_repo' with actual values)
GITHUB_DATA_URL = "https://raw.githubusercontent.com/buithehaiuts/at2_ml_as_a_service_experiments/main/data/processed/"

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/data/{file_name}")
async def fetch_data(file_name: str):
    # Construct the full URL for the dataset
    url = f"{GITHUB_DATA_URL}{file_name}"

    # Fetch the data from GitHub
    response = requests.get(url)

    # Check if the response is successful
    if response.status_code == 200:
        # Assuming the data is in pickle format
        try:
            # For pickle files, we might need to read it into a BytesIO buffer
            import pandas as pd
            from io import BytesIO

            data = pd.read_pickle(BytesIO(response.content))
            return data.to_dict()  # Convert DataFrame to dictionary format
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=404, detail="Data file not found")
