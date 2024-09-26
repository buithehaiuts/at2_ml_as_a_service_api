from fastapi import FastAPI
from backend.api import app as api_app  # Import your FastAPI app from api.py

# Initialize the main FastAPI app
app = FastAPI()

# Include the routes from api.py
app.mount("/api", api_app)  # This mounts your API under the /api path
