from fastapi import FastAPI
from backend.api import app as api_app  # Import your FastAPI app from api.py

# Initialize the main FastAPI app
app = FastAPI()

# Include the routes from api.py
app.mount("/api", api_app)  # This mounts your API under the /api path

# If you want to add any additional routes or middleware to the main app, do so here.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
