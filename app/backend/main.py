# main.py
from fastapi import FastAPI
from api import SalesAPI
import uvicorn

# Initialize the SalesAPI which includes the FastAPI app
sales_api = SalesAPI()

# If you want to add any additional routes or middleware to the main app, do so here.

if __name__ == "__main__":
    uvicorn.run(sales_api.app, host="0.0.0.0", port=8000)
