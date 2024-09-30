# backend/api.py

from fastapi import FastAPI

class SalesAPI:
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/health/")
        def health_check():
            return {"status": "healthy"}

        @self.app.get("/sales/")
        def read_sales():
            # Your logic to read sales data goes here
            return {"sales": []}  # Example response

        # Add other endpoints as needed

# This allows the FastAPI app to be imported and run directly
api = SalesAPI()
