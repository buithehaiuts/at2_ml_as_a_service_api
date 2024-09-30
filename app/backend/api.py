from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Pydantic model for a Sale
class Sale(BaseModel):
    id: int
    amount: float
    date: str

# Pydantic model for the response containing multiple sales
class SalesResponse(BaseModel):
    sales: List[Sale]

class SalesAPI:
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/health/")
        def health_check():
            return {"status": "healthy"}

        @self.app.get("/sales/", response_model=SalesResponse)
        def read_sales():
            # Your logic to read sales data goes here
            # Replace the following with actual data retrieval logic
            sample_sales = [
                {"id": 1, "amount": 150.0, "date": "2024-01-01"},
                {"id": 2, "amount": 250.0, "date": "2024-01-02"},
            ]
            return {"sales": sample_sales}  # Example response with sample data

        # You can add more endpoints as needed, e.g., for sales prediction

# This allows the FastAPI app to be imported and run directly
api = SalesAPI()

# Run the application if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api.app, host="0.0.0.0", port=8000)
