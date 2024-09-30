from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Pydantic model for a Sale
class Sale(BaseModel):
    id: int
    amount: float
    date: str

# Pydantic model for the response containing multiple sales
class SalesResponse(BaseModel):
    sales: List[Sale]

# Pydantic model for the request payload for sales prediction
class SalesPredictionRequest(BaseModel):
    date: str
    store_id: int
    item_id: int

# Pydantic model for the response of predictions
class SalesPredictionResponse(BaseModel):
    predicted_amount: float
    date: str
    store_id: int
    item_id: int

class SalesAPI:
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/health/")
        def health_check():
            return {"status": "healthy"}

        @self.app.get("/sales/national/", response_model=SalesResponse)
        def read_national_sales(date: str):
            # Replace this with actual logic to get national sales data
            sample_sales = [
                {"id": 1, "amount": 150.0, "date": date},
                {"id": 2, "amount": 250.0, "date": date},
            ]
            return {"sales": sample_sales}

        @self.app.post("/sales/stores/items/", response_model=SalesPredictionResponse)
        def predict_sales(request: SalesPredictionRequest):
            # Replace this with actual logic for predicting sales
            # For demonstration, we return a dummy prediction
            predicted_amount = 100.0  # Dummy predicted amount
            return SalesPredictionResponse(
                predicted_amount=predicted_amount,
                date=request.date,
                store_id=request.store_id,
                item_id=request.item_id
            )

# This allows the FastAPI app to be imported and run directly
api = SalesAPI()

# Run the application if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api.app, host="0.0.0.0", port=8000)
