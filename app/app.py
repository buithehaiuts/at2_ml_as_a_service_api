from fastapi import FastAPI
from starlette.responses import JSONResponse

# Instantiate a FastAPI class
app = FastAPI()

# Create a GET endpoint for the root
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/deploy")
async def home_root():
    return {"message": "Render deployment"}

# Create a POST endpoint for making mock predictions
@app.post("/predict")
async def predict(data: dict):
    # Mock prediction logic
    mock_prediction = {"prediction": "mock_prediction_result"}
    
    return JSONResponse(content=mock_prediction)

# To run the FastAPI app, you would use:
# uvicorn api:app --reload
