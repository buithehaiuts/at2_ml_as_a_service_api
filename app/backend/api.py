from fastapi import FastAPI

# Instantiate a FastAPI class
app = FastAPI()

# Create a GET endpoint for the root ("/") path
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Create a GET endpoint for deployment testing
@app.get("/deploy")
async def home_root():
    return {"message": "Render deployment"}

# Create a POST endpoint for mock prediction
@app.post("/predict")
async def predict(data: dict):
    # Here you would normally load a model and predict based on 'data'
    mock_prediction = {"prediction": "This is a mock prediction."}
    return mock_prediction
