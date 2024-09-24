from fastapi import FastAPI

# Instantiate a FastAPI class
app = FastAPI()

# Create a GET endpoint for the root ("/") path
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Create a GET endpoint for the "/deploy" path
@app.get("/deploy")
async def home_root():
    return {"message": "Render deployment"}

# To run the FastAPI app, use the following command:
# uvicorn your_file_name:app --reload
