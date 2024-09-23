from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

# instantiate a FasAPI class
app = FastAPI()

# load model

# create a GET endpoint for the real_root() function on root
@app.get("/")
def read_root():
    return {"Hello": "World"}
    
@app.get("/deploy")
async def home_root():
<<<<<<< HEAD
    return {"message":"render deployment"}
=======
    return {"message":"render deployment"}
>>>>>>> a859eadf207ad47733c8c74cb6f8cb962e60c666
