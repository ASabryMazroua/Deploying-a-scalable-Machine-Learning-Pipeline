#To start the server use the cmd to type: uvicorn main:app --reload

# Put the code for your API here.
import os
import numpy as np
import pandas as pd
#Import libraries related to fastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
#Import the inference function to be used to predict the values
from starter.starter.ml.model import inference
from starter.stater.ml.data import process_data
#Import the model to be used to predict
model = pd.read_pickle(r"starter/model/model.pkl")

#Initial a FastAPI instance
app = FastAPI()

#Give Heroku the ability to pull in data from DVC upon app start up.
#if "DYNO" in os.environ and os.path.isdir(".dvc"):
#    os.system("dvc config core.no_scm true")
#    if os.system("dvc pull") != 0:
#        exit("dvc pull failed")
#    os.system("rm -r .dvc .apt/usr/lib/dvc")

# pydantic models
class DataIn(BaseModel):
    #The input should be alist of 108 values 
    age : int
    workclass : str
    fnlgt : int
    education : str 
    education_num : int
    marital_status : str
    occupation : str
    relationship : str
    race : str
    sex : str
    capital_gain : int
    capital_loss : int
    hours_per_week : int
    native_country : str

class DataOut(BaseModel):
    #The forecast output will be either >50K or <50K 
    forecast: str = "Income > 50k"

def convert(input_row):
    '''
    This function is used to convert the list input into an array with a shape of (1, 108)
    
    input : input_row 
    type: list
    
    output: a numpy ndarray of shape (1, 108)
    '''
    result = np.array(input_row)
    result = result.reshape(1,108)
    return result

# routes
@app.get("/welcome")
async def welcome():
    return {"Welcome": "to the Model!"}


@app.post("/predict", response_model=DataOut, status_code=200)
def get_prediction(payload: DataIn):
    #Reading the input row
    input_row = payload.input_row
    #Checing that its length is 108
    if len(input_row) != 108:
        raise HTTPException(status_code=400, detail="Data length is not accurate! Please Enter a list of 108 elements")
    # Process the data with the process_data function.
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
    X_processed, y_processed, encoder, lb = process_data(X, categorical_features=cat_features, training=False)
    #Calling the inference function to make a prediction  
    prediction_outcome = inference(model, convert(input_row))[0]
    
    #Interpreting the prediction for the end user
    if prediction_outcome == 0:
        prediction_outcome = "Income < 50k"
    elif prediction_outcome == 1:
        prediction_outcome = "Income > 50k"
    #Building the response dictionary
    response_object = {"forecast": prediction_outcome}
    return response_object
