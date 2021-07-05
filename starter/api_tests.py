# -*- coding: utf-8 -*-
"""
Created on Sun May 16 01:43:14 2021

@author: Ahmed

A 3 functions defined to test the API Get/Post functionalities
"""
import json
from starlette.testclient import TestClient
from .main import app

client = TestClient(app)

# A function to test the get
def test_get():
    response = client.get("/welcome")
    assert response.status_code == 200

# A function to test the post on a predicted value of Salaray >50K
def test_post_1():
    input_dict = {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education_num": 13,
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States"
                    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["forecast"] == "Income < 50k"
    
# A function to test the post on a predicted value of Salaray >50K
def test_post_2():
    input_dict = {
                    "age": 31,
                    "workclass": "Private",
                    "fnlgt": 45781,
                    "education": "Masters",
                    "education_num": 14,
                    "marital_status": "Never-married",
                    "occupation": "Prof-specialty",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Female",
                    "capital_gain": 14084,
                    "capital_loss": 0,
                    "hours_per_week": 50,
                    "native_country": "United-States"
                    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["forecast"] == "Income > 50k"
    