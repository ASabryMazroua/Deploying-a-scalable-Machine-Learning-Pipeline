# -*- coding: utf-8 -*-
"""
Created on Sat May 15 02:44:14 2021

@author: Ahmed
"""
import numpy as np
import pandas as pd
import os 
import sklearn
from sklearn.model_selection import train_test_split
from .starter.ml.data import process_data
from .starter.ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv(os.path.join(os.getcwd(),r"starter/data/census_clean.csv"))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Process the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


def test_train_model():
    #Test that this function will return a trained RandomForestClassifier
    model = train_model(X_train, y_train)
    assert type(model) == sklearn.ensemble._forest.RandomForestClassifier
    
def test_compute_model_metrics():
    #Test this will return the three metrics and each ranges from 0 to 1
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    metrics = compute_model_metrics(y_train, preds)
    assert len(metrics) == 3
    assert type(metrics) == tuple
    for metric in metrics:
        assert metric >=0 and metric <= 1


def test_inference():
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert len(preds) == len(X_train) #Assert that the length is the same as x_train
    assert np.all((preds==0)|(preds == 1)) == True #Assert all values are 0 or 1


if __name__ == "__main__":
    test_train_model()
    test_compute_model_metrics()
    test_inference()