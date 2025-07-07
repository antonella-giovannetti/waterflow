import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from waterflow.app import model
import pandas as pd
import mlflow.xgboost
import pandas as pd

def test_prediction_is_consistent():
    sample = pd.DataFrame([{
        "ph": 7.0,
        "Hardness": 150,
        "Solids": 8000,
        "Chloramines": 7,
        "Sulfate": 250,
        "Conductivity": 420,
        "Organic_carbon": 6,
        "Trihalomethanes": 80,
        "Turbidity": 3.0
    }])
    proba = model.predict_proba(sample)[0][1]
    assert 0 <= proba <= 1
    assert round(proba, 2) == round(proba, 2)  

def test_model_predicts_potable_for_valid_data():
    sample = pd.DataFrame([{
        "ph": 7.2,
        "Hardness": 130.0,
        "Solids": 80000.0,
        "Chloramines": 7.0,
        "Sulfate": 250.0,
        "Conductivity": 400.0,
        "Organic_carbon": 5.0,
        "Trihalomethanes": 40.0,
        "Turbidity": 2.0
    }])
    prediction = model.predict(sample)[0]
    print("Prediction:", prediction)
    assert prediction == 1 

def test_model_predicts_non_potable_for_bad_data():
    sample = pd.DataFrame([{
        "ph": 4.9,
        "Hardness": 90,
        "Solids": 12000,
        "Chloramines": 2.5,
        "Sulfate": 120,
        "Conductivity": 230,
        "Organic_carbon": 10,
        "Trihalomethanes": 15,
        "Turbidity": 6.5
    }])
    prediction = model.predict(sample)[0]
    print("Prediction:", prediction)
    assert prediction == 0  