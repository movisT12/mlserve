from main import app
import pytest
from fastapi.testclient import TestClient
import joblib
client=TestClient(app)
def test_home():
    respone=client.get("/")
    assert respone.status_code==200

def test_model_loads():
    bundle = joblib.load("model/model.pkl")
    assert bundle is not None

def test_scaller():
    scaller=joblib.load("model/scaller.pkl")
    assert scaller is not None

def test_transation_normal():
    transcation=client.post("/predict",json={
        "time": 0.0, "v1": -1.35, "v2": -0.07,
        "v3": 2.53, "v4": 1.37, "v5": -0.33,
        "v6": 0.46, "v7": 0.23, "v8": 0.09,
        "v9": 0.36, "v10": 0.09, "v11": -0.55,
        "v12": -0.61, "v13": -0.99, "v14": -0.31,
        "v15": 1.46, "v16": -0.47, "v17": 0.20,
        "v18": 0.02, "v19": 0.40, "v20": 0.25,
        "v21": -0.01, "v22": 0.27, "v23": -0.11,
        "v24": 0.06, "v25": 0.12, "v26": -0.18,
        "v27": 0.13, "v28": -0.02, "amount": 149.62
    })
    assert transcation.status_code==200
    assert transcation.json()["result"]=="NORMAL"

def test_fraud_result():
    transaction = client.post("/predict", json={
        "time": 406.0, "v1": -2.31, "v2": 1.95,
        "v3": -1.60, "v4": 3.99, "v5": -0.52,
        "v6": -1.42, "v7": -2.53, "v8": 1.39,
        "v9": -2.77, "v10": -2.77, "v11": 3.20,
        "v12": -2.89, "v13": -0.59, "v14": -4.28,
        "v15": 0.38, "v16": -1.14, "v17": -2.83,
        "v18": -0.01, "v19": 0.41, "v20": 0.12,
        "v21": 0.51, "v22": -0.03, "v23": -0.46,
        "v24": 0.32, "v25": 0.04, "v26": 0.17,
        "v27": 0.26, "v28": -0.14, "amount": 122.21
    })
    assert transaction.json()["result"] == "FRAUD"

def test_missing_data():
    response=client.post("/predict",json={
       "time": 0.0,
        "v1": -1.35
    })
    assert response.status_code==422  

def test_wrong_data_type():
    response = client.post("/predict", json={
        "time": "not_a_number",
        "v1": -1.35, "v2":True,
        "v3": 2.53, "v4": 1.37, "v5": -0.33,
        "v6": 0.46, "v7": 0.23, "v8": 0.09,
        "v9": 0.36, "v10": 0.09, "v11": "Name",
        "v12": -0.61, "v13": -0.99, "v14": -0.31,
        "v15": 1.46, "v16": -0.47, "v17": 0.20,
        "v18": 0.02, "v19": False, "v20": 0.25,
        "v21": -0.01, "v22": 0.27, "v23": -0.11,
        "v24": 0.06, "v25": 0.12, "v26": -0.18,
        "v27": 0.13, "v28": -0.02, "amount": 149.62
    })
    assert response.status_code ==422 
def test_probability_range():
    transaction = client.post("/predict", json={
        "time": 0.0, "v1": -1.35, "v2": -0.07,
        "v3": 2.53, "v4": 1.37, "v5": -0.33,
        "v6": 0.46, "v7": 0.23, "v8": 0.09,
        "v9": 0.36, "v10": 0.09, "v11": -0.55,
        "v12": -0.61, "v13": -0.99, "v14": -0.31,
        "v15": 1.46, "v16": -0.47, "v17": 0.20,
        "v18": 0.02, "v19": 0.40, "v20": 0.25,
        "v21": -0.01, "v22": 0.27, "v23": -0.11,
        "v24": 0.06, "v25": 0.12, "v26": -0.18,
        "v27": 0.13, "v28": -0.02, "amount": 149.62
    })
    probability = transaction.json()["probability"]
    assert 0.0 <= probability <= 1.0
