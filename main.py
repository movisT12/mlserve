from fastapi import FastAPI
import joblib
import numpy as np
from schema import Transaction
import pandas as pd


model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaller.pkl")


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([[
        transaction.time,
        transaction.v1,
        transaction.v2,
        transaction.v3,
        transaction.v4,
        transaction.v5,
        transaction.v6,
        transaction.v7,
        transaction.v8,
        transaction.v9,
        transaction.v10,
        transaction.v11,
        transaction.v12,
        transaction.v13,
        transaction.v14,
        transaction.v15,
        transaction.v16,
        transaction.v17,
        transaction.v18,
        transaction.v19,
        transaction.v20,
        transaction.v21,
        transaction.v22,
        transaction.v23,
        transaction.v24,
        transaction.v25,
        transaction.v26,
        transaction.v27,
        transaction.v28,
        transaction.amount
    ]])

    data_scaled = scaler.transform(data)


    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    return (
    {
    "prediction": int(prediction),
    "probability": float(probability),
    "result": "FRAUD" if prediction == 1 else "NORMAL"
    }
    )