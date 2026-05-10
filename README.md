# MLServe — Fraud Detection API

FastAPI microservice serving a scikit-learn classification model 
that detects credit card fraud with confidence scores in under 100ms.

## Tech Stack
- FastAPI
- scikit-learn
- Docker
- pytest

## How to Run

### Install dependencies
pip install -r requirements.txt

### Start the API
uvicorn main:app --reload

### Open docs
http://localhost:8000/docs

## API Contract

### POST /predict

Input:
{
  "time": 0.0,
  "v1": -1.35,
  ... (v1 to v28),
  "amount": 149.62
}

Output:
{
  "prediction": 0,
  "probability": 0.02,
  "result": "NORMAL"
}

## Retrain the Model
python model/train.py

## How to Swap the Model
1. Train new model → saves to model/model.pkl
2. API loads model.pkl at startup
3. No changes needed in main.py
4. Restart the server

## Run Tests
pytest tests/test_api.py -v