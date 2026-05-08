import os

from fastapi import FastAPI, HTTPException
from joblib import load

from src.api.pydantic_models import PredictRequest, PredictResponse
from src.predict import predict

MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.joblib')

app = FastAPI(title='Credit Risk Probability API')


@app.on_event('startup')
def startup_event():
    global model
    try:
        model = load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f'Could not load model from {MODEL_PATH}: {exc}')


@app.get('/')
def root():
    return {'message': 'Credit Risk Probability API', 'docs': '/docs'}


@app.post('/predict', response_model=PredictResponse)
def predict_risk(request: PredictRequest):
    try:
        probability, label = predict(model, request.dict())
        return PredictResponse(risk_probability=float(probability), risk_label=label)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
