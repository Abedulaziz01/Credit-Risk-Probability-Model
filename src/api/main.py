from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictRequest, PredictResponse
from src.model_loader import load_registered_or_local_model
from src.predict import predict

app = FastAPI(title="Credit Risk Probability API")
model = None
model_source = None


@app.on_event("startup")
def startup_event():
    global model, model_source
    try:
        model, model_source = load_registered_or_local_model()
    except Exception as exc:
        raise RuntimeError(f"Could not load model for API startup: {exc}")


@app.get("/")
def root():
    return {"message": "Credit Risk Probability API", "docs": "/docs", "model_source": model_source}


@app.get("/health")
def health():
    return {"status": "ok", "model_source": model_source}


@app.post("/predict", response_model=PredictResponse)
def predict_risk(request: PredictRequest):
    try:
        prediction = predict(model, request.model_dump())
        return PredictResponse(**prediction)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
