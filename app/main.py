"""FastAPI microservice serving the champion credit-default-risk model."""
from fastapi import FastAPI, HTTPException

from app.model import get_model, predict
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

app = FastAPI(
    title="Credit Default Risk Inference API",
    description=(
        "Serves the champion credit-default-risk model trained earlier in this "
        "project (see the model-tuning notebook) behind a REST API."
    ),
    version="1.0.0",
)


def _risk_label(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Medium"
    return "High"


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Basic liveness/readiness check — confirms the model file loaded."""
    try:
        get_model()
        return HealthResponse(status="ok", model_loaded=True)
    except Exception:
        return HealthResponse(status="model_unavailable", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_endpoint(payload: PredictionRequest) -> PredictionResponse:
    """Predict default risk for a single applicant.

    FastAPI + Pydantic validate the request body against `PredictionRequest`
    before this function body ever runs — missing required fields, wrong
    types, or out-of-range values are rejected with HTTP 422 automatically.
    """
    try:
        predicted_class, probability = predict(payload.model_dump())
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        prediction=predicted_class,
        probability_default=round(probability, 4),
        risk_label=_risk_label(probability),
    )
