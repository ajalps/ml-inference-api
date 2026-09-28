"""Loads the trained champion pipeline once and exposes a predict() helper.

The pipeline saved to model/champion_model.pkl bundles preprocessing
(imputation, scaling, encoding) and the classifier together, so a raw
dict of applicant features can be passed straight in.
"""
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "champion_model.pkl"

_model = None  # loaded lazily, once, on first request


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(payload: dict) -> Tuple[int, float]:
    """Run inference on a single applicant record.

    Args:
        payload: dict of raw feature values (as produced by
            PredictionRequest.model_dump()).

    Returns:
        (predicted_class, probability_of_default)
    """
    model = get_model()
    df = pd.DataFrame([payload])
    probability = float(model.predict_proba(df)[:, 1][0])
    predicted_class = int(model.predict(df)[0])
    return predicted_class, probability
