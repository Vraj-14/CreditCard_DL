from .preprocessor import prepare_input
from .model_loader import load_model

# Load model once
model = load_model()

def predict_fraud(amount, time_sec, v_features):
    X = prepare_input(amount, time_sec, v_features)
    proba = model.predict_proba(X)[0][1]
    label = "Fraud" if proba > 0.5 else "Legitimate"
    return {
        "probability": round(proba, 4),
        "label": label,
        "confidence": round(proba * 100, 2)
    }