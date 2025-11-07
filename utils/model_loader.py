import joblib

def load_model():
    return joblib.load('models/xgboost_model.pkl')