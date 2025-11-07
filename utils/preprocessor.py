import pandas as pd
import joblib

# Load scaler
scaler = joblib.load('utils/scaler.pkl')

# === EXACT TRAINING ORDER (MUST MATCH) ===
TRAINING_COLUMNS = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Time_scaled', 'Amount_scaled'
]

def prepare_input(amount, time_sec, v_features):
    """
    Returns DataFrame with CORRECT column order for XGBoost.
    """
    # Scale inputs
    amount_scaled = scaler.transform([[amount]])[0][0]
    time_scaled = scaler.transform([[time_sec]])[0][0]

    # Build full feature dict
    full_features = {
        **v_features,  # V1 to V28
        'Time_scaled': time_scaled,
        'Amount_scaled': amount_scaled
    }

    # Create DataFrame and REORDER to match training
    df = pd.DataFrame([full_features])
    df = df[TRAINING_COLUMNS]  # Enforce exact order

    return df