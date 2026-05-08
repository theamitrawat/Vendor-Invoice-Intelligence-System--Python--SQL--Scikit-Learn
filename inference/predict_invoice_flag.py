import joblib
import pandas as pd
from pathlib import Path


# Resolve paths relative to this file's location
_BASE_DIR   = Path(__file__).resolve().parents[1]
MODEL_PATH  = _BASE_DIR / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = _BASE_DIR / "models" / "scaler.pkl"


def load_model(model_path: Path = MODEL_PATH):
    """
    Load trained classifier model.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return joblib.load(model_path)


def load_scaler(scaler_path: Path = SCALER_PATH):
    """
    Load the feature scaler.
    """
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
    return joblib.load(scaler_path)


def predict_invoice_flag(input_data: dict) -> pd.DataFrame:
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Must contain keys: 'invoice_quantity', 'invoice_dollars', 'Freight',
                           'total_item_quantity', 'total_item_dollars'

    Returns
    -------
    pd.DataFrame
        DataFrame with predicted flag column added
    """
    required_keys = {
        "invoice_quantity", "invoice_dollars", "Freight",
        "total_item_quantity", "total_item_dollars"
    }
    missing = required_keys - set(input_data.keys())
    if missing:
        raise ValueError(f"Missing required input keys: {missing}")

    model  = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)
    
    # Scale the features
    features = ["invoice_quantity", "invoice_dollars", "Freight", 
                "total_item_quantity", "total_item_dollars"]
    input_df_scaled = scaler.transform(input_df[features])

    input_df["Predicted_Flag"] = model.predict(input_df_scaled).astype(int)

    return input_df


if __name__ == "__main__":
    # Example inference run (local testing)
    sample_data = {
        "invoice_quantity":    [100, 50],
        "invoice_dollars":     [18500, 9000],
        "Freight":             [250, 120],
        "total_item_quantity": [95, 48],
        "total_item_dollars":  [18000, 8700],
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)
