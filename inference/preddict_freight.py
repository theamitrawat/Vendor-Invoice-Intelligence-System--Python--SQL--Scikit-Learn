import joblib
import pandas as pd
from pathlib import Path


# Resolve model path relative to this file's location
_BASE_DIR  = Path(__file__).resolve().parents[1]
MODEL_PATH = _BASE_DIR / "models" / "predict_freight_model.pkl"


def load_model(model_path: Path = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    return joblib.load(model_path)


def predict_freight_cost(input_data: dict) -> pd.DataFrame:
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Must contain keys: 'Quantity', 'Dollars'

    Returns
    -------
    pd.DataFrame
        DataFrame with predicted freight cost column added
    """
    required_keys = {"Quantity", "Dollars"}
    missing = required_keys - set(input_data.keys())
    if missing:
        raise ValueError(f"Missing required input keys: {missing}")

    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df["Predicted_Freight"] = model.predict(input_df[["Quantity", "Dollars"]]).round(2)

    return input_df


if __name__ == "__main__":
    # Example inference run (local testing)
    sample_data = {
        "Quantity": [100, 50],
        "Dollars":  [18500, 9000],
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)
