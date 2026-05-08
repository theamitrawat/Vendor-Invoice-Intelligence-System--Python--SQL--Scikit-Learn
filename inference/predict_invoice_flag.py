import joblib
import pandas as pd


MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """

    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


def load_scaler(scaler_path: str = SCALER_PATH):
    """
    Load the feature scaler.
    """

    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)

    return scaler


def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Input invoice data

    Returns
    -------
    pd.DataFrame
        DataFrame with predicted flag
    """

    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)
    
    # Scale the features
    features = ["invoice_quantity", "invoice_dollars", "Freight", 
                "total_item_quantity", "total_item_dollars"]
    input_df_scaled = scaler.transform(input_df[features])

    input_df["Predicted_Flag"] = (
        model.predict(input_df_scaled).round()
    )

    return input_df


if __name__ == "__main__":

    # Example inference run (local testing)
    sample_data = {
        "invoice_quantity": [100, 50],
        "invoice_dollars": [18500, 9000],
        "Freight": [250, 120],
        "total_item_quantity": [95, 48],
        "total_item_dollars": [18000, 8700]
    }

    prediction = predict_invoice_flag(sample_data)

    print(prediction)