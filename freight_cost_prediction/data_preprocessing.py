import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path: str) -> pd.DataFrame:
    """
    Load vendor invoice data from SQLite database.
    """
    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        VendorNumber,
        Quantity,
        Dollars,
        Freight
    FROM vendor_invoice
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess vendor invoice data.
    """
    df = df.dropna().copy()
    df = df.drop_duplicates()

    numeric_columns = ["Quantity", "Dollars", "Freight"]
    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )
    df = df.dropna(subset=numeric_columns)
    df = df[df["Dollars"] != 0]

    df["freight_per_dollar"] = df["Freight"] / df["Dollars"]
    return df


def prepare_features(df: pd.DataFrame):
    """
    Select features and target variable.
    """

    X = df[[
        "Quantity",
        "Dollars"
    ]]

    y = df["Freight"]

    return X, y


def split_data(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split dataset into training and testing sets.
    """

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return x_train, x_test, y_train, y_test
