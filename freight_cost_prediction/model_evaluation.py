import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None


def train_linear_regression(x_train, y_train):
    """
    Train Linear Regression model.
    """

    model = LinearRegression()

    model.fit(x_train, y_train)

    return model


def train_decision_tree(
    x_train,
    y_train,
    max_depth=5
):
    """
    Train Decision Tree Regressor model.
    """

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42
    )

    model.fit(x_train, y_train)

    return model


def train_random_forest(
    x_train,
    y_train,
    n_estimators=100,
    max_depth=6
):
    """
    Train Random Forest Regressor model.
    """

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train, y_train)

    return model


def evaluate_model(
    model,
    x_test,
    y_test,
    model_name
):
    """
    Evaluate regression model performance.
    """

    preds = model.predict(x_test)

    mae = mean_absolute_error(y_test, preds)

    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_test, preds)
    else:
        rmse = mean_squared_error(y_test, preds) ** 0.5

    r2 = r2_score(y_test, preds)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def predict_freight_cost(
    model,
    quantity,
    dollars
):
    """
    Predict freight cost for new invoice data.
    """

    invoice_data = pd.DataFrame(
        {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }
    )

    prediction = model.predict(invoice_data)[0]

    return float(prediction)
