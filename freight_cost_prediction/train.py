import joblib
import pandas as pd

from pathlib import Path

try:
    from .data_preprocessing import (
        load_vendor_invoice_data,
        preprocess_data,
        prepare_features,
        split_data
    )
    from .model_evaluation import (
        train_linear_regression,
        train_decision_tree,
        train_random_forest,
        evaluate_model
    )
except ImportError:
    from data_preprocessing import (
        load_vendor_invoice_data,
        preprocess_data,
        prepare_features,
        split_data
    )
    from model_evaluation import (
        train_linear_regression,
        train_decision_tree,
        train_random_forest,
        evaluate_model
    )


def main():

    # =====================================
    # File & Folder Paths
    # =====================================

    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "Data" / "inventory.db"

    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    # =====================================
    # Load Dataset
    # =====================================

    print("\nLoading vendor invoice data...")

    df = load_vendor_invoice_data(db_path)

    print(f"Dataset Shape: {df.shape}")

    # =====================================
    # Preprocess Dataset
    # =====================================

    print("\nPreprocessing invoice data...")

    df = preprocess_data(df)

    print("Data preprocessing completed.")

    # =====================================
    # Prepare Features & Target
    # =====================================

    print("\nPreparing features and target variable...")

    X, y = prepare_features(df)

    print(f"Feature Shape : {X.shape}")
    print(f"Target Shape  : {y.shape}")

    # =====================================
    # Split Dataset
    # =====================================

    print("\nSplitting dataset into training and testing data...")

    x_train, x_test, y_train, y_test = split_data(
        X,
        y
    )

    print(f"Training Rows : {len(x_train)}")
    print(f"Testing Rows  : {len(x_test)}")

    # =====================================
    # Train Machine Learning Models
    # =====================================

    print("\nTraining regression models...")

    lr_model = train_linear_regression(
        x_train,
        y_train
    )

    dt_model = train_decision_tree(
        x_train,
        y_train
    )

    rf_model = train_random_forest(
        x_train,
        y_train
    )

    print("Model training completed.")

    # =====================================
    # Evaluate Models
    # =====================================

    print("\nEvaluating models...")

    results = []

    results.append(
        evaluate_model(
            lr_model,
            x_test,
            y_test,
            "Linear Regression"
        )
    )

    results.append(
        evaluate_model(
            dt_model,
            x_test,
            y_test,
            "Decision Tree Regression"
        )
    )

    results.append(
        evaluate_model(
            rf_model,
            x_test,
            y_test,
            "Random Forest Regression"
        )
    )

    # =====================================
    # Create Results DataFrame
    # =====================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="RMSE"
    )

    print("\nModel Performance Comparison")
    print(results_df)

    # =====================================
    # Select Best Model
    # =====================================

    best_model_info = results_df.iloc[0]

    best_model_name = best_model_info["Model"]

    models = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }

    best_model = models[best_model_name]

    print("\nBest Performing Model")
    print(f"Model Name : {best_model_name}")
    print(f"Best RMSE  : {best_model_info['RMSE']:.2f}")
    print(f"Best R2    : {best_model_info['R2'] * 100:.2f}%")

    # =====================================
    # Save Best Model
    # =====================================

    model_path = model_dir / "predict_freight_model.pkl"

    joblib.dump(
        best_model,
        model_path
    )

    print(f"\nBest model saved successfully.")
    print(f"Model Path : {model_path}")

    # =====================================
    # Save Model Results
    # =====================================

    results_path = model_dir / "model_results.csv"

    results_df.to_csv(
        results_path,
        index=False
    )

    print(f"Model evaluation report saved at: {results_path}")


if __name__ == "__main__":
    main()
