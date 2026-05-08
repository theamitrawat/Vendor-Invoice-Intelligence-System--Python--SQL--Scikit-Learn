from pathlib import Path
import joblib

try:
    from .data_preprocessing import (
        load_invoice_data,
        apply_labels,
        split_data,
        scale_features
    )
    from .modeling_evaluation import (
        train_random_forest,
        evaluate_classifier
    )
except ImportError:
    from data_preprocessing import (
        load_invoice_data,
        apply_labels,
        split_data,
        scale_features
    )
    from modeling_evaluation import (
        train_random_forest,
        evaluate_classifier
    )


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    # =====================================
    # File & Folder Paths
    # =====================================
    
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    
    scaler_path = model_dir / "scaler.pkl"
    model_path = model_dir / "predict_flag_invoice.pkl"

    try:
        # Load data
        print("Loading invoice data...")
        df = load_invoice_data()
        print(f"Data loaded. Shape: {df.shape}")
        
        df = apply_labels(df)
        print(f"Labels applied. Data shape: {df.shape}")

        # Prepare data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = split_data(
            df,
            FEATURES,
            TARGET
        )
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        print("Scaling features...")
        X_train_scaled, X_test_scaled = scale_features(
            X_train,
            X_test,
            str(scaler_path)
        )

        # Train and evaluate model
        print("Training Random Forest model...")
        model = train_random_forest(
            X_train_scaled,
            y_train
        )

        evaluate_classifier(
            model,
            X_test_scaled,
            y_test,
            "Random Forest Classifier"
        )

        # Save best model
        print("Saving model...")
        joblib.dump(
            model,
            str(model_path)
        )
        
        print(f"\nBest model saved successfully.")
        print(f"Model Path : {model_path}")
        print(f"Scaler Path: {scaler_path}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()