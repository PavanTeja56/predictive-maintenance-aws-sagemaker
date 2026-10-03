import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model(input_dir=None, model_dir=None):
    # Default paths for SageMaker Training Job
    # SageMaker mounts training channels under /opt/ml/input/data/
    if input_dir is None:
        input_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

    if not os.path.exists(input_dir):
        # Fallback to root /opt/ml/input/data/ if specific channel directory is not present
        if os.path.exists("/opt/ml/input/data/"):
            input_dir = "/opt/ml/input/data/"
        elif os.path.exists("data/processed"):
            input_dir = "data/processed"

    # Default output model directory for SageMaker
    if model_dir is None:
        model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model/")

    # Find training CSV in input directory
    if os.path.isdir(input_dir):
        candidate = os.path.join(input_dir, "train.csv")
        if os.path.exists(candidate):
            train_path = candidate
        else:
            csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError(f"No CSV file found in training directory: {input_dir}")
            train_path = os.path.join(input_dir, csv_files[0])
    else:
        train_path = input_dir

    print(f"Loading training data from: {train_path}")
    train = pd.read_csv(train_path)

    target_col = "Machine failure" if "Machine failure" in train.columns else train.columns[-1]
    X = train.drop(columns=[target_col])
    y = train[target_col]

    print(f"Training RandomForestClassifier on {X.shape[0]} samples with {X.shape[1]} features...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X, y)
    print("Model Training Completed successfully!")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model artifact saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train predictive maintenance model")
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        help="Path to training data directory"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model/"),
        help="Path to save trained model artifact"
    )
    args, _ = parser.parse_known_args()

    train_model(args.train, args.model_dir)