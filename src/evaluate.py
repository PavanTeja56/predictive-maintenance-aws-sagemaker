import os
import argparse
import json
import tarfile
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score


def load_model(model_input_path):
    """
    Load model artifact whether it is a .tar.gz archive, a joblib file,
    or a directory containing the model artifact.
    """
    if os.path.isfile(model_input_path):
        if tarfile.is_tarfile(model_input_path):
            extracted_dir = "/opt/ml/processing/model_extracted"
            os.makedirs(extracted_dir, exist_ok=True)
            with tarfile.open(model_input_path, "r:gz") as tar:
                tar.extractall(extracted_dir)
            target_file = os.path.join(extracted_dir, "model.joblib")
            if os.path.exists(target_file):
                return joblib.load(target_file)
            for f in os.listdir(extracted_dir):
                if f.endswith(".joblib") or f.endswith(".pkl"):
                    return joblib.load(os.path.join(extracted_dir, f))
            raise FileNotFoundError(f"No joblib model found in tar archive at {model_input_path}")
        else:
            return joblib.load(model_input_path)
    elif os.path.isdir(model_input_path):
        tar_path = os.path.join(model_input_path, "model.tar.gz")
        if os.path.exists(tar_path):
            return load_model(tar_path)
        joblib_path = os.path.join(model_input_path, "model.joblib")
        if os.path.exists(joblib_path):
            return joblib.load(joblib_path)
        for f in os.listdir(model_input_path):
            if f.endswith(".joblib") or f.endswith(".pkl"):
                return joblib.load(os.path.join(model_input_path, f))
        raise FileNotFoundError(f"No model artifact found in directory: {model_input_path}")
    else:
        raise FileNotFoundError(f"Model path does not exist: {model_input_path}")


def evaluate_model(model_path, test_path, output_path):
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    # Resolve test data path if directory
    if os.path.isdir(test_path):
        candidate = os.path.join(test_path, "test.csv")
        if os.path.exists(candidate):
            test_file = candidate
        else:
            csv_files = [f for f in os.listdir(test_path) if f.endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError(f"No test CSV file found in: {test_path}")
            test_file = os.path.join(test_path, csv_files[0])
    else:
        test_file = test_path

    print(f"Loading test data from: {test_file}")
    test = pd.read_csv(test_file)

    target_col = "Machine failure" if "Machine failure" in test.columns else test.columns[-1]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    predictions = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions, zero_division=0))

    print(f"Evaluation Results -> Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")

    evaluation = {
        "classification_metrics": {
            "accuracy": {
                "value": accuracy
            },
            "f1_score": {
                "value": f1
            }
        }
    }

    # Ensure output directory exists and save evaluation.json
    # Required path: /opt/ml/processing/evaluation/evaluation.json
    os.makedirs(output_path, exist_ok=True)
    evaluation_file = os.path.join(output_path, "evaluation.json")

    with open(evaluation_file, "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"Evaluation report successfully saved to: {evaluation_file}")
    return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="/opt/ml/processing/model/model.tar.gz",
        help="Path to model artifact or archive"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="/opt/ml/processing/test/test.csv",
        help="Path to test CSV dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/opt/ml/processing/evaluation/",
        help="Directory to save evaluation.json (/opt/ml/processing/evaluation/)"
    )
    args = parser.parse_args()

    evaluate_model(args.model, args.test, args.output)