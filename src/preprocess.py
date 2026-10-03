import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(input_path, output_path):
    # Resolve input file if a directory path is passed
    if os.path.isdir(input_path):
        csv_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in directory: {input_path}")
        file_to_read = os.path.join(input_path, csv_files[0])
    else:
        file_to_read = input_path

    print(f"Reading raw data from: {file_to_read}")
    df = pd.read_csv(file_to_read)

    # Drop non-predictive identifiers
    columns_to_drop = [col for col in ["UDI", "Product ID"] if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    # Map categorical Type column if present
    if "Type" in df.columns and df["Type"].dtype == object:
        df["Type"] = df["Type"].map({"L": 1, "M": 2, "H": 3}).fillna(1)

    target_col = "Machine failure" if "Machine failure" in df.columns else df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split dataset for training (80-20 split)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None
    )

    # Split temp dataset into test and validation (50-50 of remaining 20% -> 10% each)
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp if len(y_temp.unique()) > 1 else None
    )

    # Combine features and target for SageMaker formats
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    validation = pd.concat([X_val, y_val], axis=1)

    os.makedirs(output_path, exist_ok=True)

    # Output three distinct CSV files with exact required names
    train_path = os.path.join(output_path, "train.csv")
    test_path = os.path.join(output_path, "test.csv")
    validation_path = os.path.join(output_path, "validation.csv")

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    validation.to_csv(validation_path, index=False)

    print("Preprocessing completed successfully.")
    print(f"Saved: {train_path} (shape: {train.shape})")
    print(f"Saved: {test_path} (shape: {test.shape})")
    print(f"Saved: {validation_path} (shape: {validation.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess predictive maintenance data")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw CSV file or input directory"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save train.csv, test.csv, validation.csv"
    )
    args = parser.parse_args()

    preprocess(args.input, args.output)