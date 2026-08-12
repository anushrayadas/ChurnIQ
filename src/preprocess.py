from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    """Load processed churn dataset."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    print(f"Loaded data from: {path}")

    return df


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TotalCharges to float and handle blanks.
    """
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    missing_before = df["TotalCharges"].isnull().sum()
    print(f"Missing TotalCharges before fill: {missing_before}")

    median_value = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_value)

    missing_after = df["TotalCharges"].isnull().sum()
    print(f"Missing TotalCharges after fill: {missing_after}")

    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using one-hot encoding.
    """
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    categorical_cols = [col for col in categorical_cols if col != "Churn"]

    print(f"Encoding columns: {categorical_cols}")

    df_encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    return df_encoded


def split_data(df: pd.DataFrame):
    """
    Train/test split with stratification on target variable.
    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df["Churn"] = y_train

    test_df = X_test.copy()
    test_df["Churn"] = y_test

    return train_df, test_df


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    """
    Save train and test datasets.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")


def main():
    input_path = r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\churn_processed.csv"

    output_dir = r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed"

    df = load_data(input_path)

    df = clean_total_charges(df)

    df = encode_categorical_columns(df)

    train_df, test_df = split_data(df)

    save_splits(train_df, test_df, output_dir)

    print("\nTrain/Test Shapes")
    print("-" * 30)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")


if __name__ == "__main__":
    main()
