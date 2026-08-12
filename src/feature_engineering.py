from pathlib import Path

import pandas as pd


def load_data(train_path: str, test_path: str):
    """Load train and test datasets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train loaded: {train_df.shape}")
    print(f"Test loaded:  {test_df.shape}")

    return train_df, test_df


def add_tenure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure bucket feature.
    """
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12m", "13-24m", "25-48m", "49-72m"]

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count number of active services.
    """
    service_cols = [
        "PhoneService_Yes",
        "MultipleLines_Yes",
        "OnlineSecurity_Yes",
        "OnlineBackup_Yes",
        "DeviceProtection_Yes",
        "TechSupport_Yes",
        "StreamingTV_Yes",
        "StreamingMovies_Yes",
    ]

    existing_cols = [col for col in service_cols if col in df.columns]

    df["service_count"] = df[existing_cols].sum(axis=1)

    return df


def add_contract_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction feature between tenure and contract type.
    """
    if "Contract_One year" in df.columns:
        df["tenure_x_one_year"] = (
            df["tenure"] * df["Contract_One year"]
        )

    if "Contract_Two year" in df.columns:
        df["tenure_x_two_year"] = (
            df["tenure"] * df["Contract_Two year"]
        )

    return df


def encode_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode newly created categorical features.
    """
    df = pd.get_dummies(df, columns=["tenure_bucket"], drop_first=True)

    return df


def save_features(X_train: pd.DataFrame, X_test: pd.DataFrame, output_dir: str):
    """Save engineered feature datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_out = output_path / "X_train.csv"
    test_out = output_path / "X_test.csv"

    X_train.to_csv(train_out, index=False)
    X_test.to_csv(test_out, index=False)

    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")


def main():
    train_path = r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\train.csv"
    test_path = r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\test.csv"
    output_dir = r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed"

    train_df, test_df = load_data(train_path, test_path)

    train_df = add_tenure_buckets(train_df)
    test_df = add_tenure_buckets(test_df)

    train_df = add_service_count(train_df)
    test_df = add_service_count(test_df)

    train_df = add_contract_interaction(train_df)
    test_df = add_contract_interaction(test_df)

    train_df = encode_new_features(train_df)
    test_df = encode_new_features(test_df)

    X_train = train_df.drop(columns=["Churn"])
    X_test = test_df.drop(columns=["Churn"])

    save_features(X_train, X_test, output_dir)

    print("\nFeature Engineering Complete")
    print("-" * 35)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    print("\nNew Features Added:")
    print("- tenure_bucket_*")
    print("- service_count")
    print("- tenure_x_one_year")
    print("- tenure_x_two_year")


if __name__ == "__main__":
    main()
