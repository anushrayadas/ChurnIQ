from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


def load_datasets():
    """Load engineered train and test feature datasets."""
    train_path = r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\X_train.csv"
    test_path = r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\X_test.csv"

    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    y_train_path = r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\train.csv"
    y_test_path = r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\test.csv"

    y_train = pd.read_csv(y_train_path)["Churn"]
    y_test = pd.read_csv(y_test_path)["Churn"]

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Scale numeric features for Logistic Regression."""
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
        solver="liblinear"
    )

    model.fit(X_train, y_train)

    print("Model training completed.")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nEvaluation Metrics")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report")
    print("-" * 40)
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print("-" * 40)
    print(confusion_matrix(y_test, y_pred))

    return roc_auc


def save_artifacts(model, scaler):
    """Save trained model and scaler."""
    model_dir = Path(r"C:\Users\anush\OneDrive\Desktop\churnIQ\models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "logistic_regression.joblib"
    scaler_path = model_dir / "scaler.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nModel saved to:  {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def main():
    X_train, X_test, y_train, y_test = load_datasets()

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = train_model(X_train_scaled, y_train)

    roc_auc = evaluate_model(model, X_test_scaled, y_test)

    save_artifacts(model, scaler)

    print("\nGoal Check")
    print("-" * 40)
    if 0.78 <= roc_auc <= 0.80:
        print(f"ROC-AUC target achieved: {roc_auc:.4f}")
    else:
        print(f"ROC-AUC obtained: {roc_auc:.4f}")
        print("Target range: 0.78 - 0.80")


if __name__ == "__main__":
    main()
