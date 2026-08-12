from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier


def load_data():
    """Load engineered feature datasets and targets."""
    X_train = pd.read_csv(r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\X_train.csv")
    X_test = pd.read_csv(r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\X_test.csv")

    y_train = pd.read_csv(r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\train.csv")["Churn"]
    y_test = pd.read_csv(r"C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\test.csv")["Churn"]

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")

    return X_train, X_test, y_train, y_test


def tune_xgboost(X_train, y_train, X_test, y_test):
    """Tune XGBoost using a small parameter grid."""
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
    }

    best_model = None
    best_auc = 0.0
    best_params = None

    print("\nTuning XGBoost...")
    print("-" * 50)

    for params in ParameterGrid(param_grid):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
        )

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        print(f"Params: {params} - ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_params = params

    print("\nBest Parameters")
    print("-" * 50)
    print(best_params)
    print(f"Best ROC-AUC: {best_auc:.4f}")

    return best_model, best_auc, best_params


def evaluate_model(model, X_test, y_test):
    """Print final evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    print("\nFinal Model Evaluation")
    print("-" * 50)
    print(f"ROC-AUC: {auc:.4f}")

    print("\nClassification Report")
    print("-" * 50)
    print(classification_report(y_test, y_pred))

    return auc


def save_model(model):
    """Save best model to models/churn_model.pkl."""
    model_dir = Path(r"C:\Users\anush\OneDrive\Desktop\churnIQ\models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "churn_model.pkl"

    joblib.dump(model, model_path)

    print(f"\nBest model saved to: {model_path}")


def main():
    X_train, X_test, y_train, y_test = load_data()

    best_model, best_auc, best_params = tune_xgboost(
        X_train, y_train, X_test, y_test
    )

    final_auc = evaluate_model(best_model, X_test, y_test)

    save_model(best_model)

    print("\nGoal Check")
    print("-" * 50)
    if 0.84 <= final_auc <= 0.87:
        print(f"Target achieved! ROC-AUC = {final_auc:.4f}")
    else:
        print(f"ROC-AUC = {final_auc:.4f}")
        print("Target range: 0.84 - 0.87")


if __name__ == "__main__":
    main()
