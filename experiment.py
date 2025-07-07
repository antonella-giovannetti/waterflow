import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from mlflow.models.signature import infer_signature
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

# Detecte les outliers 
def detect_outliers_iqr(df, exclude=["Potability"]) -> dict:
    outlier_indices = {}
    for col in df.columns:
        if col in exclude:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices[col] = outliers.tolist()
    return outlier_indices

# Prétraitement des données
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    outliers = detect_outliers_iqr(df)
    all_outliers = set(sum(outliers.values(), []))
    df = df.drop(index=all_outliers).reset_index(drop=True)
    df["ph"] = df.groupby("Potability")["ph"].transform(lambda x: x.fillna(x.mean()))
    df["Trihalomethanes"] = df.groupby("Potability")["Trihalomethanes"].transform(lambda x: x.fillna(x.mean()))
    df["Sulfate"] = df.groupby("Potability")["Sulfate"].transform(lambda x: x.fillna(x.median()))
    return df


def balance_dataset(X, y):
    df = pd.concat([X, y], axis=1)
    potable = df[df["Potability"] == 1]
    non_potable = df[df["Potability"] == 0]

    min_count = min(len(potable), len(non_potable))

    potable_bal = resample(potable, replace=False, n_samples=min_count, random_state=42)
    no_potable_bal = resample(non_potable, replace=False, n_samples=min_count, random_state=42)

    df_balanced = pd.concat([potable_bal, no_potable_bal]).sample(frac=1, random_state=42).reset_index(drop=True)
    X_balanced = df_balanced.drop(columns="Potability")
    y_balanced = df_balanced["Potability"]

    return X_balanced, y_balanced


def balance_dataset_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Expérience MLFlow
def run_experiment(X_train, X_test, y_train, y_test, params: dict, run_name: str):
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5]

        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="water_potability_xgb"
        )

        mlflow.set_tag("type", "xgboost_tuning")
        print(f"\n Run '{run_name}' terminé et loggé avec MLflow.")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("water_potability_xgb_experiment")

    df = pd.read_csv("data/water_potability.csv")
    df = preprocess_data(df)
    X = df.drop(columns="Potability")
    y = df["Potability"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    param_sets = [
        {
            "params": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "random_state": 42,
                "eval_metric": "logloss"
            },
            "run_name": "xgb_100_0.1"
        },
        {
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "random_state": 42,
                "eval_metric": "logloss"
            },
            "run_name": "xgb_200_0.05"
        },
        {
            "params": {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "random_state": 42,
                "eval_metric": "logloss",
                "gamma": 0.1,
                "colsample_bytree": 1.0,
                "subsample": 0.6

            },
            "run_name": "xgb_300_0.05"
        }
    ]
    for config in param_sets:
        run_experiment(X_train, X_test, y_train, y_test, config["params"], config["run_name"])

    # Utilisation de la méthode de rééchantillonnage
    X_bal, y_bal = balance_dataset(X, y)
    X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )
    balanced_param_sets = [
        {
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "random_state": 42,
                "eval_metric": "logloss"
            },
            "run_name": "xgb_200_0.05_balanced"
        },
        {
            "params": {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "random_state": 42,
                "eval_metric": "logloss",
                "gamma": 0.1,
                "colsample_bytree": 1.0,
                "subsample": 0.6

            },
            "run_name": "xgb_300_0.05_balanced"
        }
    ]

    for config in balanced_param_sets:
        run_experiment(X_train_bal, X_test_bal, y_train_bal, y_test_bal, config["params"], config["run_name"])


    # Utilisation de SMOTE
    X_smote, y_smote = balance_dataset_smote(X, y)
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(
        X_smote, y_smote, test_size=0.2, stratify=y_smote, random_state=42
    )
    smote_param_sets =[ 
    { 
        "params": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "random_state": 42,
        "eval_metric": "logloss"
        },
        "run_name": "xgb_200_0.05_SMOTE"
    },
    {
        "params": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42,
            "eval_metric": "logloss",
            "gamma": 0.1,
            "colsample_bytree": 1.0,
            "subsample": 0.6
        },
        "run_name": "xgb_300_0.05_SMOTE"
    } ]


    for config in smote_param_sets:
        run_experiment(X_train_smote, X_test_smote, y_train_smote, y_test_smote, config["params"], config["run_name"])

