import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from data_preprocessing.preprocess import preprocess_data, load_data

import os

def train_models(data_path):
    data = load_data(data_path)
    data, _ = preprocess_data(data)

    # Drop customerID if present
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])

    X = data.drop('Churn', axis=1)
    y = data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("CustomerChurnPrediction")

    # Ensure model directories exist
    os.makedirs("model_training/random_forest_model", exist_ok=True)
    os.makedirs("model_training/xgboost_model", exist_ok=True)

    with mlflow.start_run():
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_preds)
        rf_auc = roc_auc_score(y_test, rf_preds)

        mlflow.log_metric("rf_accuracy", rf_acc)
        mlflow.log_metric("rf_auc", rf_auc)
        import shutil
        shutil.rmtree("model_training/random_forest_model", ignore_errors=True)
        mlflow.sklearn.save_model(rf, "model_training/random_forest_model")

        # XGBoost
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_preds)
        xgb_auc = roc_auc_score(y_test, xgb_preds)

        mlflow.log_metric("xgb_accuracy", xgb_acc)
        mlflow.log_metric("xgb_auc", xgb_auc)
        shutil.rmtree("model_training/xgboost_model", ignore_errors=True)
        mlflow.sklearn.save_model(xgb_model, "model_training/xgboost_model")

        print(f"Random Forest Accuracy: {rf_acc}, AUC: {rf_auc}")
        print(f"XGBoost Accuracy: {xgb_acc}, AUC: {xgb_auc}")

if __name__ == "__main__":
    # Use the downloaded Kaggle dataset path for training
    import os
    base_path = os.path.expanduser('~/.cache/kagglehub/datasets/blastchar/telco-customer-churn/versions/1')
    dataset_file = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    dataset_path = os.path.join(base_path, dataset_file)
    train_models(dataset_path)
