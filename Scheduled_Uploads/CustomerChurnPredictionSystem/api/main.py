from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.preprocess import preprocess_data

app = FastAPI()

from typing import Optional

class CustomerData(BaseModel):
    # Define fields based on features from the dataset after preprocessing
    customerID: Optional[str] = None
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

model = None

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.sklearn.load_model("model_training/random_forest_model")

@app.post("/predict")
def predict(data: CustomerData):
    import logging
    try:
        input_df = pd.DataFrame([data.dict()])
        # Drop customerID if present, as model was trained without it
        if 'customerID' in input_df.columns:
            input_df = input_df.drop(columns=['customerID'])
        # Apply preprocessing to input data to match training preprocessing
        input_df_processed, _ = preprocess_data(input_df)
        prediction = model.predict(input_df_processed)[0]
        return {"churn_prediction": int(prediction)}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": "Prediction failed", "details": str(e)}
