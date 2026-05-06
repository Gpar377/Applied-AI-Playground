# Customer Churn Prediction System

This project implements an end-to-end machine learning pipeline for customer churn prediction.

## Features

- Data preprocessing with feature engineering
- Training with multiple models (Random Forest, XGBoost)
- Model selection & evaluation with MLflow
- REST API using FastAPI for predictions
- Streamlit dashboard for visualization
- Dockerized deployment

## Setup

1. Download dataset using Kaggle or use provided dataset.
2. Train models using `model_training/train.py`.
3. Run API server using Uvicorn.
4. Run Streamlit dashboard.

## Usage

- API endpoint: `/predict` accepts customer data and returns churn prediction.
- Dashboard: Visualize model metrics and customer data insights.

## Dependencies

- Python 3.11
- scikit-learn
- xgboost
- mlflow
- fastapi
- uvicorn
- streamlit

## Author

Created by Parthiv Gopa.
