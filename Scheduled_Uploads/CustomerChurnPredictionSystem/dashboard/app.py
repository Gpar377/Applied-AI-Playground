import streamlit as st
import pandas as pd
import requests

st.title("Customer Churn Prediction Dashboard")

uploaded_file = st.file_uploader("Upload CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(data)

    if st.button("Predict Churn"):
        response = requests.post("http://localhost:8000/predict", json=data.iloc[0].to_dict())
        if response.status_code == 200:
            st.write("Prediction Result:")
            st.write(response.json())
        else:
            st.error("Prediction failed. Make sure the API is running.")
