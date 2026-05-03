import requests

url = "http://127.0.0.1:8000/predict"

sample_data = {
    "customerID": "0001-ABCD",
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12.0,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 2,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "Contract": 1,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.5
}

response = requests.post(url, json=sample_data)
try:
    print("Response:", response.json())
except Exception as e:
    print("Failed to decode JSON response:", e)
    print("Response content:", response.text)
