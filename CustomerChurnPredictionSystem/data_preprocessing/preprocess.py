import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(source):
    """
    Load dataset from CSV file path or URL.
    """
    if isinstance(source, str) and source.startswith('http'):
        data = pd.read_csv(source)
    else:
        data = pd.read_csv(source)
    return data

def preprocess_data(df):
    """
    Perform data cleaning and feature engineering.
    """
    # Example: Fill missing values
    df.fillna(method='ffill', inplace=True)

    # Example: Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Add more feature engineering steps here

    return df, label_encoders

if __name__ == "__main__":
    # Example usage
    data = load_data('data/customer_churn.csv')
    processed_data, encoders = preprocess_data(data)
    print(processed_data.head())
