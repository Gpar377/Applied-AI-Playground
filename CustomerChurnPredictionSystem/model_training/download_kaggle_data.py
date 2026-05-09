import kagglehub

def download_telco_churn_dataset():
    # Download latest version of the dataset from Kaggle
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    print("Path to dataset files:", path)
    return path

if __name__ == "__main__":
    download_telco_churn_dataset()
