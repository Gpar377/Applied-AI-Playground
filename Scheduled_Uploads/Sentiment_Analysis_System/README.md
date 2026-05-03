# Sentiment Analysis & Review Classification System

An end-to-end NLP project that classifies text (reviews, comments, tweets, feedback) into sentiment categories: Positive, Negative, or Neutral.

## Project Overview

This project demonstrates a complete Natural Language Processing pipeline including data preprocessing, model training, evaluation, and deployment via REST API. The system analyzes emotions and opinions expressed in text data.

## Features

- **Text Preprocessing Pipeline**: Cleaning, tokenization, lemmatization, stopword removal
- **Multiple ML Models**: Logistic Regression, Naive Bayes, SVM
- **Model Comparison**: Automated performance comparison across models
- **REST API**: Flask-based API for real-time predictions
- **Batch Processing**: Support for multiple text predictions
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Visualization**: Performance metrics and confusion matrix plots

## Project Structure

```
Sentiment_Analysis_System/
├── api/
│   ├── app.py              # Flask API application
│   └── test_api.py         # API testing script
├── data/
│   └── sentiment_dataset.csv  # Training dataset
├── models/
│   ├── sentiment_model.pkl    # Trained model
│   └── vectorizer.pkl         # TF-IDF vectorizer
├── screenshots/
│   ├── confusion_matrix_*.png
│   └── model_comparison.png
├── preprocessing.py        # Text preprocessing module
├── train_model.py         # Model training module
├── create_dataset.py      # Sample dataset generator
├── main.py               # Main training pipeline
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Sentiment_Analysis_System
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (automatic on first run)
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Training the Model

### Step 1: Create Sample Dataset (Optional)
```bash
python create_dataset.py
```
This creates a sample dataset with 1800 reviews (600 positive, 600 negative, 600 neutral).

### Step 2: Train the Model
```bash
python main.py
```

This will:
- Load and preprocess the data
- Train multiple models (Logistic Regression, Naive Bayes, SVM)
- Compare model performances
- Save the best model
- Generate visualization plots

### Training Output
```
SENTIMENT ANALYSIS SYSTEM - TRAINING PIPELINE
==================================================
[Step 1] Creating sample dataset...
[Step 2] Loading and preprocessing data...
[Step 3] Training and comparing models...
[Step 4] Training final model...
[Step 5] Saving model...
[Step 6] Testing predictions...
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.95 | ~0.95 | ~0.95 | ~0.95 |
| Naive Bayes | ~0.92 | ~0.92 | ~0.92 | ~0.92 |
| SVM | ~0.94 | ~0.94 | ~0.94 | ~0.94 |

## API Deployment

### Start the API Server
```bash
cd api
python app.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
GET http://localhost:5000/health
```

#### 2. Single Prediction
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "text": "This product is amazing!"
}
```

**Response:**
```json
{
  "text": "This product is amazing!",
  "sentiment": "positive",
  "confidence": 0.9876
}
```

#### 3. Batch Prediction
```bash
POST http://localhost:5000/batch_predict
Content-Type: application/json

{
  "texts": [
    "Great product!",
    "Terrible experience.",
    "It's okay."
  ]
}
```

### Testing the API

Run the test suite:
```bash
python api/test_api.py
```

Or use curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This is amazing!\"}"
```

## Using Your Own Dataset

Replace the sample dataset with your own CSV file:

1. **Format**: CSV file with columns `text` and `sentiment`
2. **Place**: Save in `data/` folder
3. **Update**: Modify `main.py` to point to your dataset

Example CSV format:
```csv
text,sentiment
"Great product! Love it!",positive
"Terrible quality.",negative
"It's okay.",neutral
```

## Key Concepts Covered

### 1. Text Preprocessing
- Lowercasing
- URL and mention removal
- Special character removal
- Tokenization
- Stopword removal
- Lemmatization

### 2. Feature Engineering
- TF-IDF Vectorization
- N-gram features (unigrams and bigrams)
- Maximum 5000 features

### 3. Machine Learning Models
- **Logistic Regression**: Best overall performance
- **Naive Bayes**: Fast training, good baseline
- **SVM**: High accuracy, slower training

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Technologies Used

- **Python 3.8+**
- **scikit-learn**: Machine learning models
- **NLTK**: Natural language processing
- **Flask**: REST API framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization

## Screenshots

Check the `screenshots/` folder for:
- Confusion matrices for each model
- Model performance comparison charts
- Training metrics visualization

## Learning Outcomes

This project demonstrates:
- End-to-end NLP pipeline development
- Text preprocessing and feature engineering
- Multiple ML model training and comparison
- Model evaluation and performance metrics
- REST API development and deployment
- Production-ready code structure
- Documentation and testing best practices

## Future Enhancements

- Deep Learning models (LSTM, BERT)
- Web UI using Streamlit or React
- Docker containerization
- Cloud deployment (AWS/Azure/GCP)
- Real-time Twitter sentiment analysis
- Multi-language support

## Project Report

**Project**: Sentiment Analysis & Review Classification System

**Objective**: Built an end-to-end NLP system to classify text sentiments into positive, negative, or neutral categories using machine learning.

**Approach**: Implemented comprehensive text preprocessing pipeline with tokenization, lemmatization, and TF-IDF vectorization. Trained and compared multiple ML models (Logistic Regression, Naive Bayes, SVM) achieving 95% accuracy. Deployed the best model via Flask REST API with batch prediction support.

**Technologies**: Python, scikit-learn, NLTK, Flask, Pandas, NumPy

**Outcome**: Successfully created a production-ready sentiment analysis system with REST API, comprehensive documentation, and automated testing. The project demonstrates strong understanding of NLP workflows, ML model development, and API deployment.

**Key Learnings**: Gained hands-on experience in text preprocessing, feature engineering, model evaluation, hyperparameter tuning, and building deployable AI applications. Developed skills in API development, error handling, and creating portfolio-ready projects.

## Author

**Your Name**
- GitHub: [@Gpar377](https://github.com/Gpar377)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset: Sample dataset created for demonstration
- Inspiration: Real-world sentiment analysis applications
- Framework: Based on industry-standard NLP practices

---

**Note**: This project is designed for educational purposes and portfolio demonstration. For production use with real datasets, additional considerations for data privacy, security, and scalability should be implemented.
