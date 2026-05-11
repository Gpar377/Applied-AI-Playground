# Usage Guide

## Installation

```bash
pip install -r requirements.txt
```

## Training the Model

### Using Sample Dataset
```bash
python main.py
```

This will:
1. Create sample dataset (1800 reviews)
2. Preprocess text data
3. Train 3 models (Logistic Regression, Naive Bayes, SVM)
4. Compare model performances
5. Save best model to models/ folder
6. Generate visualizations in screenshots/ folder

### Using Your Own Dataset

1. Prepare CSV file with columns: `text` and `sentiment`
2. Place in data/ folder
3. Update main.py:

```python
X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data(
    'data/your_dataset.csv',
    text_column='text',
    label_column='sentiment'
)
```

4. Run training:
```bash
python main.py
```

## Using the API

### Start API Server
```bash
cd api
python app.py
```

API will be available at http://localhost:5000

### API Endpoints

**Health Check**
```bash
GET http://localhost:5000/health
```

**Single Prediction**
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "text": "This product is amazing!"
}
```

Response:
```json
{
  "text": "This product is amazing!",
  "sentiment": "positive",
  "confidence": 0.9876
}
```

**Batch Prediction**
```bash
POST http://localhost:5000/batch_predict
Content-Type: application/json

{
  "texts": ["Great product!", "Terrible experience.", "It's okay."]
}
```

### Testing API

```bash
python api/test_api.py
```

Or using curl:
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"This is amazing!\"}"
```

## Using Web Interface

```bash
streamlit run app_ui.py
```

Browser will open at http://localhost:8501

## Quick Predictions

```bash
python predict.py
```

Interactive command-line interface for quick predictions.

## Troubleshooting

**Module not found**
```bash
pip install -r requirements.txt
```

**NLTK data not found**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

**Model file not found**
```bash
python main.py
```

**API connection error**
- Ensure API is running: `python api/app.py`
- Check port 5000 is available
- Try: http://localhost:5000/health
