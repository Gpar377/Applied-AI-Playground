from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import TextPreprocessor

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
MODEL_PATH = '../models/sentiment_model.pkl'
VECTORIZER_PATH = '../models/vectorizer.pkl'

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    preprocessor = TextPreprocessor()
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    vectorizer = None
    preprocessor = None

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Sentiment Analysis API',
        'status': 'active',
        'endpoints': {
            '/predict': 'POST - Predict sentiment for text',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for input text"""
    try:
        # Get input data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'message': 'Please provide text in JSON format: {"text": "your text here"}'
            }), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                'error': 'Empty text',
                'message': 'Please provide non-empty text'
            }), 400
        
        # Preprocess text
        cleaned_text = preprocessor.clean_text(text)
        
        if len(cleaned_text) == 0:
            return jsonify({
                'error': 'Text too short after preprocessing',
                'message': 'Please provide more meaningful text'
            }), 400
        
        # Vectorize and predict
        text_vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vectorized)[0]
        
        # Get prediction probability
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = float(max(probabilities))
        else:
            confidence = None
        
        # Return result
        return jsonify({
            'text': text,
            'sentiment': prediction,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict sentiment for multiple texts"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'No texts provided',
                'message': 'Please provide texts in JSON format: {"texts": ["text1", "text2"]}'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'error': 'Invalid format',
                'message': 'texts must be a list'
            }), 400
        
        results = []
        for text in texts:
            cleaned_text = preprocessor.clean_text(text)
            if len(cleaned_text) > 0:
                text_vectorized = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_vectorized)[0]
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(text_vectorized)[0]
                    confidence = float(max(probabilities))
                else:
                    confidence = None
                
                results.append({
                    'text': text,
                    'sentiment': prediction,
                    'confidence': confidence
                })
            else:
                results.append({
                    'text': text,
                    'sentiment': 'unknown',
                    'confidence': None,
                    'error': 'Text too short after preprocessing'
                })
        
        return jsonify({
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Sentiment Analysis API...")
    print("API will be available at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
