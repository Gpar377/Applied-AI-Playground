import joblib
from preprocessing import TextPreprocessor

def predict_sentiment(text):
    """Quick sentiment prediction"""
    try:
        # Load model and vectorizer
        model = joblib.load('models/sentiment_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        preprocessor = TextPreprocessor()
        
        # Preprocess and predict
        cleaned_text = preprocessor.clean_text(text)
        
        if len(cleaned_text) == 0:
            return "Error: Text too short after preprocessing"
        
        text_vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vectorized)[0]
        
        # Get confidence if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities)
            return f"Sentiment: {prediction.upper()} (Confidence: {confidence:.2%})"
        else:
            return f"Sentiment: {prediction.upper()}"
    
    except FileNotFoundError:
        return "Error: Model not found. Please train the model first by running: python main.py"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("="*60)
    print("SENTIMENT ANALYSIS - QUICK PREDICTION")
    print("="*60)
    print("\nType 'quit' to exit\n")
    
    while True:
        text = input("Enter text to analyze: ").strip()
        
        if text.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        if not text:
            print("Please enter some text.\n")
            continue
        
        result = predict_sentiment(text)
        print(f"\n{result}\n")
        print("-"*60)
