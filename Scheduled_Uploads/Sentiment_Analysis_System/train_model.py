import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from preprocessing import load_and_prepare_data, TextPreprocessor

class SentimentModel:
    def __init__(self, model_type='logistic'):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model_type = model_type
        
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'svm':
            self.model = LinearSVC(random_state=42, max_iter=1000)
        else:
            raise ValueError("Invalid model type")
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        
        print("Training complete!")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        print(f"\n{self.model_type.upper()} Model Performance:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, y_test.unique())
        
        return metrics, y_pred
    
    def plot_confusion_matrix(self, cm, classes):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'screenshots/confusion_matrix_{self.model_type}.png', dpi=300)
        print(f"Confusion matrix saved to screenshots/confusion_matrix_{self.model_type}.png")
    
    def predict(self, text):
        """Predict sentiment for new text"""
        text_vec = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vec)[0]
        return prediction
    
    def save_model(self, model_path='models/sentiment_model.pkl', 
                   vectorizer_path='models/vectorizer.pkl'):
        """Save trained model and vectorizer"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    @staticmethod
    def load_model(model_path='models/sentiment_model.pkl', 
                   vectorizer_path='models/vectorizer.pkl'):
        """Load trained model and vectorizer"""
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Model and vectorizer loaded successfully!")
        return model, vectorizer

def compare_models(X_train, X_test, y_train, y_test):
    """Compare different models"""
    models = ['logistic', 'naive_bayes', 'svm']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*50}")
        sentiment_model = SentimentModel(model_type=model_type)
        sentiment_model.train(X_train, y_train)
        metrics, _ = sentiment_model.evaluate(X_test, y_test)
        results[model_type] = metrics
    
    # Plot comparison
    plot_model_comparison(results)
    
    return results

def plot_model_comparison(results):
    """Plot model comparison"""
    df = pd.DataFrame(results).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', ax=ax)
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('screenshots/model_comparison.png', dpi=300)
    print("Model comparison saved to screenshots/model_comparison.png")

if __name__ == "__main__":
    print("Sentiment Analysis Model Training")
    print("="*50)
    
    # Note: Replace with your actual dataset path
    print("\nTo train the model, use:")
    print("1. Prepare your dataset (CSV with 'text' and 'sentiment' columns)")
    print("2. Run: python train_model.py")
