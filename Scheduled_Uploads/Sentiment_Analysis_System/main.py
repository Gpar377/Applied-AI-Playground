import os
import warnings
warnings.filterwarnings('ignore')

from preprocessing import load_and_prepare_data, TextPreprocessor
from train_model import SentimentModel, compare_models
from create_dataset import create_sample_dataset

def main():
    """Main training pipeline"""
    print("="*60)
    print("SENTIMENT ANALYSIS SYSTEM - TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Create sample dataset (if not exists)
    if not os.path.exists('data/sentiment_dataset.csv'):
        print("\n[Step 1] Creating sample dataset...")
        create_sample_dataset()
    else:
        print("\n[Step 1] Dataset already exists")
    
    # Step 2: Load and preprocess data
    print("\n[Step 2] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data(
        'data/sentiment_dataset.csv',
        text_column='text',
        label_column='sentiment'
    )
    
    # Step 3: Compare models
    print("\n[Step 3] Training and comparing models...")
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # Step 4: Train and save best model (Logistic Regression)
    print("\n[Step 4] Training final model (Logistic Regression)...")
    final_model = SentimentModel(model_type='logistic')
    final_model.train(X_train, y_train)
    final_model.evaluate(X_test, y_test)
    
    # Step 5: Save model
    print("\n[Step 5] Saving model...")
    final_model.save_model()
    
    # Step 6: Test predictions
    print("\n[Step 6] Testing predictions...")
    test_texts = [
        "This is absolutely amazing! I love it!",
        "Terrible product. Very disappointed.",
        "It's okay, nothing special."
    ]
    
    print("\nSample Predictions:")
    print("-" * 60)
    for text in test_texts:
        prediction = final_model.predict(text)
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {prediction}\n")
    
    print("="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run the API: python api/app.py")
    print("2. Test the API using the test script or Postman")
    print("3. Check screenshots/ folder for visualizations")

if __name__ == "__main__":
    main()
