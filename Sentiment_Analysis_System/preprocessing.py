import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_dataset(self, df, text_column, label_column):
        """Preprocess entire dataset"""
        print("Starting preprocessing...")
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        print(f"Preprocessing complete. Dataset size: {len(df)}")
        return df

def load_and_prepare_data(filepath, text_column='text', label_column='sentiment'):
    """Load and prepare data for training"""
    print(f"Loading data from {filepath}...")
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess
    df = preprocessor.preprocess_dataset(df, text_column, label_column)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df[label_column], 
        test_size=0.2, 
        random_state=42,
        stratify=df[label_column]
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    # Example usage
    print("Text Preprocessor Module")
    print("Import this module to use preprocessing functions")
