import pandas as pd
import numpy as np

def create_sample_dataset():
    """Create a sample sentiment dataset for testing"""
    
    # Sample positive reviews
    positive_samples = [
        "This product is amazing! I love it so much.",
        "Excellent quality and fast delivery. Highly recommended!",
        "Best purchase I've made this year. Very satisfied.",
        "Outstanding service and great product quality.",
        "Absolutely wonderful! Exceeded my expectations.",
        "I'm very happy with this purchase. Worth every penny.",
        "Fantastic product! Will definitely buy again.",
        "Great value for money. Very pleased with the quality.",
        "Superb! This is exactly what I was looking for.",
        "Love it! Perfect in every way.",
        "Incredible quality and excellent customer service.",
        "This is the best product I've ever used.",
        "Amazing! I would recommend this to everyone.",
        "Perfect! No complaints at all.",
        "Excellent product. Very happy with my purchase.",
    ] * 40  # 600 positive samples
    
    # Sample negative reviews
    negative_samples = [
        "Terrible product. Complete waste of money.",
        "Very disappointed. Poor quality and bad service.",
        "Worst purchase ever. Do not recommend.",
        "Horrible experience. Product broke after one use.",
        "Not worth the price. Very unsatisfied.",
        "Poor quality. I want my money back.",
        "Awful product. Doesn't work as advertised.",
        "Terrible customer service. Very frustrated.",
        "Completely useless. Total disappointment.",
        "Bad quality and overpriced. Avoid this.",
        "Waste of money. Very poor quality.",
        "Disappointing product. Not as described.",
        "Horrible! Broke within a week.",
        "Very bad experience. Would not buy again.",
        "Poor quality control. Defective product.",
    ] * 40  # 600 negative samples
    
    # Sample neutral reviews
    neutral_samples = [
        "It's okay. Nothing special but does the job.",
        "Average product. Met basic expectations.",
        "Decent quality for the price.",
        "It's fine. Not great, not terrible.",
        "Acceptable product. Could be better.",
        "Standard quality. Nothing to complain about.",
        "It works as expected. No surprises.",
        "Reasonable product. Gets the job done.",
        "Fair quality. About what I expected.",
        "It's alright. Serves its purpose.",
        "Moderate quality. Acceptable for the price.",
        "Normal product. Nothing extraordinary.",
        "Satisfactory. Does what it's supposed to do.",
        "Basic product. Meets minimum requirements.",
        "Okay quality. Neither good nor bad.",
    ] * 40  # 600 neutral samples
    
    # Create dataframe
    data = {
        'text': positive_samples + negative_samples + neutral_samples,
        'sentiment': ['positive'] * len(positive_samples) + 
                     ['negative'] * len(negative_samples) + 
                     ['neutral'] * len(neutral_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('data/sentiment_dataset.csv', index=False)
    print(f"Sample dataset created with {len(df)} samples")
    print(f"Distribution:\n{df['sentiment'].value_counts()}")
    print("Dataset saved to data/sentiment_dataset.csv")
    
    return df

if __name__ == "__main__":
    create_sample_dataset()
