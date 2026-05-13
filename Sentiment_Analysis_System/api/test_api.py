import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_single_prediction():
    """Test single prediction"""
    print("\n" + "="*60)
    print("Testing Single Prediction")
    print("="*60)
    
    test_cases = [
        "This product is absolutely amazing! I love it so much!",
        "Terrible experience. Very disappointed with the quality.",
        "It's okay, nothing special but does the job.",
        "Best purchase ever! Highly recommended!",
        "Waste of money. Do not buy this product."
    ]
    
    for text in test_cases:
        print(f"\nText: {text}")
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text}
        )
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Sentiment: {result.get('sentiment')}")
        if result.get('confidence'):
            print(f"Confidence: {result.get('confidence'):.4f}")

def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*60)
    print("Testing Batch Prediction")
    print("="*60)
    
    texts = [
        "Excellent product! Very satisfied.",
        "Poor quality. Not recommended.",
        "Average product, nothing special."
    ]
    
    response = requests.post(
        f"{API_URL}/batch_predict",
        json={"texts": texts}
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    for item in result.get('results', []):
        print(f"\nText: {item['text']}")
        print(f"Sentiment: {item['sentiment']}")
        if item.get('confidence'):
            print(f"Confidence: {item['confidence']:.4f}")

def test_error_cases():
    """Test error handling"""
    print("\n" + "="*60)
    print("Testing Error Cases")
    print("="*60)
    
    # Empty text
    print("\nTest 1: Empty text")
    response = requests.post(f"{API_URL}/predict", json={"text": ""})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # No text field
    print("\nTest 2: No text field")
    response = requests.post(f"{API_URL}/predict", json={})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    print("="*60)
    print("SENTIMENT ANALYSIS API - TEST SUITE")
    print("="*60)
    print("\nMake sure the API is running on http://localhost:5000")
    print("Start the API with: python api/app.py")
    
    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_error_cases()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
    
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API")
        print("Please make sure the API is running on http://localhost:5000")
    except Exception as e:
        print(f"\nERROR: {e}")
