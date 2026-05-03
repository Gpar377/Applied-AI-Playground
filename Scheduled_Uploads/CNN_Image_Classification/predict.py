"""
Prediction Script - Use trained CNN model to classify new images
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

IMG_SIZE = 150

def load_model(model_path='cnn_image_classifier.h5'):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first by running 'train_model.py'")
        sys.exit(1)
    
    model = keras.models.load_model(model_path)
    print(f"Model loaded from '{model_path}'")
    return model

def predict_image(model, image_path):
    """Predict the class of a single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return None, None
    
    # Load and preprocess image
    img = keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    predicted_class = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    # Display result
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    return predicted_class, confidence

def predict_batch(model, image_folder):
    """Predict classes for all images in a folder"""
    if not os.path.exists(image_folder):
        print(f"Error: Folder '{image_folder}' not found!")
        return
    
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in '{image_folder}'")
        return
    
    print(f"\nProcessing {len(image_files)} images...\n")
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        try:
            img = keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array, verbose=0)[0][0]
            predicted_class = "Dog" if prediction > 0.5 else "Cat"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            results.append({
                'file': img_file,
                'class': predicted_class,
                'confidence': confidence
            })
            
            print(f"{img_file:30s} -> {predicted_class:5s} ({confidence:.2%})")
        
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    return results

def main():
    """Main execution function"""
    print("=" * 60)
    print("CNN Image Classifier - Prediction Tool")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Single image:  python predict.py <image_path>")
        print("  Batch predict: python predict.py <folder_path>")
        print("\nExample:")
        print("  python predict.py cat.jpg")
        print("  python predict.py ./test_images/")
        return
    
    path = sys.argv[1]
    
    # Check if path is file or directory
    if os.path.isfile(path):
        predict_image(model, path)
    elif os.path.isdir(path):
        predict_batch(model, path)
    else:
        print(f"Error: '{path}' is not a valid file or directory!")

if __name__ == "__main__":
    main()
