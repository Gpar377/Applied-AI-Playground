import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# For demonstration, create a simple CNN model architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')  # Output calorie estimate
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = create_model()
model.summary()

# Dummy data generation for demonstration
def generate_dummy_data(num_samples=100):
    X = np.random.rand(num_samples, 64, 64, 3)
    y = np.random.randint(50, 700, size=(num_samples, 1))  # calorie values
    return X, y

X_train, y_train = generate_dummy_data()

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=16)

# Example prediction on a random image
test_img = np.random.rand(1, 64, 64, 3)
predicted_calories = model.predict(test_img)
print(f"Predicted Calories: {predicted_calories[0][0]:.2f}")

# Note: For real use, replace dummy data with actual labeled food images and calorie info.
