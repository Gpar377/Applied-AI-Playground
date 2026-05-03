"""
Alternative CNN Training Script - Using tf.keras.utils.image_dataset_from_directory
Simpler approach with automatic dataset handling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Configuration
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 20

def download_and_prepare_dataset():
    """Download dataset using keras utilities"""
    print("Downloading dataset...")
    dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    path_to_zip = keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)
    
    extract_dir = os.path.dirname(path_to_zip)
    base_dir = os.path.join(extract_dir, 'PetImages')
    
    # Check alternative paths
    if not os.path.exists(base_dir):
        for item in os.listdir(extract_dir):
            potential = os.path.join(extract_dir, item)
            if os.path.isdir(potential):
                pet_path = os.path.join(potential, 'PetImages')
                if os.path.exists(pet_path):
                    base_dir = pet_path
                    break
    
    print(f"Dataset location: {base_dir}")
    
    # Clean corrupted images
    num_removed = 0
    for folder in ['Cat', 'Dog']:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                try:
                    with open(fpath, 'rb') as f:
                        if tf.compat.as_bytes("JFIF") not in f.peek(10):
                            os.remove(fpath)
                            num_removed += 1
                except:
                    os.remove(fpath)
                    num_removed += 1
    
    print(f"Cleaned {num_removed} corrupted files")
    return base_dir

def create_datasets(base_dir):
    """Create train and validation datasets"""
    train_ds = keras.utils.image_dataset_from_directory(
        base_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        base_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    # Normalize and augment
    normalization = layers.Rescaling(1./255)
    
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization(x)), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))
    
    # Performance optimization
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def build_model():
    """Build CNN model"""
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100)
    print("\nTraining curves saved as 'training_history.png'")

def main():
    print("="*60)
    print("CNN Image Classification - Simplified Training")
    print("="*60)
    
    # Prepare dataset
    base_dir = download_and_prepare_dataset()
    
    # Create datasets
    print("\nCreating datasets...")
    train_ds, val_ds = create_datasets(base_dir)
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    model.summary()
    
    # Train
    print("\nTraining model...")
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )
    
    # Evaluate
    print("\nEvaluating model...")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Plot
    plot_history(history)
    
    # Save
    model.save('cnn_image_classifier.h5')
    print("\nModel saved as 'cnn_image_classifier.h5'")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
