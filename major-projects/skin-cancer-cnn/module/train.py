"""
Skin Cancer Detection CNN Training Module

This module contains the training script for the skin cancer classification model.
Architecture based on custom CNN optimized for skin lesion classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 1  # Binary classification (sigmoid output)


def create_model():
    """
    Create the CNN model architecture.
    
    Architecture (Larger Model):
        - Input: 224 × 224 × 3
        - Conv2D (32 filters, 3×3) + BatchNorm + ReLU + MaxPool
        - Conv2D (64 filters, 3×3) + BatchNorm + ReLU + MaxPool
        - Conv2D (128 filters, 3×3) + BatchNorm + ReLU + MaxPool
        - Conv2D (256 filters, 3×3) + BatchNorm + ReLU + MaxPool
        - Conv2D (512 filters, 3×3) + BatchNorm + ReLU + MaxPool
        - Flatten
        - Dense (1024 units) + Dropout (0.5)
        - Dense (512 units) + Dropout (0.5)
        - Output (Sigmoid, 1 unit)
    
    Returns:
        tensorflow.keras.models.Sequential: Compiled CNN model
    """
    from tensorflow.keras.layers import BatchNormalization
    
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fifth Convolutional Block
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten
        Flatten(),
        
        # Dense Layers with Dropout
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output Layer
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model with Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_data_generators(train_dir, test_dir):
    """
    Create data generators for training and validation.
    
    Args:
        train_dir (str): Path to training data directory
        test_dir (str): Path to test/validation data directory
    
    Returns:
        tuple: (train_generator, test_generator)
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    return train_generator, test_generator


def get_callbacks(model_save_path):
    """
    Create training callbacks.
    
    Args:
        model_save_path (str): Path to save the best model
    
    Returns:
        list: List of callbacks
    """
    callbacks = [
        # Save best model
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def train_model(train_dir, test_dir, model_save_path='model/skin_cancer_model.h5'):
    """
    Train the skin cancer classification model.
    
    Args:
        train_dir (str): Path to training data directory
        test_dir (str): Path to test/validation data directory
        model_save_path (str): Path to save the trained model
    
    Returns:
        tuple: (model, history)
    """
    print("=" * 50)
    print("Skin Cancer Detection - CNN Training")
    print("=" * 50)
    
    # Create model
    print("\n📦 Creating model...")
    model = create_model()
    model.summary()
    
    # Create data generators
    print("\n📊 Loading data...")
    train_generator, test_generator = get_data_generators(train_dir, test_dir)
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Get callbacks
    callbacks = get_callbacks(model_save_path)
    
    # Train the model
    print("\n🚀 Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    print("\n📈 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {model_save_path}")
    
    return model, history


def main():
    """Main function to run training."""
    # Set paths - update these according to your dataset location
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
    TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'model', 'skin_cancer_model.h5')
    
    # Check if data directories exist
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: Training directory not found: {TRAIN_DIR}")
        print("Please ensure your dataset is organized as:")
        print("  data/")
        print("    train/")
        print("      benign/")
        print("      malignant/")
        print("    test/")
        print("      benign/")
        print("      malignant/")
        return
    
    if not os.path.exists(TEST_DIR):
        print(f"❌ Error: Test directory not found: {TEST_DIR}")
        return
    
    # Train the model
    model, history = train_model(TRAIN_DIR, TEST_DIR, MODEL_SAVE_PATH)
    
    return model, history


if __name__ == '__main__':
    main()
