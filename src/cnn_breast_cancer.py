#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breast Cancer Detection using Convolutional Neural Network (CNN)
================================================================
IEEE Published Project — Automated Detection and Classification of
Breast Cancer Tumour Cells using Machine Learning and Deep Learning
on Histopathological Images.

Authors: Vanshika Garg, Vanshika Jain
Institution: Manipal University Jaipur (B.Tech, 2020-21)
Publication: https://ieeexplore.ieee.org/document/9417996

Pipeline:
    Histopathological Images → Preprocessing → CNN Classification
    → Binary Output: Malignant (1) or Benign (0)

Model Architecture:
    Conv2D (32 filters) → MaxPooling → BatchNorm
    → Conv2D (16 filters) → MaxPooling → BatchNorm → Dropout(0.2)
    → Flatten → Dense(32, ReLU) → Dense(1, Sigmoid)
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import (Convolution2D, MaxPooling2D,
                          BatchNormalization, Flatten,
                          Dense, Dropout)
from keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
IMAGE_SIZE = 64          # Resize all images to 64x64 pixels
BATCH_SIZE = 32
EPOCHS = 10
STEPS_PER_EPOCH = 45

# Update these paths to point to your dataset directory
TRAIN_DIR = "data/training_set"
TEST_DIR  = "data/test_set"

# Class labels — folder names must match these
# 0 = Benign (non-cancerous)
# 1 = Malignant (cancerous)
CLASS_LABELS = {0: "Benign", 1: "Malignant"}

# ---------------------------------------------------------------
# Data Augmentation & Preprocessing
# ---------------------------------------------------------------
# Training set: augmented to improve generalisation
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalise pixel values to [0, 1]
    shear_range=0.2,         # Random shear transformation
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True     # Random horizontal flip
)

# Test set: only rescale, no augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_dataset = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ---------------------------------------------------------------
# CNN Model Architecture
# ---------------------------------------------------------------
classifier = Sequential(name="BreastCancer_CNN")

# Block 1: Conv → MaxPool → BatchNorm
classifier.add(Convolution2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    activation='relu',
    padding='same',
    name="Conv1"
))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="Pool1"))
classifier.add(BatchNormalization(name="BN1"))

# Block 2: Conv → MaxPool → BatchNorm → Dropout
classifier.add(Convolution2D(
    filters=16,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation='relu',
    padding='same',
    name="Conv2"
))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="Pool2"))
classifier.add(BatchNormalization(name="BN2"))
classifier.add(Dropout(0.2, name="Dropout1"))   # Prevents overfitting

# Flatten + Fully Connected Layers
classifier.add(Flatten(name="Flatten"))
classifier.add(Dense(units=32, activation='relu', name="FC1"))
classifier.add(Dense(units=1, activation='sigmoid', name="Output"))  # Binary output

print(classifier.summary())

# ---------------------------------------------------------------
# Compile
# ---------------------------------------------------------------
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Binary classification loss
    metrics=['accuracy']
)

# ---------------------------------------------------------------
# Train
# ---------------------------------------------------------------
history = classifier.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_dataset,
    validation_steps=len(test_dataset),
    verbose=1
)

# ---------------------------------------------------------------
# Save Trained Model
# ---------------------------------------------------------------
os.makedirs("models", exist_ok=True)
classifier.save("models/cnn_breast_cancer.h5")
print("\nModel saved to models/cnn_breast_cancer.h5")

# ---------------------------------------------------------------
# Plot Training & Validation Accuracy / Loss
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training
axes[0].plot(history.history['accuracy'], color='green', label='Accuracy')
axes[0].plot(history.history['loss'], color='red', label='Loss')
axes[0].set_title('Training — Accuracy & Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

# Validation
axes[1].plot(history.history['val_accuracy'], color='green', label='Val Accuracy')
axes[1].plot(history.history['val_loss'], color='red', label='Val Loss')
axes[1].set_title('Validation — Accuracy & Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/training_validation_curves.png", dpi=150)
plt.show()
print("Training curves saved to results/training_validation_curves.png")

# ---------------------------------------------------------------
# Predict on a Single Image
# ---------------------------------------------------------------
def predict_image(image_path: str) -> None:
    """
    Predict whether a histopathological image is Malignant or Benign.

    Args:
        image_path: Path to the image file.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_array  = img_resized.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3) / 255.0

    probability = float(classifier.predict(img_array)[0][0])
    prediction  = CLASS_LABELS[int(probability >= 0.5)]

    print(f"\nImage        : {os.path.basename(image_path)}")
    print(f"Prediction   : {prediction}")
    print(f"Probability  : Malignant = {probability:.4f} | Benign = {1 - probability:.4f}")


# Example usage — replace with an actual image path from your dataset
# predict_image("data/test_set/1/sample_malignant.png")
