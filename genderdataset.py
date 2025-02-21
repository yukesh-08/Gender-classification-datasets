import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Define Paths
MALE = Path(r"D:\YUKESH\GI-project\gender-data\MALE")
FEMALE = Path(r"D:\YUKESH\GI-project\gender-data\FEMALE")

# Labels
Gender_type = [MALE, FEMALE]
Gender_Classification = ["male", "female"]

# Dataframe for storing image paths and labels
data = []

for idx, gender_path in enumerate(Gender_type):
    for imagepath in tqdm(list(gender_path.glob("*.jpg")), desc=f"Loading {Gender_Classification[idx]}"):
        data.append((str(imagepath), Gender_Classification[idx]))  # Store path as string, label as string

# Convert to DataFrame
df = pd.DataFrame(data, columns=["image", "Gender_type"])

# Split Data (80% Train, 20% Validation)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["Gender_type"], random_state=42)

# Image Data Generator with Advanced Augmentation
IMG_SIZE = (224, 224)  # Increased size for better feature learning
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col="image", y_col="Gender_type",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)

val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col="image", y_col="Gender_type",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)

# **Using EfficientNetB0 as a Feature Extractor**
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# **Optimized CNN Model**
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# **Compile Model**
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# **Callbacks for Best Model & Early Stopping**
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_gender_model.h5", save_best_only=True, monitor="val_accuracy")

# **Train Model (30-50 Epochs)**
EPOCHS = 30
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=[early_stopping, model_checkpoint])

# Save Final Model
model.save("final_gender_classification.h5")

print("Model training complete. Best model saved as best_gender_model.h5")
