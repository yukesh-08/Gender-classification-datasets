import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
from tqdm import tqdm
import joblib  # For saving the trained model

# Define Paths
MALE = Path(r"D:\YUKESH\GI-project\gender-data\MALE")
FEMALE = Path(r"D:\YUKESH\GI-project\gender-data\FEMALE")

# Labels
Gender_type = [MALE, FEMALE]
Gender_Classification = ["male", "female"]

# Collect image paths and labels
data = []
for idx, gender_path in enumerate(Gender_type):
    for imagepath in tqdm(list(gender_path.glob("*.jpg")), desc=f"Loading {Gender_Classification[idx]}"):
        data.append((str(imagepath), idx))  # Store path as string, label as 0/1

# Convert to DataFrame
df = pd.DataFrame(data, columns=["image", "Gender_type"])

# Train-Test Split (80% Train, 20% Test)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Gender_type"], random_state=42)

# Load Pretrained EfficientNetB3 (Feature Extractor)
IMG_SIZE = (300, 300)  # Increased size for better feature learning
base_model = EfficientNetB3(weights='imagenet', include_top=False, pooling="avg", input_shape=(300, 300, 3))

# Function to extract features from an image
def extract_features(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    features = base_model.predict(img_array)
    return features.flatten()

# Extract Features for Train and Test Data
X_train = np.array([extract_features(img) for img in tqdm(train_df["image"], desc="Extracting Train Features")])
y_train = np.array(train_df["Gender_type"])

X_test = np.array([extract_features(img) for img in tqdm(test_df["image"], desc="Extracting Test Features")])
y_test = np.array(test_df["Gender_type"])

# Train XGBoost Classifier with Optimized Hyperparameters
xgb_model = XGBClassifier(
    n_estimators=500,  # More trees for better learning
    learning_rate=0.03,  
    max_depth=8,  
    subsample=0.9,  
    colsample_bytree=0.9,  
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy * 100:.2f}%")

# Save XGBoost Model
joblib.dump(xgb_model, "xgboost_gender_model_b3.pkl")
print("Model saved as xgboost_gender_model_b3.pkl")
