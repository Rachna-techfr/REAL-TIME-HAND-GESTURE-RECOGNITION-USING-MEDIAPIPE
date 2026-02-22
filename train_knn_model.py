# train_knn_model.py
# This script trains a KNN model on the combined gesture dataset and saves the model.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("gesture_dataset.csv")
# Split into features (X) and labels (y)
X = data.drop("label", axis=1)
y = data["label"]
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Initialize KNN
knn = KNeighborsClassifier(n_neighbors=5)
# Train
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)
# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {acc * 100:.2f}%")
# Save model
joblib.dump(knn, "gesture_knn_model.pkl")
print("💾 Model saved as gesture_knn_model.pkl")
# Class distribution
print("\nClass counts:\n", y.value_counts())

