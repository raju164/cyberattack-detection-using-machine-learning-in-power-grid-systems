import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
import tensorflow as tf

# Disable GPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load and preprocess data
df = pd.read_csv('/combined_output.csv')
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

# Separate features and target
X = df.drop('marker', axis=1)
y = df['marker']

# Encode target labels (e.g., "Attack" -> 1, "Normal" -> 0)
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # Converts strings to integers

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Standardize feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure data is in proper format for TensorFlow
X_train_scaled = np.array(X_train_scaled, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test_scaled = np.array(X_test_scaled, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    verbose=0  # Suppress epoch details
)

# Predict
train_pred = (model.predict(X_train_scaled) > 0.5).astype(int)
test_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

# Evaluate performance for training data
print("\033[031m\033[1mTraining Data Performance Metrics:")
print(f"Accuracy       : {accuracy_score(y_train, train_pred):.2f}")
print(f"Precision      : {precision_score(y_train, train_pred):.2f}")
print(f"Recall         : {recall_score(y_train, train_pred):.2f}")
print(f"F1-Score       : {f1_score(y_train, train_pred):.2f}\n")

print("\033[032m\033[1mTraining Classification Report:\n", classification_report(y_train, train_pred), '\n')
print("\033[034m\033[1mTraining Confusion Matrix:\n", confusion_matrix(y_train, train_pred), '\n')

# Evaluate performance for testing data
print("\033[031m\033[1mTesting Data Performance Metrics:")
print(f"Accuracy       : {accuracy_score(y_test, test_pred):.2f}")
print(f"Precision      : {precision_score(y_test, test_pred):.2f}")
print(f"Recall         : {recall_score(y_test, test_pred):.2f}")
print(f"F1-Score       : {f1_score(y_test, test_pred):.2f}\n")

print("\033[032m\033[1mTesting Classification Report:\n", classification_report(y_test, test_pred), '\n')
print("\033[034m\033[1mTesting Confusion Matrix:\n", confusion_matrix(y_test, test_pred), '\n')