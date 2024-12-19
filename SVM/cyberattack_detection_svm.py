import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import SVC

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

# Train SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=21)
svm_model.fit(X_train_scaled, y_train)

# Predict
train_pred = svm_model.predict(X_train_scaled)
test_pred = svm_model.predict(X_test_scaled)

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