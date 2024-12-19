import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler 
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN 
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score,classification_report, confusion_matrix )
from sklearn.model_selection import train_test_split 
import warnings 
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
# Load the dataset
df = pd.read_csv('combined_output.csv')
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)] # Clean the dataset
# Encode the target column 
encoder = LabelEncoder()
df['marker'] = encoder.fit_transform(df['marker'])
# Define features (X) and target (y)
X = df.drop('marker', axis=1)
y = df['marker']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
# Standardize the data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Apply PCA for dimensionality reduction
pca = PCA(n_components=2) # Adjust the number of components if needed
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=4) # Adjust parameters based on your dataset
dbscan.fit(X_train_pca)
# Predict DBSCAN labels for training and testing data
train_labels = dbscan.labels_ # Labels for training data
test_labels = dbscan.fit_predict(X_test_pca) # Predict for test data
# Map DBSCAN labels (-1 for noise) to binary classification (0: Normal, 1: Cyberattack)
y_train_pred_dbscan = np.where(train_labels == -1, 1, 0)
y_test_pred_dbscan = np.where(test_labels == -1, 1, 0)
# Evaluate performance for training data
print("\033[031m\033[1mTraining Data Performance Metrics:")
print(f"Accuracy : {accuracy_score(y_train,y_train_pred_dbscan):.2f}")
print(f"Precision : {precision_score(y_train, y_train_pred_dbscan):.2f}")
print(f"Recall : {recall_score(y_train, y_train_pred_dbscan):.2f}") 
print(f"F1-Score : {f1_score(y_train, y_train_pred_dbscan):.2f}\n")
print("\033[032m\033[1mTraining Classification Report:\n", classification_report(y_train,y_train_pred_dbscan), '\n')
print("\033[034m\033[1mTraining Confusion Matrix:\n", confusion_matrix(y_train,y_train_pred_dbscan), '\n')
# Evaluate performance for testing data
print("\033[031m\033[1mTesting Data Performance Metrics:")
print(f"Accuracy : {accuracy_score(y_test, y_test_pred_dbscan):.2f}")
print(f"Precision : {precision_score(y_test, y_test_pred_dbscan):.2f}")
print(f"Recall : {recall_score(y_test, y_test_pred_dbscan):.2f}") 
print(f"F1-Score : {f1_score(y_test, y_test_pred_dbscan):.2f}\n")
print("\033[032m\033[1mTesting Classification Report:\n", classification_report(y_test,y_test_pred_dbscan), '\n')
print("\033[034m\033[1mTesting Confusion Matrix:\n", confusion_matrix(y_test,y_test_pred_dbscan), '\n')
# Visualize clustering results for training data
plt.figure(figsize=(10, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_labels, cmap='viridis', s=5)
plt.title("PCA + DBSCAN Clustering Results (Training Data)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster Label')
plt.show()
# Visualize clustering results for testing data
plt.figure(figsize=(10, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_labels, cmap='viridis', s=5)
plt.title("PCA + DBSCAN Clustering Results (Testing Data)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster Label')
plt.show()