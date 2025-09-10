# Importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading and reading the dataset
heart = pd.read_csv("heart_cleveland_upload.csv")

# Creating a copy of the dataset to avoid affecting the original dataset
heart_df = heart.copy()

# Renaming some of the columns
heart_df = heart_df.rename(columns={'condition': 'target'})
print(heart_df.head())

# Model building
# Splitting data into X and y. Here y contains target data and X contains all other features.
X = heart_df.drop(columns='target')
y = heart_df['target']

# Splitting our dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating a Random Forest Classifier
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train_scaled, y_train)

# Making predictions
y_pred = model.predict(X_test_scaled)

# Evaluating the model
accuracy = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

print('Classification Report\n', classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n')

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Creating a pickle file for the classifier
filename = 'new_heart-disease-prediction-model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

# Saving the scaler to apply the same transformation in the Flask app
scaler_filename = 'scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)

