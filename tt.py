import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset from CSV
data = pd.read_csv("heart_cleveland_upload.csv")

# Extract features (X) and target variable (y)
X = data.iloc[:, :13]  # Assuming your features are in the first 13 columns
y = data.iloc[:, 13]   # Assuming your target variable is in the 14th column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save the model to a binary file
filename = 'new_heart-disease-prediction-model.pkl'
pickle.dump(model, open(filename, 'wb'))

