import numpy as np
import pandas as pd
import pickle

# Load the model
model = pickle.load(open('new_heart-disease-prediction-model.pkl', 'rb'))

# Load the test data
test_data = pd.read_csv('sample_data_with_condition.csv')

# Extract features from the test data (excluding the 'condition' column)
X_test = test_data.drop(columns=['condition'])

# Make predictions
predictions = model.predict(X_test)

print(predictions)

