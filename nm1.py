# Load your dataset
import pandas as pd

# Assuming your dataset is stored in a CSV file named 'heart_cleveland_upload.csv'
data = pd.read_csv('heart_cleveland_upload.csv')

# Extract features (X) and target variable (y)
X = data.drop(columns=['target'])  # Assuming 'target' is the name of the target variable
y = data['target']

# Now you can proceed to split the data into training and testing sets and train your model

