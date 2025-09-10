import pickle

# Load the trained model
with open('heart-disease-prediction-knn-model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the filename for the new binary file
filename = 'new_heart-disease-prediction-model.pkl'

# Save the model to the new binary file
with open(filename, 'wb') as file:
    pickle.dump(model, file)

