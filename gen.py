import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 100

# Generate random values for each feature
data = {
    'age': np.random.randint(29, 77, num_samples),
    'sex': np.random.randint(0, 2, num_samples),
    'cp': np.random.randint(0, 4, num_samples),
    'trestbps': np.random.randint(94, 201, num_samples),
    'chol': np.random.randint(126, 565, num_samples),
    'fbs': np.random.randint(0, 2, num_samples),
    'restecg': np.random.randint(0, 3, num_samples),
    'thalach': np.random.randint(71, 203, num_samples),
    'exang': np.random.randint(0, 2, num_samples),
    'oldpeak': np.random.uniform(0, 6.2, num_samples),
    'slope': np.random.randint(0, 3, num_samples),
    'ca': np.random.randint(0, 5, num_samples),
    'thal': np.random.randint(0, 4, num_samples),
    'condition': np.random.randint(0, 2, num_samples)  # Class label: 0 or 1
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv('heart_cleveland_upload.csv', index=False)

