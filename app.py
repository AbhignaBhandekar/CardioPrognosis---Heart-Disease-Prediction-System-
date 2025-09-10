# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest Classifier model
filename = 'new_heart-disease-prediction-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            print("Input data:", data)  # Debugging line to check input data

            my_prediction = model.predict(data)
            print("Prediction:", my_prediction[0])  # Debugging line to check prediction

            return render_template('result.html', prediction=int(my_prediction[0]))
        except ValueError as e:
            print("ValueError:", e)
            return render_template('error.html', message="Invalid input. Please check the values and try again.")

if __name__ == '__main__':
    app.run(debug=True,port=5011)

