from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__) #initializes Flask application

# Load the trained model
model = joblib.load("models/linear_regression_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Render the form on the homepage

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from the form using the new names
        area = float(request.form["area"])  # House area in square feet
        bedrooms = float(request.form["bedrooms"])  # Number of bedrooms
        age = float(request.form["age"])  # Age of the house in years

        # Create feature array for prediction
        features = np.array([[area, bedrooms, age]])

        # Make prediction
        prediction = model.predict(features)

        # Render the result page with prediction
        return render_template("result.html", prediction=round(prediction[0], 2)) 
        #extracts the prediction from numpy array to a number, with 2 decimal places

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
