
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained ML model
with open("electricity_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    units = float(request.form["units"])
    people = int(request.form["people"])
    appliances = int(request.form["appliances"])
    hours = float(request.form["hours"])

    if (units < 0 or people <= 0 or appliances < 0
            or hours < 0 or hours > 24):
        return "Please enter valid electricity usage details.", 400

    # Prepare input
    input_data = np.array([[
        units, people, appliances, hours
    ]])

    # Predict bill
    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    return render_template(
        "result.html",
        prediction=round(prediction, 2),
        units=units
    )


if __name__ == "__main__":
    app.run(debug=True)
