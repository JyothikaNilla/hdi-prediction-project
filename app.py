from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    life = float(request.form["life"])
    mean = float(request.form["mean"])
    expected = float(request.form["expected"])
    gni = float(request.form["gni"])

    features = np.array([[life, mean, expected, gni]])

    prediction = model.predict(features)[0]

    prediction = round(float(prediction), 3)

    if prediction >= 0.800:
        category = "🌟 Very High Human Development"
        color = "#2ecc71"

    elif prediction >= 0.700:
        category = "🟢 High Human Development"
        color = "#27ae60"

    elif prediction >= 0.550:
        category = "🟡 Medium Human Development"
        color = "#f1c40f"

    else:
        category = "🔴 Low Human Development"
        color = "#e74c3c"

    return render_template(
        "result.html",
        prediction=prediction,
        category=category,
        color=color,
        life=life,
        mean=mean,
        expected=expected,
        gni=gni
    )


if __name__ == "__main__":
    app.run(debug=True)