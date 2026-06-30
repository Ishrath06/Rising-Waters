from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# -----------------------------
# Load trained model & scaler
# -----------------------------
model = joblib.load("floods.save")
scaler = joblib.load("transform.save")


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")


# -----------------------------
# RESULT PAGES
# -----------------------------
@app.route("/chance")
def chance():
    return render_template("chance.html")


@app.route("/no_chance")
def no_chance():
    return render_template("no_chance.html")


# -----------------------------
# PREDICTION ROUTE
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:
            # -----------------------------
            # Collect input features
            # -----------------------------
            features = [
                float(request.form["temp"]),
                float(request.form["humidity"]),
                float(request.form["cloud"]),
                float(request.form["annual"]),
                float(request.form["janfeb"]),
                float(request.form["marmay"]),
                float(request.form["junsep"]),
                float(request.form["octdec"]),
                float(request.form["avgjune"]),
                float(request.form["sub"])
            ]

            # Convert to numpy array
            data = np.array([features])

            # Scale input data
            data = scaler.transform(data)

            # -----------------------------
            # Prediction probability
            # -----------------------------
            prob = model.predict_proba(data)[0][1] * 100

            print(f"🌊 Flood Risk Percentage: {prob:.2f}%")

            # -----------------------------
            # Decision logic
            # -----------------------------
            if prob >= 50:
                return render_template(
                    "chance.html",
                    risk=round(prob, 2)
                )
            else:
                return render_template(
                    "no_chance.html",
                    risk=round(prob, 2)
                )

        except Exception as e:
            print("Error occurred:", e)
            return render_template("index.html")

    return render_template("index.html")


# -----------------------------
# RUN APPLICATION
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)