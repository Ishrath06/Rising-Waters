from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("floods.save")
scaler = joblib.load("transform.save")


@app.route("/")
def home():
    return render_template("home.html")


# Route for Flood Predicted page
@app.route("/chance")
def chance():
    return render_template("chance.html")


# Route for No Flood page
@app.route("/no_chance")
def no_chance():
    return render_template("no_chance.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        data = [[
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
        ]]

        data = scaler.transform(np.array(data))

        result = model.predict(data)

        if result[0] == 1:
            return redirect(url_for("chance"))
        else:
            return redirect(url_for("no_chance"))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)