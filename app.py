from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("floods.save")
scaler = joblib.load("transform.save")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/chance")
def chance():
    return render_template("chance.html")


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

        # scale input
        data = scaler.transform(np.array(data))

        # 🔥 GET PROBABILITY (MAIN FIX)
        prob = model.predict_proba(data)[0][1] * 100

        print("Flood Risk %:", prob)

        # decision based on probability
        if prob >= 50:
            return render_template("chance.html", risk=round(prob, 2))
        else:
            return render_template("no_chance.html", risk=round(prob, 2))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)