from flask import Flask, render_template, request
import pandas as pd
import joblib
import tensorflow as tf
import json

app = Flask(__name__)

# Load models
scaler = joblib.load("scaler.pkl")
svm_model = joblib.load("svm_model.pkl")
ann_model = tf.keras.models.load_model("ann_model.h5")

# Load accuracies
with open("model_accuracies.json", "r") as f:
    model_accuracies = json.load(f)

# Username features
def extract_username_features(username):
    username = str(username)
    length = len(username)
    letters = sum(c.isalpha() for c in username)
    digits = sum(c.isdigit() for c in username)
    has_name = 1 if letters > digits else 0
    return length, has_name


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form["username"]
        followers = float(request.form["followers"])
        following = float(request.form["following"])
        posts = float(request.form["posts"])
        bio_length = float(request.form["bio_length"])
        has_profile_pic = int(request.form["has_profile_pic"])
        external_url = int(request.form["external_url"])
        account_age_days = float(request.form["account_age_days"])
        activity_score = float(request.form["activity_score"])

        # Extract username features
        length, has_name = extract_username_features(username)

        row = pd.DataFrame([{
            "username_length": length,
            "has_name": has_name,
            "followers": followers,
            "following": following,
            "posts": posts,
            "bio_length": bio_length,
            "account_age_days": account_age_days,
            "activity_score": activity_score,
            "has_profile_pic": has_profile_pic,
            "external_url": external_url
        }])

        # Scale data
        scaled = scaler.transform(row)

        # Predictions
        ann_pred = int((ann_model.predict(scaled)[0][0] > 0.5))
        svm_pred = int(svm_model.predict(scaled)[0])

        # Final decision (if any model says FAKE → FAKE)
        votes = [ann_pred, svm_pred]
        final_pred = 1 if sum(votes) >= 1 else 0

        # Output text
        if final_pred == 1:
            result = "FAKE ❌"
        else:
            result = "REAL ✅"

        return render_template(
            "result.html",
            result=result,
            username=username,
            model_accuracies=model_accuracies
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
