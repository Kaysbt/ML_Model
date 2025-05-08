from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and expected features
model = joblib.load("random_forest_model.pkl")
model_features = joblib.load("model_features.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        # Extract form data
        data = {
            "TimeSpentOnCourse": float(request.form["TimeSpentOnCourse"]),
            "NumberOfVideosWatched": int(request.form["NumberOfVideosWatched"]),
            "NumberOfQuizzesTaken": int(request.form["NumberOfQuizzesTaken"]),
            "QuizScores": float(request.form["QuizScores"]),
            "CompletionRate": float(request.form["CompletionRate"]),
            "DeviceType": 0 if request.form["DeviceType"] == "Desktop" else 1,
        }

        course_category = request.form["CourseCategory"]
        input_dict = {**data, f"CourseCategory_{course_category}": 1}

        # Fill in missing columns
        for col in model_features:
            if col not in input_dict:
                input_dict[col] = 0

        # Prepare DataFrame
        X_input = pd.DataFrame([input_dict])[model_features]

        # Predict
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][1]

        result = {
            "prediction": pred,
            "probability": f"{proba:.2f}"
        }

    return render_template("index.html", result=result)



if __name__ == "__main__":
    app.run(debug=True)
