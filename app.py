from flask import Flask, render_template, request
import os
import joblib
import pandas as pd

app = Flask(__name__)

# Project folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "placement_model.pkl")

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    MODEL_ERROR = None
except Exception as e:
    model = None
    MODEL_ERROR = str(e)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html",
            error="Model could not be loaded. Check the model file and scikit-learn version."
        )

    try:
        # Read form data and create one student record
        student = pd.DataFrame([{
            "CGPA": float(request.form["cgpa"]),
            "Internships": int(request.form["internships"]),
            "Projects": int(request.form["projects"]),
            "Workshops/Certifications": int(request.form["workshops"]),
            "AptitudeTestScore": float(request.form["aptitude"]),
            "SoftSkillsRating": float(request.form["softskills"]),
            "ExtracurricularActivities": request.form["extracurricular"],
            "PlacementTraining": request.form["training"],
            "SSC_Marks": float(request.form["ssc"]),
            "HSC_Marks": float(request.form["hsc"])
        }])

        # Validate numeric input ranges
        if not 0 <= student["CGPA"].iloc[0] <= 10:
            raise ValueError("CGPA must be between 0 and 10.")

        if not 0 <= student["AptitudeTestScore"].iloc[0] <= 100:
            raise ValueError("Aptitude score must be between 0 and 100.")

        if not 0 <= student["SoftSkillsRating"].iloc[0] <= 5:
            raise ValueError("Soft skills rating must be between 0 and 5.")

        for column in ["SSC_Marks", "HSC_Marks"]:
            if not 0 <= student[column].iloc[0] <= 100:
                raise ValueError("SSC and HSC marks must be between 0 and 100.")

        for column in ["Internships", "Projects", "Workshops/Certifications"]:
            if student[column].iloc[0] < 0:
                raise ValueError("Counts cannot be negative.")

        # Predict placement status
        prediction = model.predict(student)[0]

        # Calculate model score when supported
        probability = None
        try:
            probabilities = model.predict_proba(student)[0]
            classes = list(model.classes_)

            if "Placed" in classes:
                placed_index = classes.index("Placed")
                probability = round(
                    float(probabilities[placed_index]) * 100, 2
                )
        except (AttributeError, ValueError, TypeError):
            probability = None

        # Personalized placement readiness tips
        tips = []

        cgpa = float(student["CGPA"].iloc[0])
        internships = int(student["Internships"].iloc[0])
        projects = int(student["Projects"].iloc[0])
        aptitude = float(student["AptitudeTestScore"].iloc[0])
        soft_skills = float(student["SoftSkillsRating"].iloc[0])
        training = student["PlacementTraining"].iloc[0]

        if cgpa < 7:
            tips.append(
                "Focus on improving your CGPA and core CS concepts."
            )

        if internships == 0:
            tips.append(
                "Look for internships to gain practical experience."
            )

        if projects < 2:
            tips.append(
                "Build at least two strong projects and showcase them on GitHub."
            )

        if aptitude < 60:
            tips.append(
                "Practice quantitative aptitude and logical reasoning regularly."
            )

        if soft_skills < 3:
            tips.append(
                "Work on communication, teamwork, and interview skills."
            )

        if training == "No":
            tips.append(
                "Consider placement training and mock interviews."
            )

        if not tips:
            tips.append(
                "Keep improving your technical skills and practice mock interviews."
            )

        # Custom readiness score (not a probability)
        readiness_score = 0
        readiness_score += min(cgpa / 10 * 25, 25)
        readiness_score += min(internships / 2 * 20, 20)
        readiness_score += min(projects / 3 * 20, 20)
        readiness_score += min(aptitude / 100 * 20, 20)
        readiness_score += min(soft_skills / 5 * 15, 15)
        readiness_score = round(readiness_score)

        return render_template(
            "index.html",
            prediction=prediction,
            probability=probability,
            tips=tips,
            readiness_score=readiness_score
        )

    except Exception as e:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error="Unable to process prediction. Please check the inputs and model compatibility."
        )


if __name__ == "__main__":
    app.run(debug=True)