# AI Placement Intelligence System

An AI-based web application that predicts a student's placement status using academic performance, technical preparation, and skill-related information.

## Project Overview

The AI Placement Intelligence System uses Machine Learning to estimate whether a student is likely to be placed based on their profile.

It also provides:
- Placement prediction
- Model confidence score
- Career readiness score
- Personalized improvement suggestions

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- CSS
- JavaScript
- Joblib
- Jupyter Notebook

## Machine Learning Models

Two classification algorithms were trained and compared:

1. Logistic Regression
2. Random Forest Classifier

**Selected Model:** Logistic Regression

Test accuracy: **79.19%**

## Dataset

The project uses the Campus Recruitment dataset available on Hugging Face.

- Original records: 7,225
- Duplicate records removed: 43
- Final records: 7,182
- Missing values: None

## Input Features

The model uses the following student information:

- CGPA
- Internships
- Projects
- Workshops/Certifications
- Aptitude Test Score
- Soft Skills Rating
- Extracurricular Activities
- Placement Training
- SSC Marks
- HSC Marks

## Project Structure

```text
LIVE/
│
├── app.py
├── data/
│   └── train.csv
│
├── models/
│   └── placement_model.pkl
│
├── notebooks/
│   └── 01_Placement_Analysis.ipynb
│
├── static/
│
├── templates/
│   └── index.html
│
└── README.md


