# Heart Disease Prediction — Internship Project

This project predicts the presence of heart disease using machine learning
(Logistic Regression, Decision Tree, Random Forest) and serves an interactive
Flask web app with:

- Email-based login/registration
- SQLite-based prediction history
- Modern UI with Bootstrap and Chart.js

## Quick Start

1. Put the dataset at `data/heart.csv`  
   (must contain a `target` column and features like: age, sex, cp, trestbps, chol, fbs,
   restecg, thalach, exang, oldpeak, slope, ca, thal).

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
# Windows:
# venv\Scripts\Activate.ps1
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
