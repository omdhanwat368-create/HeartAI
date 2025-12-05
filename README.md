# Heart Disease Prediction — Internship Project

This project predicts the presence of heart disease using machine learning
(Logistic Regression, Decision Tree, Random Forest) and serves an interactive
Flask web app with:

- Email-based login/registration
- SQLite-based prediction history
- Modern UI with Bootstrap and Chart.js

  
<img width="1920" height="1080" alt="Screenshot (50)" src="https://github.com/user-attachments/assets/978cd3d3-bd15-453b-8882-e7a33c09768e" />
<img width="1920" height="1080" alt="Screenshot (51)" src="https://github.com/user-attachments/assets/ef3434d3-24a2-49d2-b102-435051fa82a1" />
<img width="1920" height="1080" alt="Screenshot (52)" src="https://github.com/user-attachments/assets/78bc436b-2303-4bc5-bd4b-a4ac78817c29" />
<img width="1920" height="1080" alt="Screenshot (53)" src="https://github.com/user-attachments/assets/3e9286ae-ce0f-45da-8c06-0146131684aa" />


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
