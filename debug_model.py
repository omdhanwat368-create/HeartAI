import joblib, pandas as pd, json, os
from sklearn.metrics import accuracy_score

MODEL = "models/best_model.pkl"
DATA  = "data/heart.csv"

print("Model exists:", os.path.exists(MODEL))
print("Data exists:", os.path.exists(DATA))

m = joblib.load(MODEL)
print("Model type:", type(m))
try:
    print("Pipeline steps:", getattr(m, "named_steps", "no named_steps"))
except Exception as e:
    print("named_steps error:", e)

df = pd.read_csv(DATA)
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
if "target" in df.columns:
    print("Target counts:", df["target"].value_counts().to_dict())

# Low-risk example to test:
low = {"age":35,"sex":0,"cp":0,"trestbps":120,"chol":180,"fbs":0,
       "restecg":0,"thalach":165,"exang":0,"oldpeak":0.0,"slope":1,"ca":0,"thal":2}
Xlow = pd.DataFrame([low])
print("\nLow-risk example:", low)
try:
    p = m.predict_proba(Xlow)[0][1]
    print("predict_proba for class=1:", p)
    print("prediction (threshold 0.5):", int(p>=0.5))
except Exception as e:
    print("Error running predict_proba on example:", e)

# Evaluate on all rows
if "target" in df.columns:
    X = df.drop(columns=["target"])
    y = df["target"].values
    probs = m.predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)
    print("\nAverage prob for target=0 rows:", float(probs[y==0].mean()))
    print("Average prob for target=1 rows:", float(probs[y==1].mean()))
    print("Accuracy (threshold 0.5):", accuracy_score(y, preds))
    print("Accuracy if we invert predictions:", accuracy_score(y, 1-preds))
    # show few rows with low prob and high prob
    import numpy as np
    idx_low = np.where(probs < 0.2)[0][:5]
    idx_high = np.where(probs > 0.8)[0][:5]
    print("\nSample low-prob rows indexes and probs:", list(zip(idx_low.tolist(), probs[idx_low].tolist())))
    print("Sample high-prob rows indexes and probs:", list(zip(idx_high.tolist(), probs[idx_high].tolist())))
else:
    print("No target column to compare against.")
