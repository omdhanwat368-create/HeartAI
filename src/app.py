# src/app.py
import os
import json
import sqlite3
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import difflib

# Paths & app
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DB_PATH = os.path.join(BASE_DIR, "app.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")


# ---------------------- DB INIT ----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        input_json TEXT,
        prediction INTEGER,
        probability REAL,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


# ---------------------- LOAD MODEL ----------------------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Loaded model:", MODEL_PATH)
    except Exception as e:
        print("Failed to load model:", e)
else:
    print("Model file not found:", MODEL_PATH)


# ---------------------- HELPERS ----------------------
def save_history(user_email, input_data, prediction, probability):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO history (user_email, input_json, prediction, probability, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_email, json.dumps(input_data), int(prediction), float(probability), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


# ---------------------- ROUTES ----------------------

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/dashboard")
def index():
    if "user" in session:
        return render_template("index.html", user=session.get("user"))
    return redirect(url_for("login"))


# FIX for BuildError: register dashboard endpoint alias
try:
    if "dashboard" not in app.view_functions:
        app.add_url_rule("/dashboard", endpoint="dashboard", view_func=index)
except:
    pass


# ---------------------- AUTH ----------------------
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Provide both email and password", "warning")
            return redirect(url_for("register"))

        pw_hash = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                        (email, pw_hash, datetime.utcnow().isoformat()))
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError:
            flash("Email already registered", "danger")
            return redirect(url_for("register"))

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            session["user"] = email
            flash("Logged in successfully", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out", "info")
    return redirect(url_for("landing"))


# ---------------------- PREDICT UI PAGE ----------------------
@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html", user=session.get("user"))


# ---------------------- PREDICT API ----------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    global model
    if model is None:
        return jsonify({"error": "Model not available. Train first."}), 503

    data = request.get_json() if request.is_json else request.form.to_dict()

    expected = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
                'exang','oldpeak','slope','ca','thal']

    row = []
    parsed = {}

    for field in expected:
        if field not in data:
            return jsonify({"error": f"Missing {field}"}), 400

        try:
            val = float(data[field])
        except:
            val = 0.0

        parsed[field] = val
        row.append(val)

    X = pd.DataFrame([row], columns=expected)
    proba = float(model.predict_proba(X)[0][1])

    THRESH = float(os.environ.get("RISK_THRESHOLD", 0.5))
    pred = int(proba >= THRESH)

    # Save history (no crash if error)
    try:
        save_history(session.get("user"), parsed, pred, proba)
    except Exception as e:
        print("History save failed:", e)

    return jsonify({
        "prediction": pred,
        "probability": proba,
        "threshold": THRESH,
        "risk": "high" if pred == 1 else "low"
    })


# BACKWARD COMPATIBILITY: allow POST /predict to also work
try:
    if "predict_post_alias" not in app.view_functions:
        app.add_url_rule("/predict", endpoint="predict_post_alias",
                         view_func=api_predict, methods=["POST"])
except:
    pass


# ---------------------- HISTORY ----------------------
@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    email = session.get("user")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, input_json, prediction, probability, created_at FROM history WHERE user_email=? ORDER BY id DESC", (email,))
    rows = cur.fetchall()
    conn.close()

    entries = []
    for r in rows:
        entries.append({
            "id": r[0],
            "input": json.loads(r[1]),
            "prediction": r[2],
            "probability": r[3],
            "created_at": r[4]
        })

    return render_template("history.html", entries=entries, user=email)


# ---------------------- METRICS ----------------------
@app.route("/api/metrics")
def api_metrics():
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return jsonify(json.load(f))

    return jsonify({"error": "metrics not found"}), 404


# ---------------------- PRINT ROUTES ----------------------
def print_routes():
    print("\nRegistered Routes:")
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        print(f"{rule.rule}  --> {rule.endpoint}   [{','.join(rule.methods)}]")
    print("")

print_routes()

# --- Chat assistant: simple local rule-based + retrieval responder ---


# small knowledge base (expandable)
_CHAT_KB = [
    {
        "id": "causes_heart_disease",
        "title": "Common causes of heart disease",
        "keywords": ["cause", "causes", "why", "risk factors", "risk"],
        "content": [
            "High blood pressure (hypertension)",
            "High cholesterol",
            "Smoking",
            "Diabetes",
            "Obesity and physical inactivity",
            "Unhealthy diet",
            "Family history and age"
        ],
        "summary": "Heart disease is usually due to multiple risk factors — controlling blood pressure, cholesterol, and lifestyle can reduce risk."
    },
    {
        "id": "prevention_tips",
        "title": "Prevention & lifestyle tips",
        "keywords": ["prevent", "prevention", "tips", "how to reduce risk", "lifestyle"],
        "content": [
            "Adopt a heart-healthy diet (more fruits, vegetables, whole grains)",
            "Exercise regularly — aim for 150 minutes/week moderate activity",
            "Stop smoking and avoid secondhand smoke",
            "Manage blood pressure and cholesterol with medication when prescribed",
            "Maintain a healthy weight",
            "Limit alcohol and reduce stress"
        ],
        "summary": "Small, consistent lifestyle changes can meaningfully lower heart risk over time."
    },
    {
        "id": "symptoms_when_to_see_doctor",
        "title": "When to see a doctor",
        "keywords": ["when to see", "doctor", "symptom", "symptoms", "visit", "checkup"],
        "content": [
            "New or worsening chest pain or pressure",
            "Shortness of breath at rest or with minimal exertion",
            "Fainting or near-fainting episodes",
            "Sudden severe weakness or numbness on one side (possible stroke)",
            "Palpitations with dizziness"
        ],
        "summary": "If you have worrying symptoms — get medical attention quickly. For sudden severe symptoms, call emergency services."
    },
    {
        "id": "tests_for_heart",
        "title": "Common tests for heart evaluation",
        "keywords": ["test", "tests", "ecg", "ekg", "stress test", "blood test"],
        "content": [
            "Electrocardiogram (ECG/EKG)",
            "Echocardiogram (heart ultrasound)",
            "Exercise stress test or pharmacologic stress test",
            "Cardiac biomarkers (troponin) and blood tests",
            "Coronary angiography (invasive) when needed"
        ],
        "summary": "Doctors use a combination of tests based on symptoms and risk to evaluate heart health."
    },
    {
        "id": "oldpeak_explain",
        "title": "What is Oldpeak?",
        "keywords": ["oldpeak", "old peak", "st depression"],
        "content": [
            "Oldpeak measures ST-segment depression during exercise relative to rest and can reflect ischemia.",
            "Higher oldpeak values often indicate worse exercise-induced ischemia; clinical interpretation requires context."
        ],
        "summary": "Oldpeak is a numeric measure used in cardiac datasets — ask your clinician to interpret with other tests."
    },
    {
        "id": "read_more",
        "title": "Further reading & resources",
        "keywords": ["more", "resources", "learn", "read"],
        "content": [
            "World Health Organization: cardiovascular diseases",
            "American Heart Association patient resources",
            "Local public health heart disease prevention programs"
        ],
        "summary": "Trusted public health sites and cardiology societies provide patient guides and action steps."
    }
]

# emergency triggers (very conservative)
_EMERGENCY_KEYWORDS = [
    "chest pain", "severe chest", "pressure in chest", "shortness of breath",
    "faint", "passing out", "collapse", "unconscious", "sudden weakness", "numbness",
    "slurred speech", "difficulty breathing", "heavy chest"
]

def _is_emergency(text):
    if not text: return False
    t = text.lower()
    for kw in _EMERGENCY_KEYWORDS:
        if kw in t:
            return True
    return False

def _match_kb(text):
    """Return best KB item by simple token overlap + difflib similarity"""
    if not text:
        return None
    txt = text.lower()
    # exact keyword hits
    for item in _CHAT_KB:
        for kw in item.get("keywords", []):
            if kw in txt:
                return item
    # fuzzy match against titles
    titles = [it["title"] for it in _CHAT_KB]
    close = difflib.get_close_matches(text, titles, n=1, cutoff=0.55)
    if close:
        for it in _CHAT_KB:
            if it["title"] == close[0]:
                return it
    # fallback: try partial word overlap across content and keywords
    words = [w for w in txt.split() if len(w) > 2]
    best = None; best_score = 0
    for it in _CHAT_KB:
        pool = " ".join(it.get("keywords", []) + [it.get("title","")] + it.get("content",[])).lower()
        score = sum(1 for w in words if w in pool)
        if score > best_score:
            best_score = score; best = it
    if best_score > 0:
        return best
    return None

@app.route("/chat")
def chat_page():
    # simple page rendered by template
    return render_template("chat.html", user=session.get("user"))

@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json() or {}
    text = (payload.get("message") or "").strip()
    # emergency short-circuit
    if _is_emergency(text):
        return jsonify({
            "type": "emergency",
            "text": "If you are experiencing chest pain, difficulty breathing, fainting, or sudden severe symptoms — call your local emergency number immediately (e.g. 112/911) or go to the nearest emergency department. This system cannot replace emergency care.",
            "quick_replies": ["Call emergency services", "Find nearest hospital"]
        }), 200

    # small canned replies for greetings and simple asks
    greetings = ["hi","hello","hey","good morning","good afternoon","good evening"]
    if text.lower() in greetings or text.lower().startswith("hi") or text.lower().startswith("hello"):
        return jsonify({
            "type": "text",
            "text": "Hi — I'm HeartAssist. I can explain heart risk factors, give prevention tips, and suggest when to see a doctor. Try asking: 'What causes heart disease?', 'How to reduce my risk?', or 'When should I see a doctor?'",
            "quick_replies": ["Causes", "Prevention tips", "When to see a doctor"]
        })

    # match KB
    item = _match_kb(text)
    if item:
        # structure a helpful card
        return jsonify({
            "type": "card",
            "title": item["title"],
            "summary": item.get("summary"),
            "bullets": item.get("content", []),
            "text": item.get("summary"),
            "quick_replies": ["More on tests", "Lifestyle tips", "When to see a doctor"]
        })

    # fallback: simple heuristic answer
    # try to detect numeric questions or ask for clarification
    if any(ch.isdigit() for ch in text) and ("risk" in text or "probab" in text):
        return jsonify({
            "type": "text",
            "text": "If you want to interpret a numeric risk or probability, please provide the probability value (e.g. 'my probability is 0.72') or use the Predict tool on the site. For interpretation, share the percent and I'll explain.",
            "quick_replies": ["How to interpret probability", "Use Predict tool"]
        })

    # default fallback safe response
    return jsonify({
        "type": "text",
        "text": "Sorry — I didn't understand that exactly. Try asking about causes, prevention tips, typical tests, or when to seek medical help.",
        "quick_replies": ["Causes", "Prevention tips", "When to see a doctor"]
    })

# ---------------------- RUN ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
