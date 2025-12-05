# src/train.py
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Make sure local src is importable when running from project root
this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
    sys.path.append(this_dir)

from model_utils import load_data, preprocess

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    print(f"--- {name} ---")
    print(classification_report(y_test, preds, digits=4))
    if probs is not None:
        print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if probs is not None else None
    }

def get_feature_names_from_preprocessor(preprocessor):
    """
    Returns the transformed feature names after ColumnTransformer + OneHotEncoder.
    Works with sklearn >= 1.0 OneHotEncoder.get_feature_names_out
    """
    feature_names = []
    # preprocessor.transformers_ is a list of (name, transformer, columns)
    for name, trans, cols in preprocessor.transformers_:
        if trans is None:
            # passthrough (rare here)
            if hasattr(cols, "__iter__"):
                feature_names.extend(list(cols))
            else:
                feature_names.append(cols)
            continue
        # numeric pipeline
        try:
            # If transformer is a Pipeline
            if hasattr(trans, 'named_steps') and 'onehot' in trans.named_steps:
                # categorical pipeline with OneHotEncoder
                ohe = trans.named_steps['onehot']
                # get_feature_names_out requires the original feature names
                ohe_names = list(ohe.get_feature_names_out(cols))
                feature_names.extend(ohe_names)
            else:
                # numeric transformer - keep original column names
                if hasattr(cols, "__iter__"):
                    feature_names.extend(list(cols))
                else:
                    feature_names.append(cols)
        except Exception:
            # fallback: append cols names
            if hasattr(cols, "__iter__"):
                feature_names.extend(list(cols))
            else:
                feature_names.append(cols)
    return feature_names

def main(data_path="data/heart.csv"):
    print("Loading data from", data_path)
    df = load_data(data_path)
    X, y, preprocessor = preprocess(df)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = {
        "logreg": Pipeline(steps=[("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000))]),
        "dt": Pipeline(steps=[("pre", preprocessor), ("clf", DecisionTreeClassifier(random_state=42))]),
        "rf": Pipeline(steps=[("pre", preprocessor), ("clf", RandomForestClassifier(random_state=42))]),
    }

    param_grids = {
        "logreg": {"clf__C": [0.1, 1, 10]},
        "dt": {"clf__max_depth": [3, 5, None]},
        "rf": {"clf__n_estimators": [50, 100], "clf__max_depth": [5, None]}
    }

    results = {}
    best_models = {}

    for name, pipe in pipelines.items():
        print(f"\nTraining {name} ...")
        gs = GridSearchCV(pipe, param_grid=param_grids[name], cv=5, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"{name} best params:", gs.best_params_)
        best_models[name] = gs.best_estimator_
        results[name] = evaluate_model(name, gs.best_estimator_, X_test, y_test)

    # choose best by ROC-AUC on test
    best_name = None
    best_score = -1
    for name, model in best_models.items():
        try:
            probs = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, probs)
        except Exception:
            score = -1
        print(f"{name} ROC-AUC = {score:.4f}")
        if score > best_score:
            best_score = score
            best_name = name

    print("\nBest model:", best_name, "score:", best_score)

    out_dir = os.path.join(this_dir, "..", "models")
    os.makedirs(out_dir, exist_ok=True)

    # Save raw best (un-calibrated) model for feature importances & inspection
    raw_save_path = os.path.join(out_dir, "best_raw_model.pkl")
    print("Saving raw best model to", raw_save_path)
    joblib.dump(best_models[best_name], raw_save_path)

    # Calibrate the best model using Platt scaling (sigmoid). Wrap the pipeline.
    print("Calibrating best model with CalibratedClassifierCV (cv=5)...")
    calibrator = CalibratedClassifierCV(best_models[best_name], cv=5, method='sigmoid')
    # fit calibrator on training data (pipeline handles preprocessing)
    calibrator.fit(X_train, y_train)
    calibrated_save_path = os.path.join(out_dir, "best_model.pkl")
    joblib.dump(calibrator, calibrated_save_path)
    print("Saved calibrated model to", calibrated_save_path)

    # recompute test metrics using calibrated model
    probs_cal = calibrator.predict_proba(X_test)[:, 1]
    preds_cal = (probs_cal >= 0.5).astype(int)
    results_cal = {
        "calibrated_test_auc": float(roc_auc_score(y_test, probs_cal)),
        "calibrated_test_accuracy": float(accuracy_score(y_test, preds_cal))
    }
    print("Calibrated ROC-AUC:", results_cal["calibrated_test_auc"])
    print("Calibrated Accuracy:", results_cal["calibrated_test_accuracy"])

    # Save metrics (include original results and calibrated metrics)
    metrics = {
        "models": results,
        "best_model": best_name,
        "best_score_test_roc_auc": float(best_score),
        **results_cal
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to", metrics_path)

    # Attempt to extract feature importances from the raw classifier
    try:
        # raw_model is a pipeline: pre -> clf
        raw_model = best_models[best_name]
        clf = raw_model.named_steps.get("clf", None)
        if clf is None:
            # If not a pipeline, try attribute access
            clf = getattr(raw_model, "clf", None)

        if hasattr(clf, "feature_importances_"):
            # Build feature names from preprocessor
            pre = raw_model.named_steps["pre"]
            feat_names = get_feature_names_from_preprocessor(pre)
            importances = clf.feature_importances_
            if len(importances) == len(feat_names):
                fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
            else:
                # fallback - match by index
                fi_df = pd.DataFrame({"feature": [f"f{i}" for i in range(len(importances))], "importance": importances})
            fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
            fi_out = os.path.join(out_dir, "feature_importances.csv")
            fi_df.to_csv(fi_out, index=False)
            print("Saved feature importances to", fi_out)
        else:
            print("Classifier does not expose feature_importances_. Skipping feature importance export.")
    except Exception as e:
        print("Error extracting feature importances:", e)

if __name__ == "__main__":
    main()
