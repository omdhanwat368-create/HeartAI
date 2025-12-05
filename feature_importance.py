# feature_importance.py
import pandas as pd
import os, joblib

fi_path = os.path.join("models", "feature_importances.csv")
if os.path.exists(fi_path):
    df = pd.read_csv(fi_path)
    print("Top feature importances:")
    print(df.head(20).to_string(index=False))
else:
    print("feature_importances.csv not found at models/ (it may be that the raw model does not expose feature_importances_).")
    # Fallback: attempt to load raw model and inspect
    raw_path = os.path.join("models", "best_raw_model.pkl")
    if os.path.exists(raw_path):
        print("Loaded raw model at", raw_path)
        model = joblib.load(raw_path)
        try:
            pre = model.named_steps["pre"]
            clf = model.named_steps["clf"]
            print("Model pipeline steps:", list(model.named_steps.keys()))
            if hasattr(clf, "feature_importances_"):
                import numpy as np
                feat_names = []
                try:
                    num_feats = pre.transformers_[0][2]
                    ohe = pre.transformers_[1][1].named_steps["onehot"]
                    cat_cols = pre.transformers_[1][2]
                    cat_names = list(ohe.get_feature_names_out(cat_cols))
                    feat_names = list(num_feats) + cat_names
                except Exception:
                    feat_names = [f"f{i}" for i in range(len(clf.feature_importances_))]
                fi = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_})
                fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
                print(fi.head(30).to_string(index=False))
                fi.to_csv("models/feature_importances_from_raw.csv", index=False)
                print("Saved models/feature_importances_from_raw.csv")
            else:
                print("Raw classifier does not have feature_importances_")
        except Exception as ex:
            print("Could not introspect pipeline:", ex)
    else:
        print("No raw model found to inspect.")
