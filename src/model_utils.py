import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(path="data/heart.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()
    df = df.drop_duplicates()

    if 'target' not in df.columns:
        raise ValueError('Dataset must contain "target" column.')

    # numeric
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_features:
        numeric_features.remove('target')

    # Categorical columns common in heart datasets:
    possible_cat = ['cp', 'restecg', 'slope', 'ca', 'thal']
    categorical_features = [c for c in possible_cat if c in df.columns]

    numeric_features = [c for c in numeric_features if c not in categorical_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X = df.drop(columns=["target"])
    y = df["target"].values

    return X, y, preprocessor
