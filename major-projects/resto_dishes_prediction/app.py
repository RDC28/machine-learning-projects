# app.py (Scaled Logistic Regression version)

from flask import Flask, render_template, redirect, url_for, session, request
import pickle
import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # <--- Added StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

# local validator
from modules.data_validator import validate_dataset, DatasetValidationError

app = Flask(__name__)
app.secret_key = "dev_secret_key"

MODEL_DIR = "trained_models"
DATASET_DIR = "dish_datasets"
DEFAULT_TARGET_COL = "performance_tier"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)


# -----------------------------
# UTILITIES
# -----------------------------
def load_model(model_name):
    path = os.path.join(MODEL_DIR, model_name)
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "pipeline" in obj:
        return obj
    else:
        return {
            "pipeline": obj,
            "feature_columns": None,
            "target_col": DEFAULT_TARGET_COL
        }


def load_dataset(dataset_name):
    return pd.read_csv(os.path.join(DATASET_DIR, dataset_name))


def get_feature_columns(df, target_col=DEFAULT_TARGET_COL):
    return [c for c in df.columns if c != target_col]


def form_to_dataframe_safe(form, expected_features):
    row = {}
    for col in expected_features:
        raw = form.get(col)

        if raw is None or raw == "":
            row[col] = np.nan
            continue

        coerced = pd.to_numeric(raw, errors="coerce")
        if not np.isnan(coerced):
            row[col] = int(coerced) if coerced == int(coerced) else float(coerced)
        else:
            row[col] = raw

    return pd.DataFrame([row])


# -----------------------------
# MODEL TRAINING (LOGISTIC)
# -----------------------------
def train_custom_model(df: pd.DataFrame, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)

    class_counts = Counter(y)
    stratify_arg = y if min(class_counts.values()) >= 2 else None

    if stratify_arg is None:
        print("⚠️ Warning: class with <2 samples; no stratification.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    # --- UPDATED PREPROCESSOR ---
    preprocessor = ColumnTransformer(
        transformers=[
            # Numeric: Impute missing -> Scale
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())  # <--- SCALING ADDED HERE
            ]), numeric_cols),
            
            # Categorical: Impute missing -> OneHotEncode
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_cols)
        ],
        remainder="drop"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1
            ))
        ]
    )

    pipeline.fit(X_train, y_train)

    try:
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        print(f"✅ Custom Logistic Regression (Scaled) accuracy: {acc:.4f}")
        print("Class distribution:", dict(class_counts))
    except Exception as e:
        print("⚠️ Evaluation failed:", e)

    return {
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "target_col": target_col
    }


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/use_dataset/<dataset_name>")
def use_dataset(dataset_name):
    dataset_map = {
        "indian": ("indian_lr.pkl", "indian_dishes.csv"),
        "japanese": ("japanese_lr.pkl", "japanese_dishes.csv"),
        "italian": ("italian_lr.pkl", "italian_dishes.csv"),
    }

    if dataset_name not in dataset_map:
        return redirect(url_for("index"))

    model_file, dataset_file = dataset_map[dataset_name]

    session["model_name"] = model_file
    session["dataset_file"] = dataset_file
    session["dataset_type"] = dataset_name
    session.pop("target_col", None)

    return redirect(url_for("predict"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model_name = session.get("model_name")
    dataset_file = session.get("dataset_file")

    if not model_name or not dataset_file:
        return redirect(url_for("index"))

    df = load_dataset(dataset_file)
    feature_columns = get_feature_columns(df, session.get("target_col", DEFAULT_TARGET_COL))

    if request.method == "POST":
        model_obj = load_model(model_name)
        pipeline = model_obj["pipeline"]
        expected_features = model_obj.get("feature_columns") or feature_columns

        input_df = form_to_dataframe_safe(request.form, expected_features)

        try:
            # The pipeline automatically scales the input_df here because
            # 'scaler' is part of the saved pipeline structure.
            pred = pipeline.predict(input_df)[0]
            proba_str = None

            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(input_df)[0]
                pairs = sorted(
                    zip(pipeline.classes_, probs),
                    key=lambda x: x[1],
                    reverse=True
                )
                proba_str = ", ".join(f"{c}:{p:.3f}" for c, p in pairs)

            return redirect(url_for("result", prediction=pred, proba=proba_str))

        except Exception as e:
            return f"Prediction error: {e}", 500

    return render_template(
        "predict.html",
        feature_columns=feature_columns,
        dataset_name=session.get("dataset_type")
    )


@app.route("/result")
def result():
    prediction = request.args.get("prediction")
    proba = request.args.get("proba")

    if not prediction:
        return redirect(url_for("index"))

    return render_template("result.html", prediction=prediction, proba=proba)


@app.route("/custom_data", methods=["GET", "POST"])
def custom_data():
    if request.method == "POST":
        file = request.files.get("dataset")
        target_col = request.form.get("target_column")

        if not file or not file.filename.endswith(".csv"):
            return "Only CSV files allowed", 400

        save_path = os.path.join(DATASET_DIR, "custom_dishes.csv")
        file.save(save_path)

        df = pd.read_csv(save_path)

        try:
            usable_features = validate_dataset(df, target_col)
        except DatasetValidationError as e:
            return f"Dataset error: {str(e)}", 400

        df_trimmed = df[usable_features + [target_col]]
        
        # This will now scale the data during training
        metadata = train_custom_model(df_trimmed, target_col)

        model_path = os.path.join(MODEL_DIR, "custom_lr.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(metadata, f)

        session["model_name"] = "custom_lr.pkl"
        session["dataset_file"] = "custom_dishes.csv"
        session["dataset_type"] = "custom"
        session["target_col"] = target_col

        return redirect(url_for("predict"))

    return render_template("custom_data.html")


if __name__ == "__main__":
    app.run(debug=True)