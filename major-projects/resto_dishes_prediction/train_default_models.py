import os
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

from dish_data import generate_themed_menu_datasets


# -----------------------------
# CONFIG
# -----------------------------
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "performance_tier"
RANDOM_STATE = 42


# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_and_save_model(df, model_name):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(str)

    class_counts = Counter(y)
    min_class_count = min(class_counts.values())

    # Decide whether stratified split is possible
    stratify_arg = y if min_class_count >= 2 else None

    if stratify_arg is None:
        print(
            f"⚠️ Warning: {model_name} has a class with <2 samples. "
            "Proceeding without stratified split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_arg
    )

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # 🔥 CORRECT preprocessing with scaling
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1
            ))
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy of {model_name}: {acc:.4f}")
    print("Class distribution:", dict(class_counts))

    model_path = os.path.join(MODEL_DIR, model_name)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"💾 Saved model → {model_path}\n")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("Generating themed datasets...\n")

    indian_df, japanese_df, italian_df = generate_themed_menu_datasets(
        n_rows_per_cuisine=400
    )

    print("Training Indian Logistic Regression model...")
    train_and_save_model(indian_df, "indian_lr.pkl")

    print("Training Japanese Logistic Regression model...")
    train_and_save_model(japanese_df, "japanese_lr.pkl")

    print("Training Italian Logistic Regression model...")
    train_and_save_model(italian_df, "italian_lr.pkl")

    print("🎉 All Logistic Regression models trained and saved successfully.")