import pandas as pd

class DatasetValidationError(Exception):
    pass


def validate_dataset(df: pd.DataFrame, target_col: str):
    if df.shape[0] < 50:
        raise DatasetValidationError("Dataset must contain at least 50 rows.")

    if target_col not in df.columns:
        raise DatasetValidationError("Target column not found in dataset.")

    y = df[target_col]

    if y.nunique() < 2:
        raise DatasetValidationError("Target column must have at least 2 classes.")

    X = df.drop(columns=[target_col])

    usable_features = []

    for col in X.columns:
        if X[col].isna().mean() > 0.4:
            continue
        if X[col].nunique() <= 1:
            continue
        usable_features.append(col)

    if len(usable_features) < 2:
        raise DatasetValidationError("At least 2 usable feature columns required.")

    return usable_features