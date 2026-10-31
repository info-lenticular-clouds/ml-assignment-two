from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"

BANDS = ("green", "red", "nir1", "nir2")
PIXELS = (
    "top_left",
    "top_center",
    "top_right",
    "mid_left",
    "center",
    "mid_right",
    "bot_left",
    "bot_center",
    "bot_right",
)
FEATURE_COLUMNS = [f"{pixel}_{band}" for pixel in PIXELS for band in BANDS]
TARGET_COLUMN = "land_cover"

CLASS_MAP = {
    1: "red soil",
    2: "cotton crop",
    3: "grey soil",
    4: "damp grey soil",
    5: "soil with vegetation stubble",
    7: "very damp grey soil",
}
CLASS_NAMES = list(CLASS_MAP.values())


def read_uci_split(path):
    """Read one UCI Statlog file (36 spectral values and an integer class code)."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if frame.shape[1] != 37:
        raise ValueError(f"{path} should have 37 columns, found {frame.shape[1]}.")
    frame.columns = FEATURE_COLUMNS + ["class_code"]
    frame[TARGET_COLUMN] = frame["class_code"].astype(int).map(CLASS_MAP)
    if frame[TARGET_COLUMN].isna().any():
        unknown = sorted(frame.loc[frame[TARGET_COLUMN].isna(), "class_code"].unique())
        raise ValueError(f"Unrecognised class codes in {path}: {unknown}")
    return frame.drop(columns=["class_code"])


def _as_class_name(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return CLASS_MAP.get(int(text), text)
        lowered = text.lower()
        for name in CLASS_NAMES:
            if lowered == name.lower():
                return name
        return text
    if isinstance(value, (int, np.integer)):
        return CLASS_MAP.get(int(value), str(int(value)))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return CLASS_MAP.get(int(value), str(int(value)))
    return str(value)


def prepare_frame(frame):
    """Return feature matrix and target labels from an uploaded or bundled CSV."""
    renamed = {column: column.strip() for column in frame.columns}
    working = frame.rename(columns=renamed).copy()
    missing = [column for column in FEATURE_COLUMNS if column not in working.columns]
    if missing:
        preview = ", ".join(missing[:6])
        extra = "" if len(missing) <= 6 else f" and {len(missing) - 6} more"
        raise ValueError(
            "The CSV is missing required feature columns "
            f"({len(missing)}): {preview}{extra}."
        )
    if TARGET_COLUMN not in working.columns:
        raise ValueError(
            f"The CSV must include the target column '{TARGET_COLUMN}'."
        )

    features = working[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError(
            "Feature columns must be numeric spectral values. "
            "Some cells could not be read as numbers."
        )
    target = working[TARGET_COLUMN].map(_as_class_name)
    if target.isna().any():
        raise ValueError("The target column contains empty values.")
    unknown = sorted(set(target) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(
            "Unrecognised land-cover labels: " + ", ".join(unknown) + "."
        )
    return features, target


def decision_scores(model, features):
    """Class scores for multiclass AUC. Probabilities are used when the model has them."""
    final_estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(final_estimator, "predict_proba"):
        return model.predict_proba(features)
    return model.decision_function(features)


def weighted_ovr_auc(y_true, y_score, labels):
    """Weighted one-vs-rest AUC.

    Each class is scored against the rest, then averaged by class support.
    Binary AUC accepts either probabilities or decision scores, which keeps
    the support-vector model comparable with the probability-based models.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    aucs = []
    weights = []
    for index, label in enumerate(labels):
        binary = (y_true == label).astype(int)
        positives = int(binary.sum())
        if positives == 0 or positives == len(binary):
            continue
        aucs.append(roc_auc_score(binary, y_score[:, index]))
        weights.append(positives)
    if not aucs:
        return np.nan
    return float(np.average(aucs, weights=weights))


def classification_metrics(y_true, y_pred, y_score, labels):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": weighted_ovr_auc(y_true, y_score, labels),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=labels
        ),
        "recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=labels
        ),
        "f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=labels
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def score_model(model, features, target):
    predicted = model.predict(features)
    scores = decision_scores(model, features)
    labels = list(model.classes_)
    metrics = classification_metrics(target, predicted, scores, labels)
    return metrics, predicted
