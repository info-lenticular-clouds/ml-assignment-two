import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model_utils import (
    DATA_DIR,
    MODEL_DIR,
    MODEL_FILES,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    read_uci_split,
    score_model,
)


def build_models():
    # Spectral bands share a 0-255 scale but differ in spread, so every model
    # except the tree is fit on standardised features. The tree uses raw values
    # because axis-aligned splits do not depend on feature scale.
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=16,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                        metric="euclidean",
                    ),
                ),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GaussianNB(var_smoothing=1e-9)),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=10.0,
                        gamma="scale",
                        class_weight="balanced",
                        cache_size=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def main():
    train = read_uci_split(DATA_DIR / "sat.trn")
    test = read_uci_split(DATA_DIR / "sat.tst")
    feature_columns = [column for column in train.columns if column != TARGET_COLUMN]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    test.to_csv(TEST_DATA_PATH, index=False)

    models = build_models()
    summary = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "n_features": len(feature_columns),
        "class_counts_train": train[TARGET_COLUMN].value_counts().to_dict(),
        "class_counts_test": test[TARGET_COLUMN].value_counts().to_dict(),
        "models": {},
    }

    print(f"Train rows: {len(train)} | Test rows: {len(test)} | Features: {len(feature_columns)}")
    print("Train class counts:")
    print(train[TARGET_COLUMN].value_counts().to_string())
    print()
    header = f"{'Model':<24} {'Accuracy':>8} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'MCC':>8}"
    print(header)
    print("-" * len(header))

    for name, model in models.items():
        model.fit(train[feature_columns], train[TARGET_COLUMN])
        metrics, _ = score_model(model, test[feature_columns], test[TARGET_COLUMN])
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])
        summary["models"][name] = {key: round(float(value), 4) for key, value in metrics.items()}
        print(
            f"{name:<24} {metrics['accuracy']:8.4f} {metrics['auc']:8.4f} "
            f"{metrics['precision']:10.4f} {metrics['recall']:8.4f} "
            f"{metrics['f1']:8.4f} {metrics['mcc']:8.4f}"
        )

    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved models and {metrics_path.name}. Wrote {TEST_DATA_PATH.name}.")


if __name__ == "__main__":
    main()
