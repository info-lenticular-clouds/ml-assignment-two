"""Streamlit front end for the saved Landsat land-cover classifiers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model_utils import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    load_models,
    prepare_frame,
    score_model,
)

st.set_page_config(
    page_title="Landsat Land-Cover Classifier",
    page_icon="🛰️",
    layout="wide",
)

METRIC_LABELS = [
    ("accuracy", "Accuracy"),
    ("auc", "AUC"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("mcc", "MCC"),
]


@st.cache_resource
def get_models():
    return load_models()


def confusion_figure(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted land cover")
    ax.set_ylabel("Actual land cover")
    ax.set_title("Confusion matrix")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def main():
    models = get_models()

    st.title("Landsat land-cover classification")
    st.write(
        "Each row is a 3×3 neighbourhood from a Landsat multispectral scene. "
        "The saved models predict the land cover of the centre pixel. "
        "Upload a test CSV to score it, or leave the uploader empty to use "
        "the bundled UCI test set. The app does not retrain the models."
    )

    with st.sidebar:
        st.header("Controls")
        selected = st.selectbox("Model", list(models))
        upload = st.file_uploader(
            "Upload test CSV",
            type=["csv"],
            help="Include the 36 spectral feature columns and the land_cover target.",
        )
        st.markdown(
            f"**Features:** {len(FEATURE_COLUMNS)} spectral values  \n"
            f"**Target column:** `{TARGET_COLUMN}`  \n"
            "**Classes:** red soil, cotton crop, grey soil, "
            "damp grey soil, soil with vegetation stubble, very damp grey soil"
        )

    if upload is not None:
        try:
            frame = pd.read_csv(upload)
        except Exception:
            st.error("That file could not be read as a CSV.")
            st.stop()
        source = upload.name
    else:
        frame = pd.read_csv(TEST_DATA_PATH)
        source = f"bundled {Path(TEST_DATA_PATH).name}"

    st.caption(f"Scoring `{source}` · {len(frame):,} rows")

    try:
        features, target = prepare_frame(frame)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    comparison_rows = []
    selected_metrics = None
    selected_predictions = None
    for name, model in models.items():
        metrics, predicted = score_model(model, features, target)
        comparison_rows.append({"ML Model Name": name, **metrics})
        if name == selected:
            selected_metrics = metrics
            selected_predictions = predicted

    st.subheader(selected)
    metric_columns = [*st.columns(3), *st.columns(3)]
    for column, (key, label) in zip(metric_columns, METRIC_LABELS):
        value = selected_metrics[key]
        shown = "—" if pd.isna(value) else f"{value:.4f}"
        column.metric(label, shown)

    if pd.isna(selected_metrics["auc"]):
        st.warning(
            "AUC could not be computed. The uploaded file needs examples "
            "from more than one land-cover class."
        )

    chart_column, report_column = st.columns([1.15, 1])
    with chart_column:
        figure = confusion_figure(target, selected_predictions)
        st.pyplot(figure)
        plt.close(figure)
    with report_column:
        st.markdown("**Classification report**")
        report = classification_report(
            target,
            selected_predictions,
            labels=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        report_frame = (
            pd.DataFrame(report)
            .T.loc[CLASS_NAMES, ["precision", "recall", "f1-score", "support"]]
            .rename(
                columns={
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1-score": "F1",
                    "support": "Support",
                }
            )
        )
        st.dataframe(report_frame.style.format("{:.3f}", subset=["Precision", "Recall", "F1"]))

    st.subheader("All five models on this file")
    comparison = (
        pd.DataFrame(comparison_rows)
        .set_index("ML Model Name")
        .rename(
            columns={
                "accuracy": "Accuracy",
                "auc": "AUC",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "mcc": "MCC",
            }
        )
    )
    st.dataframe(comparison.style.format("{:.4f}"))

    preview = features.head(12).copy()
    preview["Actual"] = target.head(12).to_numpy()
    preview["Predicted"] = selected_predictions[:12]
    with st.expander("Sample predictions from the selected model"):
        st.dataframe(preview, hide_index=True)


if __name__ == "__main__":
    main()
