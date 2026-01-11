# src/randomforest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.config import (
    OUT_RND,
    RF_N_ESTIMATORS,
    RF_N_ESTIMATORS_CV,
    RF_MAX_DEPTH,
    RF_MAX_FEATURES,
    RF_MIN_SAMPLES_SPLIT,
    RF_MIN_SAMPLES_LEAF,
    RF_CLASS_WEIGHT,
    RF_BOOTSTRAP,
    RF_RANDOM_STATE,
    RF_TEST_SIZE,
    RF_USE_CV,
    RF_N_FOLDS,
)


def run_random_forest(X: pd.DataFrame, y: pd.Series):
    """
    Train and evaluate a Random Forest classifier using parameters from config.py.
    Saves metrics and plots to OUT_RND.
    """

    OUT_RND.mkdir(parents=True, exist_ok=True)

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS_CV if RF_USE_CV else RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        max_features=RF_MAX_FEATURES,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight=RF_CLASS_WEIGHT,
        bootstrap=RF_BOOTSTRAP,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )

    # -------------------------
    # CROSS-VALIDATION
    # -------------------------
    if RF_USE_CV:
        print(f"[RF] running stratified {RF_N_FOLDS}-fold cross-validation")

        skf = StratifiedKFold(
            n_splits=RF_N_FOLDS,
            shuffle=True,
            random_state=RF_RANDOM_STATE,
        )

        y_pred = cross_val_predict(rf, X, y, cv=skf, n_jobs=1)
        split_name = f"{RF_N_FOLDS}fold_cv"

    # -------------------------
    # TRAIN / TEST SPLIT
    # -------------------------
    else:
        print(f"[RF] running train/test split ({int((1 - RF_TEST_SIZE) * 100)}/{int(RF_TEST_SIZE * 100)})")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=RF_TEST_SIZE,
            stratify=y,
            random_state=RF_RANDOM_STATE,
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        split_name = f"train_test_{int((1 - RF_TEST_SIZE) * 100)}_{int(RF_TEST_SIZE * 100)}"

    # -------------------------
    # METRICS
    # -------------------------
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

    metrics_txt = (
        f"Evaluation mode: {split_name}\n"
        f"Accuracy:  {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall:    {rec:.4f}\n"
        f"F1-score:  {f1:.4f}\n"
    )

    (OUT_RND / "metrics.txt").write_text(metrics_txt)

    # classification report
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(OUT_RND / "classification_report.csv")

    # -------------------------
    # CONFUSION MATRIX
    # -------------------------
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y))

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=90)
    plt.title(f"Confusion Matrix ({split_name})")
    plt.tight_layout()
    plt.savefig(OUT_RND / "confusion_matrix.png", dpi=300)
    plt.close()

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------
    if not RF_USE_CV:
        importances = rf.feature_importances_
        idx = np.argsort(importances)[::-1][:20]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            X.columns[idx][::-1],
            importances[idx][::-1],
        )
        ax.set_title("Top-20 Feature Importances (Random Forest)")
        plt.tight_layout()
        plt.savefig(OUT_RND / "feature_importance.png", dpi=300)
        plt.close()

    print(f"[RF] results saved to {OUT_RND}")

