"""Use linear and RBF SVMs to classify the UCI Spambase dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "spambase.data"
NAMES_PATH = HERE / "data" / "spambase.names"
OUTPUT_DIR = HERE / "results"
RANDOM_STATE = 42
TARGET_NAME = "spam"


def load_feature_names(path: Path) -> list[str]:
    """Read the 57 feature names from the source archive's names file."""
    names: list[str] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        line = line.strip()
        if not line or line.startswith("|") or ":" not in line:
            continue
        name = line.split(":", maxsplit=1)[0].strip()
        if name:
            names.append(name)
    if len(names) != 57:
        raise ValueError(f"Expected 57 feature names, found {len(names)} in {path}")
    return names


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    feature_names = load_feature_names(NAMES_PATH)
    frame = pd.read_csv(DATA_PATH, header=None, names=[*feature_names, TARGET_NAME])
    if frame.shape != (4601, 58):
        raise ValueError(f"Unexpected dataset shape: {frame.shape}; expected (4601, 58)")
    if frame.isna().any().any():
        raise ValueError("The source dataset contains missing values; inspect before training.")
    if not set(frame[TARGET_NAME].unique()).issubset({0, 1}):
        raise ValueError("The target must contain only the labels 0 and 1.")
    return frame[feature_names], frame[TARGET_NAME].astype(int)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray]:
    y_pred = model.predict(X_test)
    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    svc = model.named_steps["svc"] if hasattr(model, "named_steps") else None
    return (
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_spam": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "recall_spam": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "f1_spam": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_score),
            "average_precision": average_precision_score(y_test, y_score),
            "support_vectors": int(svc.n_support_.sum()) if svc is not None else None,
        },
        y_pred,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()

    # Group identical feature vectors so exact duplicates cannot appear on both
    # sides of a train/test or cross-validation split.
    groups = pd.factorize(pd.MultiIndex.from_frame(X), sort=False)[0]
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_indices, test_indices = next(outer_cv.split(X, y, groups))
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    groups_train = groups[train_indices]

    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    linear_search = GridSearchCV(
        estimator=make_pipeline(StandardScaler(), SVC(kernel="linear")),
        param_grid={"svc__C": [0.1, 1, 10, 100]},
        scoring="f1",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    rbf_search = GridSearchCV(
        estimator=make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        param_grid={
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.001, 0.01, 0.1],
        },
        scoring="f1",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    linear_search.fit(X_train, y_train, groups=groups_train)
    rbf_search.fit(X_train, y_train, groups=groups_train)

    models = {
        "多数类基线": DummyClassifier(strategy="most_frequent"),
        "线性 SVM（交叉验证调参）": linear_search.best_estimator_,
        "RBF SVM（交叉验证调参）": rbf_search.best_estimator_,
    }
    models["多数类基线"].fit(X_train, y_train)

    cv_scores = {
        "线性 SVM（交叉验证调参）": linear_search.best_score_,
        "RBF SVM（交叉验证调参）": rbf_search.best_score_,
    }
    selected_name = max(cv_scores, key=cv_scores.get)
    selected_model = models[selected_name]

    result_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        predictions[name] = y_pred
        result_rows.append(
            {
                "model": name,
                "cv_best_f1_spam": cv_scores.get(name),
                **metrics,
            }
        )
    results = pd.DataFrame(result_rows)
    results.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    cv_rows: list[dict] = []
    for name, search in (("linear", linear_search), ("rbf", rbf_search)):
        cv_frame = pd.DataFrame(search.cv_results_)
        cv_frame["kernel_family"] = name
        cv_rows.append(
            cv_frame[
                [
                    "kernel_family",
                    "params",
                    "mean_test_score",
                    "std_test_score",
                    "mean_train_score",
                    "rank_test_score",
                ]
            ]
        )
    pd.concat(cv_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "cv_results.csv", index=False, encoding="utf-8-sig"
    )

    selected_pred = predictions[selected_name]
    matrix = confusion_matrix(y_test, selected_pred, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_non_spam_0", "actual_spam_1"],
        columns=["predicted_non_spam_0", "predicted_spam_1"],
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv", encoding="utf-8-sig")

    class_counts = y.value_counts().reindex([0, 1])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(["Non-spam (0)", "Spam (1)"], class_counts.values, color=["#4C78A8", "#F58518"])
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("Emails")
    for index, count in enumerate(class_counts.values):
        axes[0].text(index, count + 25, f"{count:,}", ha="center")

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Non-spam (0)", "Spam (1)"],
    ).plot(ax=axes[1], cmap="Blues", colorbar=False, values_format="d")
    plot_model_name = "Linear SVM" if selected_name.startswith("线性") else "RBF SVM"
    axes[1].set_title(f"Confusion matrix: {plot_model_name}")
    figure.savefig(OUTPUT_DIR / "svm_spambase_results.png", dpi=160)
    plt.close(figure)

    duplicate_rows = int(pd.concat([X, y.rename(TARGET_NAME)], axis=1).duplicated().sum())
    duplicate_feature_rows = int(X.duplicated().sum())
    conflicting_feature_groups = int((pd.concat([X, y.rename(TARGET_NAME)], axis=1)
                                      .groupby(list(X.columns), dropna=False)[TARGET_NAME]
                                      .nunique() > 1).sum())

    print(f"Dataset: {len(X)} rows, {X.shape[1]} features, {int(y.isna().sum())} missing target values")
    print(f"Class counts (0=non-spam, 1=spam): {class_counts.to_dict()}")
    print(
        "Duplicates: "
        f"{duplicate_rows} identical full rows; {duplicate_feature_rows} repeated feature rows; "
        f"{conflicting_feature_groups} feature groups with conflicting labels"
    )
    print(f"Grouped split: {len(train_indices)} train / {len(test_indices)} test rows")
    print(f"Linear SVM best CV F1: {linear_search.best_score_:.4f}; params={linear_search.best_params_}")
    print(f"RBF SVM best CV F1: {rbf_search.best_score_:.4f}; params={rbf_search.best_params_}")
    print(f"Selected by training CV F1: {selected_name}")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Test confusion matrix [[TN, FP], [FN, TP]]: {matrix.tolist()}")
    print(f"Results written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
