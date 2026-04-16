import json
import warnings
from pathlib import Path
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Reuse preprocessing helpers when possible.
# This expects preprocessing.py to be in the same folder as this script.
from preprocessing import (
    encode_features,
    load_data,
    split_features_target,
)

warnings.filterwarnings("ignore")



# Configuration

DATA_FILE = "heart.csv"
RANDOM_STATE = 42
RESULTS_DIR = Path("modeling_results")
RESULTS_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLUMNS = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]



# Utility functions

def prepare_data(file_path: str):
    """
    Load the raw dataset, one-hot encode the categorical columns,
    and create train/validation/test splits.

    Split strategy:
    - 20% test set
    - Remaining 80% split into 75% train / 25% validation
      which gives 60% train / 20% validation / 20% test overall.
    """
    df = load_data(file_path)
    X, y = split_features_target(df)
    X_encoded = encode_features(X, CATEGORICAL_COLUMNS)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X, y, split_name: str):
    """
    Compute the main binary classification metrics for a given split.
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }

    return metrics, predictions, probabilities


def save_confusion_matrix(cm, labels, output_path, title):
    """
    Save a simple confusion matrix figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(image)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true, y_prob, output_path, title):
    """
    Save ROC curve figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def compare_training_behavior(model_name, base_model, X_train, y_train, X_val, y_val):
    """
    Compare training and validation accuracy as the training set size grows.

    This is a good substitute for 'training behavior' on tabular models,
    especially for Random Forest, which does not train across epochs like a neural network.
    """
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    train_scores = []
    val_scores = []

    n_samples = len(X_train)

    for fraction in fractions:
        size = max(10, int(n_samples * fraction))
        X_subset = X_train.iloc[:size]
        y_subset = y_train.iloc[:size]

        current_model = clone(base_model)
        current_model.fit(X_subset, y_subset)

        train_pred = current_model.predict(X_subset)
        val_pred = current_model.predict(X_val)

        train_scores.append(accuracy_score(y_subset, train_pred))
        val_scores.append(accuracy_score(y_val, val_pred))

    behavior_df = pd.DataFrame(
        {
            "train_fraction": fractions,
            "train_accuracy": train_scores,
            "validation_accuracy": val_scores,
        }
    )

    behavior_df.to_csv(RESULTS_DIR / f"{model_name}_training_behavior.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(fractions, train_scores, marker="o", label="Train Accuracy")
    plt.plot(fractions, val_scores, marker="o", label="Validation Accuracy")
    plt.xlabel("Fraction of training set used")
    plt.ylabel("Accuracy")
    plt.title(f"Training Behavior - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_name}_training_behavior.png", dpi=200, bbox_inches="tight")
    plt.close()

    return behavior_df



# Modeling functions

def tune_logistic_regression(X_train, y_train):
    """
    Tune Logistic Regression using cross-validated grid search.

    Notes:
    - max_iter is increased so the solver has enough time to converge.
    - Because one-hot encoded features can have different scales,
      scaling from preprocessing is strongly recommended.
    """
    base_model = LogisticRegression(random_state=RANDOM_STATE)

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"],
        "penalty": ["l2"],
        "max_iter": [1000],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search

def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost using cross-validated grid search.
    """
    base_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "gamma": [0, 1],
        "reg_alpha": [0, 0.5, 1],
        "reg_lambda": [1, 2, 5],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,  
        refit=True,
    )
    search.fit(X_train, y_train)
    return search

def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest using cross-validated grid search.
    """
    base_model = RandomForestClassifier(random_state=RANDOM_STATE)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def feature_importance_report(model, feature_names, model_name="random_forest"):
    """
    Save feature importances for tree-based models.
    """
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importances.to_csv(RESULTS_DIR / f"{model_name}_feature_importance.csv", index=False)

    top_n = importances.head(10)
    plt.figure(figsize=(8, 5))
    plt.barh(top_n["feature"][::-1], top_n["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top 10 {model_name.replace('_', ' ').title()} Feature Importances")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_name}_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()

    return importances



# Main workflow

def main():
    print("Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(DATA_FILE)

    # Save split shapes for documentation.
    split_summary = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "rows": [len(X_train), len(X_val), len(X_test)],
            "columns": [X_train.shape[1], X_val.shape[1], X_test.shape[1]],
        }
    )
    split_summary.to_csv(RESULTS_DIR / "data_split_summary.csv", index=False)

    print("Tuning Logistic Regression...")
    log_search = tune_logistic_regression(X_train, y_train)
    best_log_model = log_search.best_estimator_

    print("Tuning Random Forest...")
    rf_search = tune_random_forest(X_train, y_train)
    best_rf_model = rf_search.best_estimator_

    print("Tuning XGBoost...")
    xgb_search = tune_xgboost(X_train, y_train)
    best_xgb_model = xgb_search.best_estimator_

    # Save best hyperparameters.
    best_params = {
        "logistic_regression": log_search.best_params_,
        "random_forest": rf_search.best_params_,
        "xgboost": xgb_search.best_params_,
    }
    with open(RESULTS_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Evaluate both models on train / validation / test.
    all_results = []
    detailed_reports = {}

    for model_name, model in [
        ("logistic_regression", best_log_model),
        ("random_forest", best_rf_model),
        ("xgboost", best_xgb_model),
    ]:
        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            metrics_dict, y_pred, y_prob = evaluate_model(model, X_split, y_split, split_name)
            metrics_dict["model"] = model_name
            all_results.append(metrics_dict)

            if split_name == "test":
                report = classification_report(y_split, y_pred, output_dict=True)
                detailed_reports[model_name] = report

                cm = confusion_matrix(y_split, y_pred)
                save_confusion_matrix(
                    cm,
                    labels=["No Heart Disease", "Heart Disease"],
                    output_path=RESULTS_DIR / f"{model_name}_confusion_matrix.png",
                    title=f"Confusion Matrix - {model_name.replace('_', ' ').title()}",
                )
                save_roc_curve(
                    y_split,
                    y_prob,
                    RESULTS_DIR / f"{model_name}_roc_curve.png",
                    f"ROC Curve - {model_name.replace('_', ' ').title()}",
                )

    results_df = pd.DataFrame(all_results)
    results_df = results_df[
        ["model", "split", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    results_df.to_csv(RESULTS_DIR / "model_performance_summary.csv", index=False)

    with open(RESULTS_DIR / "classification_reports.json", "w") as f:
        json.dump(detailed_reports, f, indent=2)

    # Compare training behavior.
    print("Comparing training behavior...")
    compare_training_behavior(
        "logistic_regression",
        best_log_model,
        X_train,
        y_train,
        X_val,
        y_val,
    )
    compare_training_behavior(
        "random_forest",
        best_rf_model,
        X_train,
        y_train,
        X_val,
        y_val,
    )
    compare_training_behavior(
        "xgboost",
        best_xgb_model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # Feature importance for tree-based models.
    print("Saving feature importance for Random Forest...")
    feature_importance_report(best_rf_model, X_train.columns, model_name="random_forest")
    print("Saving feature importance for XGBoost...")
    feature_importance_report(best_xgb_model, X_train.columns, model_name="xgboost")

    print("\nDone. Files were saved to the modeling_results folder.")
    print("Best Logistic Regression parameters:", log_search.best_params_)
    print("Best Random Forest parameters:", rf_search.best_params_)
    print("Best XGBoost parameters:", xgb_search.best_params_)
    print("\nModel performance summary:")
    print(results_df)


if __name__ == "__main__":
    main()
