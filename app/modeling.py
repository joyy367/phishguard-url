"""
modeling.py
-----------
Defines the five candidate models and the shared evaluation helper.
All model definitions live here so that train_models.py and tests can
import a single source of truth.
"""

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
import numpy as np


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_candidate_models() -> dict:
    """
    Return an ordered dict of {name: estimator} for all five candidates.
    Hyperparameters are fixed before validation comparison.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs",
                n_jobs=-1,
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=18,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=42,
        ),
        "MLP Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                early_stopping=True,
                learning_rate="adaptive",
                random_state=42,
                max_iter=300,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model(model, X, y, model_name: str = "") -> dict:
    """
    Run a full suite of security-focused classification metrics.

    Parameters
    ----------
    model   : fitted sklearn estimator
    X       : feature matrix (array-like)
    y       : true labels (0 = legitimate, 1 = phishing)
    model_name : optional label for the result dict

    Returns
    -------
    dict of metric name -> value
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    phish_precision = precision_score(y, y_pred, pos_label=1, zero_division=0)
    phish_recall = recall_score(y, y_pred, pos_label=1, zero_division=0)
    phish_f1 = f1_score(y, y_pred, pos_label=1, zero_division=0)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "weighted_f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "phish_precision": phish_precision,
        "phish_recall": phish_recall,
        "phish_f1": phish_f1,
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "log_loss": log_loss(y, y_proba),
        "brier_score": brier_score_loss(y, y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    return metrics


def select_best_model(validation_results: list) -> str:
    """
    Apply the predefined selection rule:
      Primary  : highest phishing F1
      Tie-break: highest phishing recall → highest weighted F1 → lowest log loss

    Parameters
    ----------
    validation_results : list of metric dicts from evaluate_model()

    Returns
    -------
    name of the selected model (str)
    """
    ranked = sorted(
        validation_results,
        key=lambda m: (
            m["phish_f1"],
            m["phish_recall"],
            m["weighted_f1"],
            -m["log_loss"],
        ),
        reverse=True,
    )
    return ranked[0]["model_name"]
