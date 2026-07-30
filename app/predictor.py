"""
predictor.py
------------
Loads the saved model artifacts (pkl files) and exposes predict_url().

Model artifacts (saved by notebooks/03_model_training.ipynb):
    models/best_model.pkl             - trained selected classifier
    models/best_model_name.pkl        - string name of selected model
    models/feature_names.pkl          - ordered list of 22 feature names
    models/best_model_evaluation.json - evaluation metrics from test set
"""

import json
import os
import pickle
import warnings

import pandas as pd

from app.features import FEATURE_NAMES, extract_url_features, normalize_url

# Silence sklearn feature names warning (model works correctly, this is cosmetic)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ---------------------------------------------------------------------------
# Artifact paths  (matching your existing models/ folder structure)
# ---------------------------------------------------------------------------
_ROOT = os.path.join(os.path.dirname(__file__), "..")
_MODEL_PKL      = os.path.join(_ROOT, "models", "best_model.pkl")
_MODEL_NAME_PKL = os.path.join(_ROOT, "models", "best_model_name.pkl")
_FEATURES_PKL   = os.path.join(_ROOT, "models", "feature_names.pkl")
_EVAL_JSON      = os.path.join(_ROOT, "models", "best_model_evaluation.json")

# ---------------------------------------------------------------------------
# Module-level singletons (loaded once on first call)
# ---------------------------------------------------------------------------
_model        = None
_model_name   = None
_eval_metrics = None


def _load():
    global _model, _model_name, _eval_metrics

    if _model is None:
        # --- best_model.pkl ---
        if not os.path.exists(_MODEL_PKL):
            raise FileNotFoundError(
                f"Model not found at '{_MODEL_PKL}'. "
                "Run notebook 03_model_training.ipynb first."
            )
        with open(_MODEL_PKL, "rb") as f:
            _model = pickle.load(f)

        # --- best_model_name.pkl ---
        if not os.path.exists(_MODEL_NAME_PKL):
            raise FileNotFoundError(
                f"Model name not found at '{_MODEL_NAME_PKL}'. "
                "Run notebooks/03_model_training.ipynb again."
            )
        with open(_MODEL_NAME_PKL, "rb") as f:
            _model_name = pickle.load(f)

        # --- feature_names.pkl: validate contract ---
        if not os.path.exists(_FEATURES_PKL):
            raise FileNotFoundError(
                f"Feature contract not found at '{_FEATURES_PKL}'. "
                "Run notebooks/03_model_training.ipynb again."
            )
        with open(_FEATURES_PKL, "rb") as f:
            stored_features = pickle.load(f)
        if list(stored_features) != FEATURE_NAMES:
            raise RuntimeError(
                "Saved feature_names.pkl does not match the live FEATURE_NAMES "
                "contract. Retrain the model by running notebooks/03_model_training.ipynb."
            )

        # --- best_model_evaluation.json ---
        if not os.path.exists(_EVAL_JSON):
            raise FileNotFoundError(
                f"Evaluation metadata not found at '{_EVAL_JSON}'. "
                "Run notebooks/03_model_training.ipynb again."
            )
        with open(_EVAL_JSON) as f:
            _eval_metrics = json.load(f)

    return _model, _model_name, _eval_metrics


# ---------------------------------------------------------------------------
# Risk level helper
# ---------------------------------------------------------------------------

def _risk_level(phish_prob: float) -> str:
    if phish_prob >= 0.75:
        return "High"
    elif phish_prob >= 0.40:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def _build_recommendations(label: str, phish_prob: float) -> list:
    if label == "Phishing":
        return [
            "Do NOT visit this URL or enter any credentials.",
            "Do not click any links associated with this address.",
            "Report the URL to your security team or use a threat-intelligence feed.",
            "If you have already visited it, change your passwords immediately.",
            f"Model phishing score: {phish_prob*100:.1f}%",
        ]
    else:
        legit_prob = 1.0 - phish_prob
        return [
            "No phishing pattern was detected in the URL text.",
            "This result analyses the URL structure only — it is NOT a safety guarantee.",
            "The application does not visit the destination or inspect page content.",
            "Always verify the sender and context before entering sensitive information.",
            f"Model legitimate score: {legit_prob*100:.1f}%",
        ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_url(raw_url: str) -> dict:
    """
    Validate, normalise and classify a URL.

    Returns dict with: url, label, phish_prob, confidence, risk_level,
                       features, recommendations, model_name, coverage
    """
    model, model_name, _ = _load()

    # Normalise (raises ValueError on invalid input)
    normalised = normalize_url(raw_url)

    # Extract 22 features
    feat_dict = extract_url_features(normalised)

    # One-row DataFrame in correct column order
    frame = pd.DataFrame(
        [[feat_dict[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    # Predict
    phish_prob  = float(model.predict_proba(frame)[0, 1])
    pred_class  = int(model.predict(frame)[0])
    label       = "Phishing" if pred_class == 1 else "Legitimate"
    confidence  = phish_prob if label == "Phishing" else 1.0 - phish_prob

    return {
        "url":             normalised,
        "label":           label,
        "phish_prob":      phish_prob,
        "confidence":      confidence,
        "risk_level":      _risk_level(phish_prob),
        "features":        feat_dict,
        "recommendations": _build_recommendations(label, phish_prob),
        "model_name":      model_name,
        "coverage":        f"{len(FEATURE_NAMES)}/{len(FEATURE_NAMES)}",
        "n_features":      len(FEATURE_NAMES),
    }


def get_metadata() -> dict:
    """Return evaluation metrics + model name for the Overview page."""
    _, model_name, eval_metrics = _load()
    return {
        "model_name":    model_name,
        "feature_names": FEATURE_NAMES,
        "n_features":    len(FEATURE_NAMES),
        "test_metrics":  eval_metrics,
        # row counts come from the eval JSON if present
        "train_rows":    eval_metrics.get("train_rows", "N/A"),
        "val_rows":      eval_metrics.get("val_rows",   "N/A"),
        "test_rows":     eval_metrics.get("test_rows",  "N/A"),
        "dev_rows":      eval_metrics.get("dev_rows",   "N/A"),
    }


def model_ready() -> bool:
    """Return True only when the complete deployment artifact set exists."""
    return all(
        os.path.exists(path)
        for path in (_MODEL_PKL, _MODEL_NAME_PKL, _FEATURES_PKL, _EVAL_JSON)
    )
