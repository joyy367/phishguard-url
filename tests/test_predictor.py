"""
test_predictor.py
-----------------
Tests for model loading, schema validation and prediction output.
Skipped automatically when the model artifact is not present.
"""

import os
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PRESENT = (
    os.path.exists(os.path.join(ROOT, "models", "best_model.pkl"))
    and os.path.exists(os.path.join(ROOT, "models", "best_model_name.pkl"))
)

skip_no_model = pytest.mark.skipif(
    not MODEL_PRESENT,
    reason="Model artifact not found – run notebooks/03_model_training.ipynb first",
)


@skip_no_model
class TestPredictorSchema:

    def test_metadata_has_feature_names(self):
        from app.predictor import get_metadata
        from app.features import FEATURE_NAMES
        meta = get_metadata()
        assert meta["feature_names"] == FEATURE_NAMES

    def test_metadata_n_features(self):
        from app.predictor import get_metadata
        meta = get_metadata()
        assert meta["n_features"] == 22

    def test_metadata_model_name_is_selected_candidate(self):
        from app.predictor import get_metadata
        meta = get_metadata()
        assert meta.get("model_name") in {
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Histogram Gradient Boosting",
            "MLP Neural Network",
        }

    def test_metadata_has_test_metrics(self):
        from app.predictor import get_metadata
        meta = get_metadata()
        tm = meta.get("test_metrics", {})
        for key in ["accuracy", "phish_f1", "phish_recall", "roc_auc"]:
            assert key in tm


@skip_no_model
class TestPredictUrl:

    def _predict(self, url: str) -> dict:
        from app.predictor import predict_url
        return predict_url(url)

    def test_legitimate_example_com(self):
        result = self._predict("https://example.com")
        assert result["label"] in {"Legitimate", "Phishing"}

    def test_returns_all_required_keys(self):
        result = self._predict("https://example.com")
        required = {"url", "label", "phish_prob", "confidence",
                    "risk_level", "features", "recommendations",
                    "model_name", "coverage"}
        assert required.issubset(result.keys())

    def test_phish_prob_range(self):
        result = self._predict("https://example.com")
        assert 0.0 <= result["phish_prob"] <= 1.0

    def test_confidence_range(self):
        result = self._predict("https://example.com")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_matches_label(self):
        result = self._predict("https://example.com")
        if result["label"] == "Phishing":
            assert abs(result["confidence"] - result["phish_prob"]) < 1e-6
        else:
            assert abs(result["confidence"] - (1 - result["phish_prob"])) < 1e-6

    def test_label_valid(self):
        result = self._predict("https://example.com")
        assert result["label"] in {"Legitimate", "Phishing"}

    def test_risk_level_valid(self):
        result = self._predict("https://example.com")
        assert result["risk_level"] in {"Low", "Medium", "High"}

    def test_coverage_22_of_22(self):
        result = self._predict("https://example.com")
        assert result["coverage"] == "22/22"

    def test_features_dict_22_keys(self):
        result = self._predict("https://example.com")
        assert len(result["features"]) == 22

    def test_recommendations_non_empty(self):
        result = self._predict("https://example.com")
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    def test_invalid_url_raises(self):
        from app.predictor import predict_url
        with pytest.raises(ValueError):
            predict_url("http://127.0.0.1/admin")

    def test_model_name_in_result(self):
        result = self._predict("https://example.com")
        assert isinstance(result["model_name"], str)
        assert len(result["model_name"]) > 0

    def test_normalised_url_in_result(self):
        result = self._predict("example.com")
        # Missing scheme should be prepended
        assert result["url"].startswith("https://")


@skip_no_model
class TestModelReady:

    def test_model_ready_true(self):
        from app.predictor import model_ready
        assert model_ready() is True
