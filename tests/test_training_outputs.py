"""
test_training_outputs.py
------------------------
Validates the saved training artefacts: metadata JSON, validation CSV,
test metrics CSV, and row-level test results.
Skipped automatically when artefacts are not present.
"""

import json
import os
import pytest
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
REPORTS = os.path.join(ROOT, "reports")
MODELS = os.path.join(ROOT, "models")

_artefacts_present = (
    os.path.exists(os.path.join(MODELS, "best_model.pkl"))
    and os.path.exists(os.path.join(MODELS, "best_model_evaluation.json"))
    and os.path.exists(os.path.join(REPORTS, "validation_comparison.csv"))
    and os.path.exists(os.path.join(REPORTS, "final_test_metrics.csv"))
)

skip_no_artefacts = pytest.mark.skipif(
    not _artefacts_present,
    reason="Training artefacts not found – run scripts/train_models.py first",
)


@skip_no_artefacts
class TestMetadata:

    def _eval(self):
        with open(os.path.join(MODELS, "best_model_evaluation.json")) as f:
            return json.load(f)

    def _feat(self):
        import pickle
        with open(os.path.join(MODELS, "feature_names.pkl"), "rb") as f:
            return pickle.load(f)

    def test_model_name_present(self):
        ev = self._eval()
        assert "model_name" in ev
        assert len(ev["model_name"]) > 0

    def test_feature_names_22(self):
        from app.features import FEATURE_NAMES
        feat = self._feat()
        assert list(feat) == FEATURE_NAMES

    def test_split_sizes_positive(self):
        ev = self._eval()
        assert ev.get("train_rows", 0) > 0
        assert ev.get("val_rows", 0) > 0
        assert ev.get("test_rows", 0) > 0

    def test_dev_rows_equals_train_plus_val(self):
        ev = self._eval()
        assert ev.get("dev_rows") == ev.get("train_rows", 0) + ev.get("val_rows", 0)

    def test_test_metrics_accuracy_in_range(self):
        ev = self._eval()
        assert 0.0 <= ev.get("accuracy", -1) <= 1.0

    def test_test_metrics_phish_f1_in_range(self):
        ev = self._eval()
        assert 0.0 <= ev.get("phish_f1", -1) <= 1.0

    def test_confusion_matrix_values_present(self):
        ev = self._eval()
        for key in ["true_negatives", "false_positives", "false_negatives", "true_positives"]:
            assert key in ev
            assert ev[key] >= 0


@skip_no_artefacts
class TestValidationComparison:

    def _df(self):
        return pd.read_csv(os.path.join(REPORTS, "validation_comparison.csv"))

    def test_five_models_compared(self):
        df = self._df()
        assert len(df) == 5, f"Expected 5 models, got {len(df)}"

    def test_required_columns_present(self):
        df = self._df()
        for col in ["model_name", "phish_f1", "phish_recall", "accuracy"]:
            assert col in df.columns

    def test_no_null_values(self):
        df = self._df()
        assert df.isnull().sum().sum() == 0

    def test_phish_f1_in_range(self):
        df = self._df()
        assert (df["phish_f1"].between(0, 1)).all()

    def test_random_forest_present(self):
        df = self._df()
        assert df["model_name"].str.contains("Random Forest").any()

    def test_selected_model_has_highest_phish_f1(self):
        df = self._df()
        import pickle
        from app.modeling import select_best_model
        with open(os.path.join(MODELS, "best_model_name.pkl"), "rb") as f:
            selected = pickle.load(f)
        assert selected == select_best_model(df.to_dict(orient="records"))

    def test_grouped_split_metadata(self):
        with open(os.path.join(MODELS, "best_model_evaluation.json")) as f:
            evaluation = json.load(f)
        assert evaluation.get("domain_overlap") == 0
        assert "group" in evaluation.get("split_strategy", "").lower()


@skip_no_artefacts
class TestFinalTestMetrics:

    def _df(self):
        return pd.read_csv(os.path.join(REPORTS, "final_test_metrics.csv"))

    def test_single_row(self):
        df = self._df()
        assert len(df) == 1

    def test_accuracy_matches_evaluation_json(self):
        df = self._df()
        with open(os.path.join(MODELS, "best_model_evaluation.json")) as f:
            ev = json.load(f)
        meta_acc = ev.get("accuracy", None)
        assert meta_acc is not None
        csv_acc = df["accuracy"].iloc[0]
        assert abs(meta_acc - csv_acc) < 1e-6


@skip_no_artefacts
class TestPermutationImportance:

    def _df(self):
        path = os.path.join(REPORTS, "permutation_importance.csv")
        if not os.path.exists(path):
            pytest.skip("permutation_importance.csv not found")
        return pd.read_csv(path)

    def test_22_features(self):
        df = self._df()
        assert len(df) == 22

    def test_required_columns(self):
        df = self._df()
        assert "feature" in df.columns
        assert "importance_mean" in df.columns

    def test_feature_names_match_contract(self):
        from app.features import FEATURE_NAMES
        df = self._df()
        assert set(df["feature"].tolist()) == set(FEATURE_NAMES)
