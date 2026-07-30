"""Tests for cleaning and domain-exclusive data splitting."""

import os
import tempfile

import pandas as pd
import pytest

from app.features import FEATURE_NAMES
from scripts.train_models import load_and_clean, make_splits


def _temporary_csv(rows):
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        dir=reports_dir,
        delete=False,
        encoding="utf-8",
        newline="",
    )
    pd.DataFrame(rows).to_csv(handle, index=False)
    handle.close()
    return handle.name


def test_cleaning_recomputes_features_and_removes_invalid_and_duplicates():
    path = _temporary_csv([
        {"url": "https://example.com/path", "label": 0},
        {"url": "https://example.com/path", "label": 0},
        {"url": "https://known-phish.com/login#account", "label": 1},
        {"url": "not-a-public-host", "label": 0},
    ])
    try:
        cleaned = load_and_clean(path)
    finally:
        os.remove(path)

    assert len(cleaned) == 2
    assert list(cleaned["label"]) == [0, 1]
    assert cleaned[FEATURE_NAMES].notnull().all().all()
    assert cleaned.loc[cleaned["label"].eq(1), "url"].iloc[0].endswith("#account")


def test_conflicting_labels_are_rejected_before_deduplication():
    path = _temporary_csv([
        {"url": "https://example.com", "label": 0},
        {"url": "https://example.com", "label": 1},
    ])
    try:
        with pytest.raises(ValueError, match="Conflicting labels"):
            load_and_clean(path)
    finally:
        os.remove(path)


def test_grouped_split_has_zero_domain_overlap():
    rows = []
    for group_index in range(100):
        label = 1 if group_index % 5 == 0 else 0
        for variant in range(2):
            row = {
                "url": f"https://group{group_index}.test/path/{variant}",
                "dom": f"group{group_index}.test",
                "tld": "test",
                "label": label,
            }
            row.update({name: float(variant) for name in FEATURE_NAMES})
            rows.append(row)
    frame = pd.DataFrame(rows)

    train, validation, test = make_splits(frame, random_state=42)

    train_domains = set(train["dom"])
    validation_domains = set(validation["dom"])
    test_domains = set(test["dom"])
    assert not (train_domains & validation_domains)
    assert not (train_domains & test_domains)
    assert not (validation_domains & test_domains)
    assert len(train) + len(validation) + len(test) == len(frame)
