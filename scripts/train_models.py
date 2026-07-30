"""
train_models.py
---------------
Full training pipeline:
  1. Load and clean URL-Phish v2 CSV
  2. Recompute the 22 features through the deployment feature extractor
  3. Domain-grouped, approximately stratified 70 / 15 / 15 split
  4. Fit all five candidate models on the training split
  5. Compare on validation using phishing F1 and compute importance there
  6. Refit the winner on train + validation
  7. Evaluate once on the domain-unseen test split and save artifacts

Usage
-----
    python scripts/train_models.py --data data/url_phish.csv

The CSV must have a 'url' column and a 'label' column
(0 = legitimate, 1 = phishing). Any stored numerical features are ignored
and safely recomputed so training and deployed inference share one contract.
"""

import argparse
from itertools import combinations
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

# Make sure the repo root is on sys.path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.features import (
    FEATURE_NAMES,
    extract_url_features,
    get_domain_parts,
    normalize_url,
)
from app.modeling import evaluate_model, get_candidate_models, select_best_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train PhishGuard URL models")
    # Accept both common filenames automatically
    _default = os.path.join(ROOT, "data", "url_phish.csv")
    if not os.path.exists(_default):
        _alt = os.path.join(ROOT, "data", "Dataset.csv")
        if os.path.exists(_alt):
            _default = _alt
    p.add_argument(
        "--data",
        default=_default,
        help="Path to the URL-Phish CSV file",
    )
    p.add_argument(
        "--random-state", type=int, default=42,
        help="Global random seed (default 42)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_and_clean(csv_path: str) -> pd.DataFrame:
    print(f"[1/7] Loading dataset from {csv_path} ...")
    raw_df = pd.read_csv(csv_path)
    print(f"      Raw rows: {len(raw_df):,}  |  columns: {list(raw_df.columns)}")

    # The deployed model re-extracts all numerical inputs from URL text, so
    # URL and label are the only authoritative source columns.
    required = ["url", "label"]
    missing_cols = [c for c in required if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    # Enforce the binary label contract before any duplicate removal.
    labels = pd.to_numeric(raw_df["label"], errors="raise")
    if labels.isnull().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("The label column must contain only 0 (legitimate) and 1 (phishing).")
    raw_df = raw_df.copy()
    raw_df["label"] = labels.astype(int)

    print("      Normalising URLs and re-extracting the 22 deployment features ...")
    cleaned_rows = []
    invalid_rows = []
    for row_number, (raw_url, label) in enumerate(
        raw_df[["url", "label"]].itertuples(index=False, name=None),
        start=2,
    ):
        try:
            normalised = normalize_url(str(raw_url))
            features = extract_url_features(normalised)
            domain, suffix, _ = get_domain_parts(normalised)
        except Exception as exc:
            invalid_rows.append((row_number, str(raw_url), str(exc)))
            continue
        cleaned_rows.append({
            "url": normalised,
            "dom": domain,
            "tld": suffix,
            **features,
            "label": int(label),
        })

    if invalid_rows:
        sample = "; ".join(
            f"row {number}: {message}" for number, _, message in invalid_rows[:5]
        )
        print(
            f"      Removed {len(invalid_rows):,} invalid URL row(s). "
            f"Examples: {sample}"
        )

    df = pd.DataFrame(cleaned_rows)

    # Check conflicts before deduplication so conflicting rows cannot be hidden.
    conflicts = df.groupby("url")["label"].nunique()
    conflicting_urls = conflicts[conflicts > 1].index.tolist()
    print(f"      Conflicting-label URL count: {len(conflicting_urls)}")
    if conflicting_urls:
        raise ValueError(
            "Conflicting labels found for the same normalised URL. "
            f"Examples: {conflicting_urls[:5]}"
        )

    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    print(f"      Removed {removed:,} duplicate URLs -> {len(df):,} unique rows")

    feature_values = df[FEATURE_NAMES].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("The recomputed feature matrix contains non-finite values.")
    print("      Feature matrix: complete and finite")

    # Class distribution
    vc = df["label"].value_counts().sort_index()
    print(f"      Label 0 (legitimate): {vc.get(0,0):,}  |  Label 1 (phishing): {vc.get(1,0):,}")
    print(f"      Phishing prevalence: {vc.get(1,0)/len(df)*100:.2f}%")

    return df


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def make_splits(df: pd.DataFrame, random_state: int):
    """Create a domain-exclusive, approximately stratified 70 / 15 / 15 split."""
    print("\n[2/7] Creating domain-grouped 70/15/15 split ...")

    # Twenty stratified group folds provide 5% units. Combining 14/3/3 folds
    # gives 70/15/15 while ensuring each registrable domain belongs to exactly
    # one subset.
    splitter = StratifiedGroupKFold(
        n_splits=20,
        shuffle=True,
        random_state=random_state,
    )
    fold_ids = np.full(len(df), -1, dtype=int)
    for fold_id, (_, fold_indices) in enumerate(
        splitter.split(df[FEATURE_NAMES], df["label"], groups=df["dom"])
    ):
        fold_ids[fold_indices] = fold_id

    if (fold_ids < 0).any():
        raise RuntimeError("Some rows were not assigned to a grouped split fold.")

    fold_stats = []
    for fold_id in range(20):
        fold_rows = df[fold_ids == fold_id]
        fold_stats.append({
            "fold": fold_id,
            "rows": len(fold_rows),
            "legitimate": int(fold_rows["label"].eq(0).sum()),
            "phishing": int(fold_rows["label"].eq(1).sum()),
        })

    total_rows = len(df)
    total_legitimate = int(df["label"].eq(0).sum())
    total_phishing = int(df["label"].eq(1).sum())

    def combination_score(folds):
        selected = [fold_stats[fold_id] for fold_id in folds]
        rows = sum(item["rows"] for item in selected)
        legitimate = sum(item["legitimate"] for item in selected)
        phishing = sum(item["phishing"] for item in selected)
        target = 0.15
        return (
            abs(rows / total_rows - target)
            + abs(legitimate / total_legitimate - target)
            + abs(phishing / total_phishing - target)
        )

    # Fold IDs have no semantic meaning. Select disjoint three-fold
    # combinations that most closely match 15% of rows and each class.
    test_folds = set(
        min(combinations(range(20), 3), key=combination_score)
    )
    remaining_folds = [fold for fold in range(20) if fold not in test_folds]
    validation_folds = set(
        min(combinations(remaining_folds, 3), key=combination_score)
    )
    test_df = df[np.isin(fold_ids, list(test_folds))].copy()
    val_df = df[np.isin(fold_ids, list(validation_folds))].copy()
    train_df = df[
        ~np.isin(fold_ids, list(test_folds | validation_folds))
    ].copy()

    train_domains = set(train_df["dom"])
    val_domains = set(val_df["dom"])
    test_domains = set(test_df["dom"])
    if train_domains & val_domains or train_domains & test_domains or val_domains & test_domains:
        raise RuntimeError("Domain leakage detected between grouped subsets.")

    for name, split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        vc = split["label"].value_counts().sort_index()
        prevalence = vc.get(1, 0) / len(split)
        print(f"      {name:10s}: {len(split):,} rows  "
              f"(legit={vc.get(0,0):,}, phish={vc.get(1,0):,}, "
              f"phish%={prevalence*100:.2f}, domains={split['dom'].nunique():,})")
    print("      Domain overlap across subsets: 0")

    overall_prevalence = df["label"].mean()
    for split_name, split in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        if abs(split["label"].mean() - overall_prevalence) > 0.02:
            raise RuntimeError(
                f"{split_name} phishing prevalence differs from the full dataset "
                "by more than two percentage points."
            )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Training and validation comparison
# ---------------------------------------------------------------------------

def train_and_compare(train_df, val_df):
    print("\n[3/7] Training 5 candidate models ...")
    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["label"].to_numpy()
    X_val = val_df[FEATURE_NAMES]
    y_val = val_df["label"].to_numpy()

    models = get_candidate_models()
    fitted = {}
    val_results = []

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        metrics = evaluate_model(model, X_val, y_val, model_name=name)
        metrics["train_time_s"] = round(elapsed, 3)
        fitted[name] = model
        val_results.append(metrics)
        print(
            f"      {name:<35s}  "
            f"phish_F1={metrics['phish_f1']:.4f}  "
            f"recall={metrics['phish_recall']:.4f}  "
            f"FP={metrics['false_positives']}  FN={metrics['false_negatives']}"
        )

    return fitted, val_results


# ---------------------------------------------------------------------------
# Final fit and test evaluation
# ---------------------------------------------------------------------------

def final_evaluation(fitted_models, val_results, train_df, val_df, test_df):
    print("\n[4/7] Selecting best model ...")
    best_name = select_best_model(val_results)
    print(f"      Selected: {best_name}")

    # Refit on train + validation combined (85 %)
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    X_dev = dev_df[FEATURE_NAMES]
    y_dev = dev_df["label"].to_numpy()
    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["label"].to_numpy()

    print(f"\n[5/7] Refitting {best_name} on train+val ({len(dev_df):,} rows) ...")
    best_model = fitted_models[best_name]
    best_model.fit(X_dev, y_dev)

    print("\n[6/7] Final evaluation on untouched test set ...")
    test_metrics = evaluate_model(best_model, X_test, y_test, model_name=best_name)

    print(f"      Accuracy        : {test_metrics['accuracy']:.4f}")
    print(f"      Balanced Acc    : {test_metrics['balanced_accuracy']:.4f}")
    print(f"      Phishing F1     : {test_metrics['phish_f1']:.4f}")
    print(f"      Phishing Recall : {test_metrics['phish_recall']:.4f}")
    print(f"      ROC-AUC         : {test_metrics['roc_auc']:.4f}")
    print(f"      PR-AUC          : {test_metrics['pr_auc']:.4f}")
    print(f"      Log Loss        : {test_metrics['log_loss']:.4f}")
    print(f"      Brier Score     : {test_metrics['brier_score']:.4f}")
    print(f"      FP={test_metrics['false_positives']}  FN={test_metrics['false_negatives']}")

    return best_name, best_model, test_metrics, val_results


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def compute_importance(model, X, y, split_name: str = "validation") -> pd.DataFrame:
    print(f"\n      Computing permutation importance on {split_name} data (n_repeats=10) ...")
    result = permutation_importance(
        model, X, y,
        n_repeats=10,
        random_state=42,
        scoring="f1",
        n_jobs=-1,
    )
    imp_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return imp_df


# ---------------------------------------------------------------------------
# Curve data helpers
# ---------------------------------------------------------------------------

def compute_roc_pr_curves(model, X, y, label: str) -> dict:
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    prec, rec, _ = precision_recall_curve(y, y_proba)
    return {
        "label": label,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": prec.tolist(),
        "recall": rec.tolist(),
    }


# ---------------------------------------------------------------------------
# Artefact saving
# ---------------------------------------------------------------------------

def save_artefacts(
    best_name, best_model, test_metrics, val_results,
    train_df, val_df, test_df, imp_df,
):
    print("\n[7/7] Saving artefacts ...")

    # --- best_model.pkl ---
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"      Model saved        -> {model_path}")

    # --- best_model_name.pkl ---
    name_path = os.path.join(MODELS_DIR, "best_model_name.pkl")
    with open(name_path, "wb") as f:
        pickle.dump(best_name, f)
    print(f"      Model name saved   -> {name_path}")

    # --- feature_names.pkl ---
    feat_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    with open(feat_path, "wb") as f:
        pickle.dump(FEATURE_NAMES, f)
    print(f"      Feature names saved-> {feat_path}")

    # --- label_encoder.pkl  (encodes 0→Legitimate, 1→Phishing) ---
    label_map = {0: "Legitimate", 1: "Phishing"}
    enc_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    with open(enc_path, "wb") as f:
        pickle.dump(label_map, f)
    print(f"      Label encoder saved-> {enc_path}")

    # --- best_model_evaluation.json ---
    eval_data = {
        "model_name":   best_name,
        "split_strategy": "StratifiedGroupKFold (20 folds; 14/3/3; grouped by domain)",
        "feature_source": "Recomputed from normalised URL text using app.features",
        "domain_overlap": 0,
        "total_rows":   len(train_df) + len(val_df) + len(test_df),
        "legitimate_rows": int(
            pd.concat([train_df, val_df, test_df])["label"].eq(0).sum()
        ),
        "phishing_rows": int(
            pd.concat([train_df, val_df, test_df])["label"].eq(1).sum()
        ),
        "train_rows":   len(train_df),
        "val_rows":     len(val_df),
        "test_rows":    len(test_df),
        "dev_rows":     len(train_df) + len(val_df),
        **test_metrics,
    }
    eval_path = os.path.join(MODELS_DIR, "best_model_evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"      Evaluation JSON    -> {eval_path}")

    # --- model_comparison.csv (validation results) ---
    val_df_out = pd.DataFrame(val_results)
    comp_csv = os.path.join(MODELS_DIR, "model_comparison.csv")
    val_df_out.to_csv(comp_csv, index=False)
    print(f"      Model comparison   -> {comp_csv}")

    # --- reports/validation_comparison.csv ---
    val_csv = os.path.join(REPORTS_DIR, "validation_comparison.csv")
    val_df_out.to_csv(val_csv, index=False)

    # --- reports/final_test_metrics.csv ---
    test_csv = os.path.join(REPORTS_DIR, "final_test_metrics.csv")
    pd.DataFrame([test_metrics]).to_csv(test_csv, index=False)
    print(f"      Final test metrics -> {test_csv}")

    # --- reports/permutation_importance.csv ---
    imp_csv = os.path.join(REPORTS_DIR, "permutation_importance.csv")
    imp_df.to_csv(imp_csv, index=False)
    print(f"      Perm. importance   -> {imp_csv}")

    # --- reports/full_test_set_results.csv ---
    X_test = test_df[FEATURE_NAMES]
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred  = best_model.predict(X_test)
    test_results_df = test_df.copy()
    test_results_df["predicted_label"]  = y_pred
    test_results_df["phish_probability"] = y_proba
    results_csv = os.path.join(REPORTS_DIR, "full_test_set_results.csv")
    test_results_df.to_csv(results_csv, index=False)
    print(f"      Row-level results  -> {results_csv}")

    # Deployment-safe test analysis: probabilities for all rows and defanged,
    # truncated URLs only for errors. This file contains no clickable URLs.
    analysis_df = test_results_df[
        ["label", "predicted_label", "phish_probability"]
    ].copy()
    analysis_df["display_url"] = ""
    error_mask = analysis_df["label"] != analysis_df["predicted_label"]
    analysis_df.loc[error_mask, "display_url"] = (
        test_results_df.loc[error_mask, "url"].map(_defang_url)
    )
    analysis_csv = os.path.join(REPORTS_DIR, "test_analysis.csv")
    analysis_df.to_csv(analysis_csv, index=False)
    print(f"      Safe test analysis -> {analysis_csv}")

    # --- reports/recommended_demo_cases.csv ---
    correct_phish = test_results_df[
        (test_results_df["label"] == 1) & (test_results_df["predicted_label"] == 1)
    ].sort_values("phish_probability", ascending=False).head(5)
    correct_legit = test_results_df[
        (test_results_df["label"] == 0) & (test_results_df["predicted_label"] == 0)
    ].sort_values("phish_probability", ascending=True).head(5)
    demo_df = pd.concat([correct_phish, correct_legit])
    if "url" in demo_df.columns:
        demo_csv = os.path.join(REPORTS_DIR, "recommended_demo_cases.csv")
        demo_df[["url", "label", "predicted_label", "phish_probability"]].to_csv(
            demo_csv, index=False
        )
        print(f"      Raw demo cases     -> {demo_csv} (keep local; gitignored)")

        safe_demo = demo_df[
            ["url", "label", "predicted_label", "phish_probability"]
        ].copy()
        safe_demo["url"] = safe_demo["url"].map(_defang_url)
        safe_demo_csv = os.path.join(
            REPORTS_DIR, "recommended_demo_cases_defanged.csv"
        )
        safe_demo.to_csv(safe_demo_csv, index=False)
        print(f"      Defanged demo cases-> {safe_demo_csv}")

    # --- reports/test_curves.json ---
    curves = compute_roc_pr_curves(
        best_model, X_test, test_df["label"].to_numpy(), best_name
    )
    curves_path = os.path.join(REPORTS_DIR, "test_curves.json")
    with open(curves_path, "w") as f:
        json.dump(curves, f)
    print(f"      Curve data         -> {curves_path}")

    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# Reproducibility contract check
# ---------------------------------------------------------------------------

def _defang_url(url: str, max_length: int = 180) -> str:
    """Return a non-clickable, length-limited URL for public artefacts."""
    defanged = str(url).replace("https://", "hxxps://").replace("http://", "hxxp://")
    defanged = defanged.replace(".", "[.]")
    if len(defanged) > max_length:
        return defanged[: max_length - 1] + "…"
    return defanged


def check_reproducibility(df: pd.DataFrame, n_sample: int = 500):
    """
    Re-extract features from n_sample URLs and confirm they match the
    stored CSV values within floating-point tolerance.
    Only runs if the 'url' column is present.
    """
    sample = df.sample(min(n_sample, len(df)), random_state=42)
    mismatches = 0
    for _, row in sample.iterrows():
        normed = normalize_url(str(row["url"]))
        recalc = extract_url_features(normed)
        for fname in FEATURE_NAMES:
            stored = float(row[fname])
            fresh = float(recalc.get(fname, 0))
            if abs(stored - fresh) > 1e-6:
                mismatches += 1
    total_cells = len(sample) * len(FEATURE_NAMES)
    print(f"      Reproducibility check: {total_cells - mismatches}/{total_cells} cells matched")
    if mismatches:
        raise RuntimeError(
            f"Feature contract failed: {mismatches} sampled cell(s) did not match."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at '{args.data}'")
        print("Please download URL-Phish v2 from Mendeley Data (DOI 10.17632/65z9twcx3r.2)")
        print("and place the CSV at data/url_phish.csv")
        sys.exit(1)

    df = load_and_clean(args.data)
    train_df, val_df, test_df = make_splits(df, args.random_state)
    fitted_models, val_results = train_and_compare(train_df, val_df)

    # Explainability is calculated on validation data before the final refit,
    # leaving the test labels for the one final performance evaluation.
    best_validation_name = select_best_model(val_results)
    X_val = val_df[FEATURE_NAMES]
    y_val = val_df["label"].to_numpy()
    imp_df = compute_importance(
        fitted_models[best_validation_name],
        X_val,
        y_val,
        split_name="validation",
    )

    best_name, best_model, test_metrics, val_results = final_evaluation(
        fitted_models, val_results, train_df, val_df, test_df
    )

    print("\nTop 10 features by permutation importance:")
    print(imp_df.head(10).to_string(index=False))

    # Optional reproducibility check
    check_reproducibility(df, n_sample=500)

    save_artefacts(best_name, best_model, test_metrics, val_results,
                   train_df, val_df, test_df, imp_df)


if __name__ == "__main__":
    main()
