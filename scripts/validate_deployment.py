"""
validate_deployment.py
-----------------------
Pre-deployment health checks. Run before pushing to GitHub / deploying to
Streamlit Community Cloud.

Usage
-----
    python scripts/validate_deployment.py

Exits with code 0 if all checks pass, code 1 if any check fails.
"""

import importlib
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"

errors = []
warnings = []


def check(name: str, condition: bool, message: str = "", is_warning: bool = False):
    if condition:
        print(f"{PASS} {name}")
    else:
        tag = WARN if is_warning else FAIL
        print(f"{tag} {name}" + (f" - {message}" if message else ""))
        if is_warning:
            warnings.append(name)
        else:
            errors.append(name)


# ---------------------------------------------------------------------------
# 1. Required files present
# ---------------------------------------------------------------------------
print("\n=== File presence ===")
required_files = [
    "app/main.py",
    "app/features.py",
    "app/modeling.py",
    "app/predictor.py",
    "app/database.py",
    "app/pdf_report.py",
    "models/best_model.pkl",
    "models/best_model_name.pkl",
    "models/feature_names.pkl",
    "models/best_model_evaluation.json",
    "requirements.txt",
    "runtime.txt",
]
for rel_path in required_files:
    full = os.path.join(ROOT, rel_path)
    check(rel_path, os.path.exists(full), f"not found at {full}")

# ---------------------------------------------------------------------------
# 2. requirements.txt sanity
# ---------------------------------------------------------------------------
print("\n=== requirements.txt ===")
req_path = os.path.join(ROOT, "requirements.txt")
if os.path.exists(req_path):
    with open(req_path) as f:
        reqs = f.read()
    must_have = ["pandas", "numpy", "scikit-learn", "streamlit", "tldextract", "reportlab", "plotly"]
    for pkg in must_have:
        check(f"  {pkg} pinned", pkg in reqs, "missing from requirements.txt")
else:
    check("requirements.txt exists", False)

# ---------------------------------------------------------------------------
# 3. Python imports
# ---------------------------------------------------------------------------
print("\n=== Python imports ===")
import_tests = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("tldextract", "tldextract"),
    ("reportlab", "reportlab"),
    ("plotly", "plotly"),
    ("streamlit", "streamlit"),
]
for module, display in import_tests:
    try:
        importlib.import_module(module)
        check(f"  import {display}", True)
    except ImportError as e:
        check(f"  import {display}", False, str(e))

# ---------------------------------------------------------------------------
# 4. Feature contract
# ---------------------------------------------------------------------------
print("\n=== Feature contract ===")
try:
    from app.features import FEATURE_NAMES, feature_vector, normalize_url
    check("FEATURE_NAMES defined", len(FEATURE_NAMES) == 22,
          f"expected 22 features, got {len(FEATURE_NAMES)}")

    # Test normalisation
    normed = normalize_url("example.com")
    check("normalize_url missing-scheme", normed.startswith("https://"))
    fragment_url = normalize_url("https://example.com/page#section")
    check("normalize_url preserves fragment contract", fragment_url.endswith("#section"))

    vec = feature_vector("https://example.com")
    check("feature_vector length", len(vec) == 22,
          f"expected 22, got {len(vec)}")
    check("feature_vector all finite",
          all(isinstance(v, (int, float)) and v == v for v in vec))

    # Private address rejection
    try:
        normalize_url("http://127.0.0.1/login")
        check("private-IP rejected", False, "should have raised ValueError")
    except ValueError as e:
        check("private-IP rejected", "private" in str(e).lower() or "loopback" in str(e).lower())

    # Invalid scheme rejection
    try:
        normalize_url("ftp://example.com")
        check("unsupported-scheme rejected", False, "should have raised ValueError")
    except ValueError:
        check("unsupported-scheme rejected", True)

except Exception as e:
    check("features module import", False, str(e))

# ---------------------------------------------------------------------------
# 5. Model artifact
# ---------------------------------------------------------------------------
print("\n=== Model artifact ===")
model_pkl  = os.path.join(ROOT, "models", "best_model.pkl")
name_pkl   = os.path.join(ROOT, "models", "best_model_name.pkl")
feat_pkl   = os.path.join(ROOT, "models", "feature_names.pkl")
eval_json  = os.path.join(ROOT, "models", "best_model_evaluation.json")

if os.path.exists(model_pkl):
    try:
        import pickle
        with open(model_pkl, "rb") as f:
            model = pickle.load(f)
        check("best_model.pkl loads", True)
        check("model has predict_proba", hasattr(model, "predict_proba"))

        with open(feat_pkl, "rb") as f:
            stored_features = pickle.load(f)
        from app.features import FEATURE_NAMES
        check("feature_names.pkl matches FEATURE_NAMES",
              list(stored_features) == FEATURE_NAMES)

        with open(eval_json) as f:
            eval_data = json.load(f)
        check("best_model_evaluation.json has accuracy", "accuracy" in eval_data)
        check("best_model_evaluation.json has model_name", bool(eval_data.get("model_name")))
        check("domain-grouped split recorded", eval_data.get("domain_overlap") == 0)
        check(
            "feature source recorded",
            "recomputed" in str(eval_data.get("feature_source", "")).lower(),
        )

    except Exception as e:
        check("model artifact valid", False, str(e))
else:
    check("model artifact present", False,
          "Run python scripts/train_models.py first", is_warning=True)

# ---------------------------------------------------------------------------
# 6. Predictor end-to-end
# ---------------------------------------------------------------------------
print("\n=== Predictor end-to-end ===")
if os.path.exists(model_pkl) and os.path.exists(name_pkl):
    try:
        from app.predictor import predict_url
        result = predict_url("https://example.com")
        check("predict_url returns dict", isinstance(result, dict))
        check("label in {Legitimate, Phishing}",
              result.get("label") in {"Legitimate", "Phishing"})
        check("phish_prob in [0, 1]",
              0.0 <= result.get("phish_prob", -1) <= 1.0)
        check("coverage is 22/22", result.get("coverage") == "22/22")
        check("recommendations non-empty",
              isinstance(result.get("recommendations"), list) and len(result["recommendations"]) > 0)
    except Exception as e:
        check("predictor end-to-end", False, str(e))
else:
    check("predictor end-to-end", False, "model not found - skipped", is_warning=True)

# ---------------------------------------------------------------------------
# 7. Database
# ---------------------------------------------------------------------------
print("\n=== Database ===")
try:
    import tempfile, sqlite3
    from app.database import _redact_url, init_db, save_scan, load_scans, clear_history

    # Redaction
    redacted = _redact_url("https://user:pass@example.com/path?token=abc#frag")
    check("URL redaction removes query+credentials",
          "token" not in redacted and "pass" not in redacted)

    check("database module imports", True)
except Exception as e:
    check("database module", False, str(e))

# ---------------------------------------------------------------------------
# 8. PDF generation
# ---------------------------------------------------------------------------
print("\n=== PDF generation ===")
try:
    from app.pdf_report import generate_pdf
    dummy = {
        "url": "https://example.com",
        "label": "Legitimate",
        "phish_prob": 0.05,
        "confidence": 0.95,
        "risk_level": "Low",
        "model_name": "Random Forest",
        "coverage": "22/22",
        "features": {name: 0.0 for name in FEATURE_NAMES},
        "recommendations": ["No phishing pattern detected."],
    }
    pdf_bytes = generate_pdf(dummy)
    check("PDF non-empty", len(pdf_bytes) > 1000)
    check("PDF starts with %PDF", pdf_bytes[:4] == b"%PDF")
except Exception as e:
    check("PDF generation", False, str(e))

# ---------------------------------------------------------------------------
# 9. Report artefacts
# ---------------------------------------------------------------------------
print("\n=== Report artefacts ===")
report_files = [
    "reports/validation_comparison.csv",
    "reports/final_test_metrics.csv",
    "reports/permutation_importance.csv",
    "reports/test_analysis.csv",
    "reports/test_curves.json",
    "reports/recommended_demo_cases_defanged.csv",
]
for rel in report_files:
    full = os.path.join(ROOT, rel)
    check(rel, os.path.exists(full),
          "run training script to generate", is_warning=True)

# ---------------------------------------------------------------------------
# 10. No secrets committed
# ---------------------------------------------------------------------------
print("\n=== Data file ===")
data_paths = ["data/url_phish.csv", "data/Dataset.csv"]
data_found = any(os.path.exists(os.path.join(ROOT, p)) for p in data_paths)
check("Dataset CSV present", data_found,
      "Place url_phish.csv or Dataset.csv in data/", is_warning=True)

sensitive = [".env", "secrets.toml", "credentials.json"]
for fname in sensitive:
    full = os.path.join(ROOT, fname)
    check(f"No {fname} in repo root", not os.path.exists(full))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
if errors:
    print(f"RESULT: {len(errors)} check(s) FAILED, {len(warnings)} warning(s)")
    for e in errors:
        print(f"  [FAIL] {e}")
    sys.exit(1)
else:
    print(f"RESULT: All checks PASSED ({len(warnings)} warning(s))")
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")
    sys.exit(0)
