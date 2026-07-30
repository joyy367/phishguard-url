"""
test_pdf_report.py
------------------
Tests for PDF generation via ReportLab.
"""

import pytest
from app.features import FEATURE_NAMES


def _make_result(label="Legitimate", phish_prob=0.05):
    return {
        "url": "https://example.com",
        "label": label,
        "phish_prob": phish_prob,
        "confidence": 1.0 - phish_prob if label == "Legitimate" else phish_prob,
        "risk_level": "Low" if phish_prob < 0.4 else ("Medium" if phish_prob < 0.75 else "High"),
        "model_name": "Random Forest",
        "coverage": "22/22",
        "features": {name: 0.0 for name in FEATURE_NAMES},
        "recommendations": ["Test recommendation"],
    }


class TestPdfGeneration:

    def test_pdf_is_bytes(self):
        from app.pdf_report import generate_pdf
        pdf = generate_pdf(_make_result())
        assert isinstance(pdf, bytes)

    def test_pdf_non_trivial_size(self):
        from app.pdf_report import generate_pdf
        pdf = generate_pdf(_make_result())
        assert len(pdf) > 1000, f"PDF too small: {len(pdf)} bytes"

    def test_pdf_starts_with_header(self):
        from app.pdf_report import generate_pdf
        pdf = generate_pdf(_make_result())
        assert pdf[:4] == b"%PDF", "PDF does not start with %PDF header"

    def test_pdf_legitimate_result(self):
        from app.pdf_report import generate_pdf
        pdf = generate_pdf(_make_result("Legitimate", 0.03))
        assert len(pdf) > 1000

    def test_pdf_phishing_result(self):
        from app.pdf_report import generate_pdf
        pdf = generate_pdf(_make_result("Phishing", 0.92))
        assert len(pdf) > 1000

    def test_pdf_with_special_chars_in_url(self):
        from app.pdf_report import generate_pdf
        result = _make_result()
        result["url"] = "https://example.com/path?a=1&b=<script>"
        pdf = generate_pdf(result)
        assert len(pdf) > 1000

    def test_pdf_all_22_features(self):
        from app.pdf_report import generate_pdf
        result = _make_result()
        # Set non-zero values to exercise table rendering
        result["features"] = {name: float(i + 1) for i, name in enumerate(FEATURE_NAMES)}
        pdf = generate_pdf(result)
        assert len(pdf) > 1000

    def test_pdf_high_risk(self):
        from app.pdf_report import generate_pdf
        pdf = generate_pdf(_make_result("Phishing", 0.97))
        assert len(pdf) > 1000

    def test_pdf_multiple_recommendations(self):
        from app.pdf_report import generate_pdf
        result = _make_result()
        result["recommendations"] = [
            "Do not visit this URL.",
            "Report to security team.",
            "Change passwords if visited.",
        ]
        pdf = generate_pdf(result)
        assert len(pdf) > 1000
