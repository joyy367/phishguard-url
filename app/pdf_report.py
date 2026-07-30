"""
pdf_report.py
-------------
Generate an in-memory PDF assessment report using ReportLab.
HTML-sensitive characters are escaped before insertion.
"""

import html
import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Colour palette
_BLUE_DARK = colors.HexColor("#1a3a5c")
_BLUE_MID = colors.HexColor("#2563EB")
_BLUE_LIGHT = colors.HexColor("#DBEAFE")
_RED = colors.HexColor("#DC2626")
_GREEN = colors.HexColor("#16A34A")
_AMBER = colors.HexColor("#D97706")
_GREY_LIGHT = colors.HexColor("#F1F5F9")
_GREY_MID = colors.HexColor("#94A3B8")


def _esc(text) -> str:
    return html.escape(str(text))


def generate_pdf(result: dict) -> bytes:
    """
    Build a PDF assessment report from a predictor.predict_url() result dict.

    Parameters
    ----------
    result : dict  – output of predictor.predict_url()

    Returns
    -------
    bytes  – raw PDF content (suitable for st.download_button)
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="PhishGuard URL – Assessment Report",
    )

    styles = getSampleStyleSheet()

    # Custom paragraph styles
    title_style = ParagraphStyle(
        "PGTitle",
        parent=styles["Heading1"],
        textColor=_BLUE_DARK,
        fontSize=18,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "PGSubtitle",
        parent=styles["Normal"],
        textColor=_GREY_MID,
        fontSize=10,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "PGSection",
        parent=styles["Heading2"],
        textColor=_BLUE_DARK,
        fontSize=12,
        spaceBefore=12,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "PGBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
    )
    small_style = ParagraphStyle(
        "PGSmall",
        parent=styles["Normal"],
        fontSize=8,
        textColor=_GREY_MID,
        leading=12,
    )
    table_value_style = ParagraphStyle(
        "PGTableValue",
        parent=body_style,
        fontSize=8,
        leading=10,
        wordWrap="CJK",
    )

    story = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    story.append(Paragraph("PhishGuard URL", title_style))
    story.append(Paragraph("URL-Only Phishing Detection Assessment", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=_BLUE_MID))
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------------------------
    # Assessment summary table
    # ------------------------------------------------------------------
    story.append(Paragraph("Assessment Summary", section_style))

    label = result.get("label", "Unknown")
    phish_prob = result.get("phish_prob", 0.0)
    confidence = result.get("confidence", 0.0)
    risk_level = result.get("risk_level", "Unknown")
    model_name = result.get("model_name", "Unknown")
    url_str = result.get("url", "")
    coverage = result.get("coverage", "22/22")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    verdict_color = _RED if label == "Phishing" else _GREEN
    risk_color = {"High": _RED, "Medium": _AMBER, "Low": _GREEN}.get(risk_level, _GREY_MID)

    summary_data = [
        ["Field", "Value"],
        ["Submitted URL", Paragraph(_esc(url_str), table_value_style)],
        ["Verdict", label],
        ["Phishing Probability", f"{phish_prob*100:.1f}%"],
        ["Verdict Score", f"{confidence*100:.1f}%"],
        ["Risk Level", risk_level],
        ["Feature Coverage", coverage],
        ["Model Used", Paragraph(_esc(model_name), table_value_style)],
        ["Scan Timestamp", timestamp],
    ]

    tbl = Table(summary_data, colWidths=[5 * cm, 12 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _BLUE_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, 1), _GREY_LIGHT),
        ("BACKGROUND", (0, 2), (-1, 2), _BLUE_LIGHT),
        ("TEXTCOLOR", (1, 2), (1, 2), verdict_color),
        ("FONTNAME", (1, 2), (1, 2), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 5), (1, 5), risk_color),
        ("FONTNAME", (1, 5), (1, 5), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _GREY_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, _GREY_MID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    story.append(Paragraph("Recommendations", section_style))
    for rec in result.get("recommendations", []):
        story.append(Paragraph(f"• {_esc(rec)}", body_style))
    story.append(Spacer(1, 0.3 * cm))

    # ------------------------------------------------------------------
    # 22-Feature vector
    # ------------------------------------------------------------------
    story.append(Paragraph("Complete Feature Vector (22/22)", section_style))

    features = result.get("features", {})
    feat_data = [["Feature", "Value", "Description"]]
    feature_descriptions = {
        "url_len": "Total characters in URL",
        "dom_len": "Registrable domain length",
        "is_ip": "Hostname is IP address (0/1)",
        "tld_len": "Public suffix length",
        "subdom_cnt": "Number of subdomain components",
        "letter_cnt": "Alphabetic character count",
        "digit_cnt": "Numeric character count",
        "special_cnt": "Non-alphanumeric character count",
        "eq_cnt": "Number of '=' signs",
        "qm_cnt": "Number of '?' characters",
        "amp_cnt": "Number of '&' characters",
        "dot_cnt": "Number of '.' characters",
        "dash_cnt": "Number of '-' characters",
        "under_cnt": "Number of '_' characters",
        "letter_ratio": "Letter count / URL length",
        "digit_ratio": "Digit count / URL length",
        "spec_ratio": "Special count / URL length",
        "is_https": "HTTPS scheme (0/1)",
        "slash_cnt": "Number of '/' characters",
        "entropy": "Shannon entropy of URL",
        "path_len": "Length of path component",
        "query_len": "Length of query component",
    }

    for feat_name, val in features.items():
        if isinstance(val, float):
            val_str = f"{val:.4f}"
        else:
            val_str = str(val)
        desc = feature_descriptions.get(feat_name, "")
        feat_data.append([feat_name, val_str, desc])

    feat_tbl = Table(feat_data, colWidths=[4 * cm, 3 * cm, 10 * cm])
    feat_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _BLUE_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _GREY_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.3, _GREY_MID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(feat_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------------------------
    # Limitations disclaimer
    # ------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5, color=_GREY_MID))
    story.append(Spacer(1, 0.2 * cm))
    disclaimer = (
        "LIMITATIONS: This assessment analyses URL text only. The application does "
        "not contact the destination, download page content or execute JavaScript. "
        "A Legitimate verdict does not guarantee the website is safe. False negatives "
        "(missed phishing URLs) can occur. Use this tool as an educational screening "
        "aid alongside other security measures."
    )
    story.append(Paragraph(disclaimer, small_style))

    doc.build(story)
    return buf.getvalue()
