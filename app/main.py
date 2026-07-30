"""
main.py
-------
PhishGuard URL – Streamlit multi-page application.
Pages: Overview | URL Scanner | Analytics | Scan History
"""

import json
import os
import sys
import uuid

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import clear_history, get_summary, load_scans, save_scan
from app.features import FEATURE_NAMES
from app.pdf_report import generate_pdf
from app.predictor import get_metadata, model_ready, predict_url

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard URL",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "phishguard_session_id" not in st.session_state:
    st.session_state.phishguard_session_id = uuid.uuid4().hex
if "last_scan_result" not in st.session_state:
    st.session_state.last_scan_result = None

# ---------------------------------------------------------------------------
# Global CSS — light theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* ── App background ── */
.stApp {
    background: #f8f9fb;
    min-height: 100vh;
}
.block-container {
    padding-top: 4rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px;
}

/* ── Hide Streamlit's top toolbar decoration that causes black bar ── */
[data-testid="stHeader"] {
    background: transparent !important;
}
header[data-testid="stHeader"] {
    background-color: transparent !important;
    backdrop-filter: none !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e6ea;
}
[data-testid="stSidebar"] * { color: #1e293b !important; }
[data-testid="stSidebar"] .stRadio label {
    cursor: pointer;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 4px 0;
    transition: color 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #2563eb !important; }

/* ── Headings ── */
h1 { color: #0f172a !important; font-weight: 800 !important; letter-spacing: -0.4px; }
h2, h3 { color: #1e293b !important; font-weight: 700 !important; }
p, li { color: #334155; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 60%, #1e40af 100%);
    border-radius: 14px;
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 40%, rgba(255,255,255,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.2rem; font-weight: 800; color: #ffffff;
    margin-bottom: 8px; line-height: 1.2;
}
.hero-subtitle { font-size: 1rem; color: rgba(255,255,255,0.8); font-weight: 400; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.35);
    color: #ffffff; font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px; margin-bottom: 12px;
}

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e6ea;
    border-radius: 10px;
    padding: 16px 14px;
    text-align: center;
    margin: 3px;
    transition: box-shadow 0.2s, transform 0.2s;
}
.metric-card:hover {
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}
.metric-card .label {
    font-size: 0.67rem; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 600; margin-bottom: 6px;
}
.metric-card .value       { font-size: 1.6rem; font-weight: 800; color: #2563eb; line-height: 1.1; }
.metric-card .value.green { color: #16a34a; }
.metric-card .value.red   { color: #dc2626; }
.metric-card .value.amber { color: #d97706; }

/* ── Risk badges ── */
.risk-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 0.73rem; font-weight: 700;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.risk-high   { background: #fef2f2; border: 1px solid #fca5a5; color: #dc2626; }
.risk-medium { background: #fffbeb; border: 1px solid #fcd34d; color: #b45309; }
.risk-low    { background: #f0fdf4; border: 1px solid #86efac; color: #16a34a; }

/* ── Verdict banners ── */
.verdict-phishing {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    border-left: 5px solid #dc2626;
    border-radius: 10px; padding: 18px 22px; margin: 12px 0;
}
.verdict-legitimate {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 5px solid #16a34a;
    border-radius: 10px; padding: 18px 22px; margin: 12px 0;
}
.verdict-title { font-size: 1.3rem; font-weight: 800; color: #0f172a; margin-bottom: 5px; }
.verdict-meta  { font-size: 0.88rem; color: #475569; }

/* ── Disclaimer box ── */
.disclaimer-box {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-left: 4px solid #f59e0b;
    border-radius: 8px; padding: 11px 15px;
    font-size: 0.84rem; color: #78350f; margin: 10px 0;
}

/* ── Scanner section label ── */
.scanner-section-label {
    font-size: 0.75rem; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 6px;
}

/* ── Pipeline steps ── */
.pipeline-step {
    background: #ffffff;
    border: 1px solid #e2e6ea;
    border-radius: 10px; padding: 16px 12px; text-align: center;
    transition: box-shadow 0.2s, transform 0.2s; height: 100%;
}
.pipeline-step:hover {
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
    transform: translateY(-2px);
}
.pipeline-icon  { font-size: 1.5rem; margin-bottom: 8px; }
.pipeline-label { font-size: 0.72rem; font-weight: 700; color: #2563eb;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.pipeline-desc  { font-size: 0.8rem; color: #64748b; line-height: 1.4; }

/* ── Recommendation cards ── */
.rec-card {
    background: #f8f9fb;
    border: 1px solid #e2e6ea;
    border-radius: 8px; padding: 11px 14px; margin: 5px 0;
    font-size: 0.86rem; color: #334155;
    display: flex; align-items: flex-start; gap: 10px;
}
.rec-icon { font-size: 0.95rem; margin-top: 1px; flex-shrink: 0; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    color: #64748b !important;
}
[data-testid="stTabs"] button[aria-selected="true"] { color: #2563eb !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ── Buttons (all) ── */
[data-testid="stButton"] > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: box-shadow 0.15s, transform 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.18) !important;
}
/* Primary button — blue (overrides Streamlit's default red/coral) */
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    padding: 10px 0 !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.22) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.32) !important;
}

/* ── All text inputs (single consolidated rule) ── */
[data-testid="stTextInputRootElement"] input,
[data-testid="stTextInput"] input {
    border-radius: 10px !important;
    background: #ffffff !important;
    border: 2px solid #e2e6ea !important;
    color: #0f172a !important;
    font-size: 0.97rem !important;
    padding: 11px 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
[data-testid="stTextInputRootElement"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.11) !important;
    outline: none !important;
}
/* Remove the outer container border that Streamlit sometimes adds */
[data-testid="stTextInputRootElement"] {
    border: none !important;
    box-shadow: none !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — clean, minimal, no model status, no emojis in nav
# ---------------------------------------------------------------------------
PAGES = ["Overview", "URL Scanner", "Analytics", "Scan History"]

with st.sidebar:
    # Brand mark
    st.markdown("""
    <div style="padding: 18px 4px 12px;">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:36px;height:36px;background:#2563eb;border-radius:8px;
             display:flex;align-items:center;justify-content:center;
             font-size:1.1rem;flex-shrink:0;">🛡️</div>
        <div>
          <div style="font-size:1rem;font-weight:800;color:#0f172a;letter-spacing:-0.2px;
               line-height:1.2;">PhishGuard URL</div>
          <div style="font-size:0.68rem;color:#94a3b8;letter-spacing:0.06em;
               text-transform:uppercase;">URL Phishing Detection</div>
        </div>
      </div>
    </div>
    <hr style="border:none;border-top:1px solid #e2e6ea;margin:0 0 14px;">
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.68rem;font-weight:700;color:#94a3b8;'
                'text-transform:uppercase;letter-spacing:0.1em;'
                'margin-bottom:8px;">Navigation</div>', unsafe_allow_html=True)

    page = st.radio("Navigate", PAGES, label_visibility="collapsed")

    st.markdown("""
    <hr style="border:none;border-top:1px solid #e2e6ea;margin:18px 0 12px;">
    <div style="font-size:0.7rem;color:#cbd5e1;text-align:left;line-height:1.6;">
      PhishGuard URL<br>EATC Assignment 2 · 2026
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metadata():
    return get_metadata() if model_ready() else {}


def _load_val_comparison():
    for rel in ["reports/validation_comparison.csv", "models/model_comparison.csv"]:
        path = os.path.join(os.path.dirname(__file__), "..", rel)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def _load_importance():
    path = os.path.join(os.path.dirname(__file__), "..", "reports", "permutation_importance.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def _load_test_curves():
    path = os.path.join(os.path.dirname(__file__), "..", "reports", "test_curves.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_test_results():
    path = os.path.join(os.path.dirname(__file__), "..", "reports", "test_analysis.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def _metric_card(label, value, col, color="blue"):
    value_class = {"blue": "", "green": " green", "red": " red", "amber": " amber"}.get(color, "")
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value{value_class}">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _risk_color(risk: str) -> str:
    return {"High": "#dc2626", "Medium": "#d97706", "Low": "#16a34a"}.get(risk, "#64748b")


def _risk_badge(risk: str) -> str:
    cls = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(risk, "")
    return f'<span class="risk-badge {cls}">{risk} Risk</span>'


def _plotly_base_layout(**kwargs):
    """Common Plotly layout — light theme."""
    base = dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fb",
        font=dict(color="#334155", family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=16, r=16, t=48, b=16),
        title_font=dict(size=14, color="#1e293b"),
    )
    base.update(kwargs)
    return base

# ===========================================================================
# PAGE 1 – OVERVIEW
# ===========================================================================

def page_overview():
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-badge">Phishing Detection · URL Analysis</div>
      <div class="hero-title">PhishGuard URL</div>
      <div class="hero-subtitle">
        Machine-learning phishing detection from URL structure alone —
        no page visits, no DNS, no WHOIS. Analyse any public URL in milliseconds.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    <strong>URL-Only Scope:</strong> This application analyses the URL <em>as text only</em>.
    It does <strong>not</strong> visit the destination, download page content, or execute JavaScript.
    A <em>Legitimate</em> prediction is <strong>not</strong> a safety guarantee.
    </div>
    """, unsafe_allow_html=True)

    meta = _load_metadata()
    tm   = meta.get("test_metrics", {})

    st.markdown("### Final Test Performance")
    st.caption("Domain-unseen 15 % test set — reported once after model selection.")
    if tm:
        c1, c2, c3, c4, c5 = st.columns(5)
        _metric_card("Accuracy",        f"{tm.get('accuracy', 0)*100:.2f}%",          c1)
        _metric_card("Phishing F1",     f"{tm.get('phish_f1', 0)*100:.2f}%",          c2)
        _metric_card("Phishing Recall", f"{tm.get('phish_recall', 0)*100:.2f}%",      c3, "green")
        _metric_card("ROC-AUC",         f"{tm.get('roc_auc', 0):.4f}",                c4)
        _metric_card("PR-AUC",          f"{tm.get('pr_auc', 0):.4f}",                 c5)

        c6, c7, c8, c9 = st.columns(4)
        _metric_card("Weighted F1",     f"{tm.get('weighted_f1', 0)*100:.2f}%",       c6)
        _metric_card("Balanced Acc.",   f"{tm.get('balanced_accuracy', 0)*100:.2f}%", c7)
        _metric_card("Log Loss",        f"{tm.get('log_loss', 0):.4f}",               c8, "amber")
        _metric_card("Brier Score",     f"{tm.get('brier_score', 0):.4f}",            c9, "amber")
    else:
        st.info("Final metrics will appear after `scripts/train_models.py` completes.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Dataset & Methodology")
    col_a, col_b = st.columns(2)

    with col_a:
        total_rows      = tm.get("total_rows")
        legitimate_rows = tm.get("legitimate_rows")
        phishing_rows   = tm.get("phishing_rows")
        if all(isinstance(v, int) for v in (total_rows, legitimate_rows, phishing_rows)):
            row_summary = (
                f"- **Rows after validation & deduplication:** {total_rows:,}\n"
                f"- **Legitimate:** {legitimate_rows:,} · **Phishing:** {phishing_rows:,}\n"
                f"- **Phishing prevalence:** {phishing_rows/total_rows*100:.2f}%"
            )
        else:
            row_summary = "- Cleaned row counts will appear after training."

        st.markdown("""
        <div class="metric-card" style="text-align:left;padding:18px 20px;">
          <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;
               letter-spacing:0.08em;font-weight:600;margin-bottom:10px;">Dataset</div>
        """, unsafe_allow_html=True)
        st.markdown(
            "**Source:** URL-Phish v2 (Mendeley Data) · DOI 10.17632/65z9twcx3r.2 · CC BY 4.0\n\n"
            f"{row_summary}\n\n"
            "**Features:** 22 lexical/structural URL features — no webpage visit required"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="metric-card" style="text-align:left;padding:18px 20px;">
          <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;
               letter-spacing:0.08em;font-weight:600;margin-bottom:10px;">Evaluation Design</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        - Domain-grouped **70 / 15 / 15** split (random_state=42)
        - No registrable-domain overlap between train, validation, and test
        - **5 candidate models** compared on validation phishing F1
        - Winner refitted on 85 % (train + validation)
        - Final metrics reported **once** on domain-unseen 15 % test set
        - All 22 features recomputed through the deployment extractor
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    if meta:
        st.markdown("### Selected Model")
        model_name = meta.get("model_name", "N/A")
        tr  = meta.get("train_rows", "N/A")
        vr  = meta.get("val_rows",   "N/A")
        tsr = meta.get("test_rows",  "N/A")
        dr  = meta.get("dev_rows",   "N/A")
        tr_str  = f"{tr:,}"  if isinstance(tr,  int) else str(tr)
        vr_str  = f"{vr:,}"  if isinstance(vr,  int) else str(vr)
        tsr_str = f"{tsr:,}" if isinstance(tsr, int) else str(tsr)
        dr_str  = f"{dr:,}"  if isinstance(dr,  int) else str(dr)
        st.markdown(f"""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
             padding:14px 18px;">
          <div style="font-size:1rem;font-weight:700;color:#1d4ed8;margin-bottom:3px;">
            {model_name}
          </div>
          <div style="font-size:0.82rem;color:#64748b;">
            Selected by highest validation phishing F1 &nbsp;·&nbsp;
            Train: {tr_str} &nbsp;·&nbsp; Val: {vr_str} &nbsp;·&nbsp;
            Test: {tsr_str} &nbsp;·&nbsp; Dev: {dr_str}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### How It Works")
    selected_model = meta.get("model_name", "Selected Classifier")
    pipeline = [
        ("1", "Input",    "Enter any public HTTP / HTTPS URL"),
        ("2", "Validate", "Normalise scheme, reject private/local addresses"),
        ("3", "Extract",  "Calculate 22 URL-derived features — no web request"),
        ("4", "Predict",  f"{selected_model} returns a phishing probability score"),
        ("5", "Report",   "View risk assessment, feature vector &amp; download PDF"),
    ]
    cols = st.columns(5)
    for col, (num, label, desc) in zip(cols, pipeline):
        col.markdown(f"""
        <div class="pipeline-step">
          <div class="pipeline-label">{num}. {label}</div>
          <div class="pipeline-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 2 – URL SCANNER
# ===========================================================================

def _render_scan_result(result: dict):
    label         = result["label"]
    phish_prob    = result["phish_prob"]
    verdict_score = result["confidence"]
    risk          = result["risk_level"]

    banner_class = "verdict-phishing" if label == "Phishing" else "verdict-legitimate"
    banner_text  = "Phishing Detected" if label == "Phishing" else "Legitimate URL"
    st.markdown(
        f'<div class="{banner_class}">'
        f'<div class="verdict-title">{banner_text}</div>'
        f'<div class="verdict-meta">'
        f'Phishing score: <strong>{phish_prob*100:.1f}%</strong>'
        f' &nbsp;·&nbsp; Confidence: <strong>{verdict_score*100:.1f}%</strong>'
        f' &nbsp;·&nbsp; {_risk_badge(risk)}'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Model: **{result['model_name']}** · Feature coverage: {result['coverage']} · "
        "Risk bands are heuristic score ranges, not guarantees."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Details", "Feature Vector", "Recommendations", "PDF Report"])

    with tab1:
        st.markdown('<div style="margin-bottom:6px;font-size:0.82rem;color:#64748b;">Normalised URL</div>',
                    unsafe_allow_html=True)
        st.code(result["url"], language=None, wrap_lines=True)

        col1, col2, col3 = st.columns(3)
        color1 = "red" if label == "Phishing" else "green"
        color3 = {"High": "red", "Medium": "amber", "Low": "green"}.get(risk, "blue")
        _metric_card("Phishing Score", f"{phish_prob*100:.2f}%",    col1, color1)
        _metric_card("Confidence",     f"{verdict_score*100:.2f}%", col2)
        _metric_card("Risk Band",      risk,                         col3, color3)

        st.markdown("<br>", unsafe_allow_html=True)

        gauge_color = "#dc2626" if label == "Phishing" else "#16a34a"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=phish_prob * 100,
            title={"text": "Phishing Score (%)", "font": {"size": 13, "color": "#64748b"}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickcolor": "#94a3b8",
                         "tickfont": {"size": 10, "color": "#94a3b8"}},
                "bar": {"color": gauge_color, "thickness": 0.24},
                "bgcolor": "#f8f9fb",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],   "color": "#dcfce7"},
                    {"range": [40, 75],  "color": "#fef9c3"},
                    {"range": [75, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#94a3b8", "width": 2},
                    "thickness": 0.75, "value": 50,
                },
            },
            number={"suffix": "%", "font": {"size": 30, "color": "#0f172a"}},
        ))
        fig.update_layout(**_plotly_base_layout(height=250, margin=dict(l=20, r=20, t=30, b=10)))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown(
            '<div style="font-size:0.84rem;color:#64748b;margin-bottom:10px;">'
            '<strong style="color:#2563eb;">22/22</strong> features extracted — no webpage visit required'
            '</div>',
            unsafe_allow_html=True,
        )
        feat_items  = list(result["features"].items())
        mid         = len(feat_items) // 2
        col_l, col_r = st.columns(2)
        for col, items in [(col_l, feat_items[:mid]), (col_r, feat_items[mid:])]:
            col.dataframe(
                pd.DataFrame(
                    [(k, f"{v:.4f}" if isinstance(v, float) else str(v)) for k, v in items],
                    columns=["Feature", "Value"],
                ),
                use_container_width=True, hide_index=True,
            )

    with tab3:
        icon = "!" if label == "Phishing" else "i"
        for rec in result["recommendations"]:
            st.markdown(
                f'<div class="rec-card"><span class="rec-icon">{icon}</span>'
                f'<span>{rec}</span></div>',
                unsafe_allow_html=True,
            )

    with tab4:
        st.markdown(
            '<div style="padding:10px 0 6px;color:#64748b;font-size:0.87rem;">'
            'Generate a full PDF report with the URL assessment, feature vector, '
            'risk summary, and all recommendations.'
            '</div>',
            unsafe_allow_html=True,
        )
        try:
            pdf_bytes = generate_pdf(result)
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="phishguard_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")


def page_scanner():
    st.markdown("""
    <div style="margin-bottom:4px;">
      <span style="font-size:1.7rem;font-weight:800;color:#0f172a;">🔍 URL Scanner</span>
    </div>
    <div style="font-size:0.9rem;color:#64748b;margin-bottom:14px;">
      Enter any public URL — a missing scheme is treated as HTTPS.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    <strong>URL-only analysis:</strong> This scanner reads URL text only and never contacts the destination.
    A <em>Legitimate</em> result is not a safety guarantee.
    </div>
    """, unsafe_allow_html=True)

    if not model_ready():
        st.error(
            "Model artifacts not found. "
            "Run `python scripts/train_models.py --data data/Dataset.csv` first."
        )
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # URL input — rendered cleanly, no broken wrapper divs
    st.markdown('<div class="scanner-section-label">URL to analyse</div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "URL to scan",
        placeholder="https://example.com/login  or  example.com",
        help="Enter a full URL or hostname. Private/local addresses are rejected.",
        label_visibility="collapsed",
    )

    scan_clicked = st.button("🔍  Scan URL", type="primary", use_container_width=True)

    if scan_clicked:
        if not url_input.strip():
            st.warning("Please enter a URL before scanning.")
        else:
            with st.spinner("Analysing URL structure…"):
                try:
                    result = predict_url(url_input)
                except ValueError as exc:
                    st.error(f"Invalid URL: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
                else:
                    st.session_state.last_scan_result = result
                    save_scan(
                        url=result["url"],
                        prediction=result["label"],
                        phish_prob=result["phish_prob"],
                        risk_level=result["risk_level"],
                        model_name=result["model_name"],
                        session_id=st.session_state.phishguard_session_id,
                    )

    if st.session_state.last_scan_result:
        st.markdown('<hr style="border:none;border-top:1px solid #e2e6ea;margin:20px 0 16px;">',
                    unsafe_allow_html=True)
        _render_scan_result(st.session_state.last_scan_result)


# ===========================================================================
# PAGE 3 – ANALYTICS
# ===========================================================================

# Columns where HIGHER is better (we highlight the max)
_HIGHER_IS_BETTER = {"accuracy", "phish_precision", "phish_recall",
                     "phish_f1", "weighted_f1", "roc_auc", "pr_auc"}
# Columns where LOWER is better (we highlight the min)
_LOWER_IS_BETTER  = {"false_positives", "false_negatives", "log_loss", "brier_score"}


def _highlight_best(raw_df: pd.DataFrame, display_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a same-shape DataFrame of CSS strings that bolds and colours the
    best value in each numeric metric column.
    """
    styles = pd.DataFrame("", index=display_df.index, columns=display_df.columns)
    for col in display_df.columns:
        if col == "model_name":
            continue
        if col in _HIGHER_IS_BETTER and col in raw_df.columns:
            best_idx = raw_df[col].idxmax()
            styles.at[best_idx, col] = (
                "background-color:#dbeafe;color:#1d4ed8;"
                "font-weight:700;border-radius:4px;"
            )
        elif col in _LOWER_IS_BETTER and col in raw_df.columns:
            best_idx = raw_df[col].idxmin()
            styles.at[best_idx, col] = (
                "background-color:#dcfce7;color:#15803d;"
                "font-weight:700;border-radius:4px;"
            )
    return styles


def page_analytics():
    st.markdown("""
    <div style="margin-bottom:4px;">
      <span style="font-size:1.7rem;font-weight:800;color:#0f172a;">Analytics</span>
    </div>
    <div style="font-size:0.9rem;color:#64748b;margin-bottom:14px;">
      Model evaluation results, validation comparison, and feature analysis.
    </div>
    """, unsafe_allow_html=True)

    meta    = _load_metadata()
    val_df  = _load_val_comparison()
    imp_df  = _load_importance()
    curves  = _load_test_curves()
    test_df = _load_test_results()

    if val_df is None:
        st.warning("No analytics data found. Run `python scripts/train_models.py` to generate reports.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Model Comparison",
        "ROC / PR Curves",
        "Confusion Matrix",
        "Feature Importance",
        "Test Errors",
    ])

    # ── Tab 1 : Model Comparison ────────────────────────────────────────
    with tab1:
        st.subheader("Validation Split — 5 Candidate Model Comparison")
        st.caption(
            "Primary selection metric: Phishing F1  ·  "
            "Tie-breaks: Phishing Recall → Weighted F1 → lower Log Loss  ·  "
            "Blue = best (higher is better)  ·  Green = best (lower is better)"
        )

        display_cols = [
            "model_name", "accuracy", "phish_precision", "phish_recall",
            "phish_f1", "weighted_f1", "roc_auc", "pr_auc",
            "false_positives", "false_negatives",
        ]
        show_cols = [c for c in display_cols if c in val_df.columns]
        raw_df    = val_df[show_cols].copy()   # numeric — used for idxmax/idxmin
        fmt_df    = raw_df.copy()              # formatted strings for display

        for col in ["accuracy", "phish_precision", "phish_recall", "phish_f1", "weighted_f1"]:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].map(lambda x: f"{x*100:.2f}%")
        for col in ["roc_auc", "pr_auc"]:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.4f}")

        styled = fmt_df.style.apply(
            lambda _: _highlight_best(raw_df, fmt_df), axis=None
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        best_name = meta.get("model_name", "")
        st.markdown(f"""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:7px;
             padding:9px 15px;font-size:0.87rem;color:#1d4ed8;margin:8px 0;">
          Selected model: <strong>{best_name}</strong> — highest phishing F1
        </div>
        """, unsafe_allow_html=True)

        if "phish_f1" in val_df.columns:
            fig_bar = px.bar(
                val_df.sort_values("phish_f1", ascending=True),
                x="phish_f1", y="model_name", orientation="h",
                color="phish_f1",
                color_continuous_scale=["#bfdbfe", "#2563eb", "#1d4ed8"],
                labels={"phish_f1": "Phishing F1", "model_name": "Model"},
                title="Validation Phishing F1 — All Candidates",
            )
            fig_bar.update_layout(**_plotly_base_layout(coloraxis_showscale=False, height=300))
            fig_bar.update_xaxes(tickformat=".2%", gridcolor="#e2e6ea", showgrid=True)
            fig_bar.update_yaxes(gridcolor="#e2e6ea")
            st.plotly_chart(fig_bar, use_container_width=True)

        if "false_positives" in val_df.columns and "false_negatives" in val_df.columns:
            fig_fp = go.Figure()
            fig_fp.add_trace(go.Bar(
                name="False Positives", x=val_df["model_name"],
                y=val_df["false_positives"],
                marker=dict(color="#fbbf24", line=dict(width=0)),
            ))
            fig_fp.add_trace(go.Bar(
                name="False Negatives", x=val_df["model_name"],
                y=val_df["false_negatives"],
                marker=dict(color="#ef4444", line=dict(width=0)),
            ))
            fig_fp.update_layout(
                **_plotly_base_layout(
                    barmode="group", height=300,
                    title="Validation — False Positives vs False Negatives",
                )
            )
            fig_fp.update_xaxes(gridcolor="#e2e6ea")
            fig_fp.update_yaxes(gridcolor="#e2e6ea")
            st.plotly_chart(fig_fp, use_container_width=True)

        tm = meta.get("test_metrics", {})
        if tm:
            st.markdown("#### Final Test Set Metrics")
            st.caption("Selected model — one evaluation on domain-unseen test set.")
            c1, c2, c3, c4 = st.columns(4)
            _metric_card("Accuracy",          f"{tm.get('accuracy', 0)*100:.2f}%",       c1)
            _metric_card("Phishing F1",        f"{tm.get('phish_f1', 0)*100:.2f}%",      c2)
            _metric_card("Phishing Recall",    f"{tm.get('phish_recall', 0)*100:.2f}%",  c3, "green")
            _metric_card("ROC-AUC",            f"{tm.get('roc_auc', 0):.4f}",            c4)
            c5, c6, c7, c8 = st.columns(4)
            _metric_card("Phishing Precision", f"{tm.get('phish_precision', 0)*100:.2f}%", c5)
            _metric_card("PR-AUC",             f"{tm.get('pr_auc', 0):.4f}",               c6)
            _metric_card("Log Loss",           f"{tm.get('log_loss', 0):.4f}",             c7, "amber")
            _metric_card("Brier Score",        f"{tm.get('brier_score', 0):.4f}",          c8, "amber")

    # ── Tab 2 : ROC / PR Curves ─────────────────────────────────────────
    with tab2:
        st.subheader("Test Set — ROC and Precision-Recall Curves")
        if curves:
            tm = meta.get("test_metrics", {})
            col_roc, col_pr = st.columns(2)

            with col_roc:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=curves["fpr"], y=curves["tpr"], mode="lines",
                    name=curves["label"],
                    line=dict(color="#2563eb", width=2.5),
                    fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dot", color="#94a3b8", width=1.5),
                    name="Random Baseline",
                ))
                fig_roc.update_layout(
                    **_plotly_base_layout(
                        title=f"ROC Curve  (AUC = {tm.get('roc_auc', 0):.4f})",
                        height=360,
                    )
                )
                fig_roc.update_xaxes(title_text="False Positive Rate",
                                     gridcolor="#e2e6ea", range=[0, 1])
                fig_roc.update_yaxes(title_text="True Positive Rate",
                                     gridcolor="#e2e6ea", range=[0, 1])
                st.plotly_chart(fig_roc, use_container_width=True)

            with col_pr:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=curves["recall"], y=curves["precision"], mode="lines",
                    name=curves["label"],
                    line=dict(color="#16a34a", width=2.5),
                    fill="tozeroy", fillcolor="rgba(22,163,74,0.07)",
                ))
                fig_pr.update_layout(
                    **_plotly_base_layout(
                        title=f"Precision-Recall Curve  (AUC = {tm.get('pr_auc', 0):.4f})",
                        height=360,
                    )
                )
                fig_pr.update_xaxes(title_text="Recall",
                                    gridcolor="#e2e6ea", range=[0, 1])
                fig_pr.update_yaxes(title_text="Precision",
                                    gridcolor="#e2e6ea", range=[0, 1])
                st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.info("Curve data not found. Run training script to generate.")

    # ── Tab 3 : Confusion Matrix ────────────────────────────────────────
    with tab3:
        st.subheader("Final Test Set — Confusion Matrix")
        tm = meta.get("test_metrics", {})
        if tm:
            tn = tm.get("true_negatives",  0)
            fp = tm.get("false_positives", 0)
            fn = tm.get("false_negatives", 0)
            tp = tm.get("true_positives",  0)

            annotations = [
                [f"<b>TN</b><br>{tn:,}<br><i>Legit → Legit</i>",
                 f"<b>FP</b><br>{fp:,}<br><i>Legit → Phishing</i>"],
                [f"<b>FN</b><br>{fn:,}<br><i>Phish → Legit</i>",
                 f"<b>TP</b><br>{tp:,}<br><i>Phish → Phishing</i>"],
            ]

            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Predicted: Legitimate", "Predicted: Phishing"],
                y=["Actual: Legitimate",    "Actual: Phishing"],
                colorscale=[
                    [0.0, "#dbeafe"],
                    [0.5, "#93c5fd"],
                    [1.0, "#1d4ed8"],
                ],
                showscale=False,
                text=annotations,
                texttemplate="%{text}",
                hovertemplate="Count: %{z}<extra></extra>",
            ))
            fig_cm.update_layout(
                **_plotly_base_layout(
                    title=f"Confusion Matrix — {meta.get('model_name', 'Selected Model')} "
                          f"(Domain-Unseen Test Set)",
                    # Square: equal width and height
                    height=460,
                    width=460,
                    margin=dict(l=16, r=16, t=52, b=16),
                )
            )
            # Force square cells
            fig_cm.update_xaxes(
                constrain="domain",
                gridcolor="#e2e6ea",
            )
            fig_cm.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
                gridcolor="#e2e6ea",
            )

            # Centre the square chart
            col_cm, col_right = st.columns([1, 1])
            with col_cm:
                st.plotly_chart(fig_cm, use_container_width=False)

            c1, c2, c3, c4 = st.columns(4)
            _metric_card("True Negatives",  f"{tn:,}", c1, "green")
            _metric_card("False Positives", f"{fp:,}", c2, "amber")
            _metric_card("False Negatives", f"{fn:,}", c3, "red")
            _metric_card("True Positives",  f"{tp:,}", c4, "green")

            st.markdown(f"""
            <div class="disclaimer-box">
            <strong>False Negatives ({fn:,}):</strong> Phishing URLs classified as legitimate —
            the most security-sensitive error type. The interface always warns that a Legitimate
            verdict is not a safety guarantee.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Test metrics not available.")

    # ── Tab 4 : Feature Importance ──────────────────────────────────────
    with tab4:
        st.subheader("Permutation Feature Importance (Post-Evaluation)")
        if imp_df is not None:
            fig_imp = px.bar(
                imp_df.head(22).sort_values("importance_mean", ascending=True),
                x="importance_mean", y="feature", orientation="h",
                error_x="importance_std",
                color="importance_mean",
                color_continuous_scale=["#bfdbfe", "#2563eb", "#1d4ed8"],
                labels={"importance_mean": "Mean Importance (F1 drop)", "feature": "Feature"},
                title="Feature Importance — Mean Decrease in Phishing F1",
            )
            fig_imp.update_layout(
                **_plotly_base_layout(coloraxis_showscale=False, height=500)
            )
            fig_imp.update_xaxes(gridcolor="#e2e6ea")
            fig_imp.update_yaxes(gridcolor="#e2e6ea")
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption(
                "Importance = mean decrease in phishing F1 when a feature is randomly permuted "
                "(n_repeats=10). Calculated after final evaluation. "
                "Does not imply causation for individual URLs."
            )
            st.dataframe(imp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Importance data not found. Run training script to generate.")

    # ── Tab 5 : Test Errors ─────────────────────────────────────────────
    with tab5:
        st.subheader("Test Set Error Analysis")
        if (test_df is not None
                and "label" in test_df.columns
                and "predicted_label" in test_df.columns):
            fp_df = test_df[(test_df["label"] == 0) & (test_df["predicted_label"] == 1)]
            fn_df = test_df[(test_df["label"] == 1) & (test_df["predicted_label"] == 0)]

            col_fp, col_fn = st.columns(2)
            with col_fp:
                st.markdown(f"""
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;
                     padding:11px 15px;margin-bottom:10px;">
                  <span style="font-weight:700;color:#b45309;">False Positives: {len(fp_df):,}</span>
                  <div style="font-size:0.78rem;color:#64748b;margin-top:2px;">
                    Legitimate URLs incorrectly flagged as phishing
                  </div>
                </div>
                """, unsafe_allow_html=True)
                if "display_url" in fp_df.columns and len(fp_df) > 0:
                    st.dataframe(fp_df[["display_url", "phish_probability"]].head(10),
                                 use_container_width=True, hide_index=True)

            with col_fn:
                st.markdown(f"""
                <div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;
                     padding:11px 15px;margin-bottom:10px;">
                  <span style="font-weight:700;color:#dc2626;">False Negatives: {len(fn_df):,}</span>
                  <div style="font-size:0.78rem;color:#64748b;margin-top:2px;">
                    Phishing URLs missed by the model (security risk)
                  </div>
                </div>
                """, unsafe_allow_html=True)
                if "display_url" in fn_df.columns and len(fn_df) > 0:
                    st.dataframe(fn_df[["display_url", "phish_probability"]].head(10),
                                 use_container_width=True, hide_index=True)

            fig_dist = go.Figure()
            legit_probs = test_df[test_df["label"] == 0]["phish_probability"]
            phish_probs = test_df[test_df["label"] == 1]["phish_probability"]
            fig_dist.add_trace(go.Histogram(
                x=legit_probs, name="Legitimate", nbinsx=50,
                marker=dict(color="#16a34a", opacity=0.65, line=dict(width=0)),
            ))
            fig_dist.add_trace(go.Histogram(
                x=phish_probs, name="Phishing", nbinsx=50,
                marker=dict(color="#dc2626", opacity=0.65, line=dict(width=0)),
            ))
            fig_dist.update_layout(
                **_plotly_base_layout(
                    barmode="overlay", height=330,
                    title="Predicted Probability Distribution — Test Set",
                )
            )
            fig_dist.update_xaxes(title_text="Phishing Probability", gridcolor="#e2e6ea")
            fig_dist.update_yaxes(title_text="Count",                gridcolor="#e2e6ea")
            fig_dist.add_vline(
                x=0.5, line_dash="dot", line_color="#94a3b8",
                annotation_text="Threshold 0.50",
                annotation_font=dict(color="#64748b", size=11),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Test results not found. Run training script to generate.")


# ===========================================================================
# PAGE 4 – SCAN HISTORY
# ===========================================================================

def page_history():
    st.markdown("""
    <div style="margin-bottom:4px;">
      <span style="font-size:1.7rem;font-weight:800;color:#0f172a;">Scan History</span>
    </div>
    <div style="font-size:0.9rem;color:#64748b;margin-bottom:14px;">
      All scans from this browser session, stored locally.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    <strong>Privacy:</strong> Query strings, credentials and fragments are removed before storage.
    Only this browser session can view or clear its records. Cloud restarts may reset history.
    </div>
    """, unsafe_allow_html=True)

    session_id = st.session_state.phishguard_session_id
    summary    = get_summary(session_id=session_id)
    scans      = load_scans(200, session_id=session_id)

    c1, c2, c3, c4, c5 = st.columns(5)
    _metric_card("Total Scans",  str(summary.get("total",       0)), c1)
    _metric_card("Phishing",     str(summary.get("phishing",    0)), c2, "red")
    _metric_card("Legitimate",   str(summary.get("legitimate",  0)), c3, "green")
    _metric_card("High Risk",    str(summary.get("high_risk",   0)), c4, "red")
    _metric_card("Medium Risk",  str(summary.get("medium_risk", 0)), c5, "amber")

    st.markdown('<hr style="border:none;border-top:1px solid #e2e6ea;margin:16px 0;">',
                unsafe_allow_html=True)

    if not scans:
        st.markdown("""
        <div style="text-align:center;padding:44px 0;color:#94a3b8;">
          <div style="font-size:2.6rem;margin-bottom:10px;opacity:0.4;">&#9906;</div>
          <div style="font-size:0.95rem;font-weight:600;color:#64748b;margin-bottom:3px;">
            No scans recorded yet
          </div>
          <div style="font-size:0.83rem;">Use the URL Scanner to get started.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    scans_df = pd.DataFrame(scans)
    st.dataframe(
        scans_df[["timestamp", "display_url", "prediction",
                  "phish_prob", "risk_level", "model_name"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "phish_prob":   st.column_config.ProgressColumn(
                "Phishing Prob", min_value=0, max_value=1, format="%.2f"
            ),
            "display_url":  st.column_config.TextColumn("URL (redacted)", width="large"),
            "timestamp":    st.column_config.TextColumn("Timestamp",      width="medium"),
            "prediction":   st.column_config.TextColumn("Prediction"),
            "risk_level":   st.column_config.TextColumn("Risk"),
            "model_name":   st.column_config.TextColumn("Model"),
        },
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_exp, col_clr = st.columns([3, 1])

    with col_exp:
        st.download_button(
            label="Export as CSV",
            data=scans_df.to_csv(index=False).encode(),
            file_name="phishguard_scan_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_clr:
        if st.button("Clear History", type="secondary", use_container_width=True):
            clear_history(session_id=session_id)
            st.session_state.last_scan_result = None
            st.success("History cleared.")
            st.rerun()

    if summary.get("total", 0) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        col_donut, _ = st.columns([1, 1])
        with col_donut:
            total = summary.get("total", 0)
            fig_donut = go.Figure(go.Pie(
                labels=["Legitimate", "Phishing"],
                values=[summary.get("legitimate", 0), summary.get("phishing", 0)],
                hole=0.62,
                marker=dict(
                    colors=["#16a34a", "#dc2626"],
                    line=dict(color="#ffffff", width=3),
                ),
                textfont=dict(size=12, color="#1e293b"),
            ))
            fig_donut.add_annotation(
                text=f"<b>{total}</b><br><span style='font-size:11px'>scans</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#1e293b"),
            )
            fig_donut.update_layout(
                **_plotly_base_layout(
                    title="Session Scan Distribution",
                    height=300, margin=dict(l=20, r=20, t=44, b=20),
                    showlegend=True,
                    legend=dict(font=dict(color="#334155", size=12)),
                )
            )
            st.plotly_chart(fig_donut, use_container_width=True)


# ===========================================================================
# Router
# ===========================================================================

def main():
    if page == "Overview":
        page_overview()
    elif page == "URL Scanner":
        page_scanner()
    elif page == "Analytics":
        page_analytics()
    elif page == "Scan History":
        page_history()


if __name__ == "__main__":
    main()
