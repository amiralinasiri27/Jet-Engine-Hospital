"""
app.py
------
Streamlit dashboard for the Jet Engine Hospital project.
"""

import sys
import json
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import data_loading, labels
from src.decision import PrognosticsSystem, DecisionPolicy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "CMAPSSData"
ICON_PATH = Path(__file__).resolve().parent / "assets" / "mission_icon.svg"

KNOWN_DATASETS = ["FD001", "FD002", "FD003", "FD004"]


# ---------------------------------------------------------------- styling --

def inject_css():
    st.markdown("""
    <style>
    /* Global Page Settings for Single-Screen Experience */
    html, body, [data-testid="stAppViewContainer"] {
        height: 100vh !important;
        overflow: hidden !important;
        background-color: #090d16 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    [data-testid="stHeader"] {
        display: none !important;
    }

    .main .block-container {
        padding: 0.5rem 1rem !important;
        max-width: 100% !important;
        height: 100vh !important;
        display: flex;
        flex-direction: column;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1rem 0.8rem !important;
    }

    /* Custom Header Banner */
    .mh-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 6px 14px;
        margin-bottom: 8px;
    }
    .mh-title-box {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .mh-title-box h1 {
        color: #f8fafc;
        font-size: 18px;
        font-weight: 700;
        margin: 0;
    }
    .mh-title-box p {
        color: #94a3b8;
        margin: 0;
        font-size: 10px;
    }
    .mh-badge {
        background: #0284c7;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Compact Cards */
    .mh-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }
    .mh-card-header {
        font-size: 11px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    /* Action Banner Styles */
    .mh-action-card {
        border-radius: 8px;
        padding: 8px 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .mh-action-CONTINUE {
        background: linear-gradient(90deg, #064e3b 0%, #022c22 100%);
        border: 1px solid #10b981;
    }
    .mh-action-INSPECT {
        background: linear-gradient(90deg, #78350f 0%, #451a03 100%);
        border: 1px solid #f59e0b;
    }
    .mh-action-STOP {
        background: linear-gradient(90deg, #7f1d1d 0%, #450a0a 100%);
        border: 1px solid #ef4444;
    }
    .mh-action-title {
        font-size: 16px;
        font-weight: 800;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .mh-action-details {
        font-size: 11px;
        color: #cbd5e1;
        text-align: right;
    }

    /* Prominent Large Font Metrics */
    .metric-value-lg {
        font-size: 32px;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1;
        margin: 2px 0;
    }
    .metric-value-lg span {
        font-size: 14px;
        font-weight: 500;
        color: #94a3b8;
    }
    .metric-sub {
        font-size: 11px;
        color: #64748b;
    }

    /* Expander adjustments */
    .streamlit-expanderHeader {
        background-color: #0f172a !important;
        border-radius: 6px !important;
        font-size: 11px !important;
        padding: 2px 6px !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    icon_html = ""
    if ICON_PATH.exists():
        icon_b64 = base64.b64encode(ICON_PATH.read_bytes()).decode()
        icon_html = f'<img src="data:image/svg+xml;base64,{icon_b64}" width="32" height="32"/>'
    else:
        icon_html = '<span style="font-size: 24px;">🚀</span>'

    st.markdown(f"""
    <div class="mh-header">
        <div class="mh-title-box">
            {icon_html}
            <div>
                <h1>Jet Engine Hospital</h1>
                <p>NASA C-MAPSS Turbofan Prognostics & Health Management</p>
            </div>
        </div>
        <div class="mh-badge">LIVE DIAGNOSTICS</div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------ artifact I/O --

def _artifacts_ready(dataset: str) -> bool:
    art_dir = ARTIFACTS_DIR / dataset
    required = ["feature_pipeline.pkl", "rul_suite.pkl", "clf_suite.pkl",
                "anomaly_suite.pkl", "decision_policy.json", "metadata.json"]
    return all((art_dir / f).exists() for f in required)


@st.cache_resource
def load_artifacts(dataset: str):
    art_dir = ARTIFACTS_DIR / dataset
    pipeline = joblib.load(art_dir / "feature_pipeline.pkl")
    rul_suite = joblib.load(art_dir / "rul_suite.pkl")
    clf_suite = joblib.load(art_dir / "clf_suite.pkl")
    anomaly_suite = joblib.load(art_dir / "anomaly_suite.pkl")
    with open(art_dir / "decision_policy.json") as f:
        policy_dict = json.load(f)
    policy = DecisionPolicy(**policy_dict["policy"])
    with open(art_dir / "metadata.json") as f:
        metadata = json.load(f)

    system = PrognosticsSystem(
        feature_pipeline=pipeline,
        rul_suite=rul_suite, rul_model_name=metadata["chosen_rul_model"],
        clf_suite=clf_suite, clf_model_name=metadata["chosen_clf_model"],
        anomaly_suite=anomaly_suite, anomaly_detector_name=metadata["chosen_anomaly_detector"],
        policy=policy, rul_interval_std=metadata.get("rul_interval_std"),
    )
    return system, metadata


@st.cache_data
def load_test_data(dataset: str):
    data = data_loading.load_dataset(dataset, DATA_DIR)
    rul_final = data["rul_final"]
    test_labeled = labels.add_test_rul(data["test"], rul_final)
    return test_labeled


@st.cache_data
def compute_engine_features(dataset: str, engine_id: int):
    system, _ = load_artifacts(dataset)
    test_labeled = load_test_data(dataset)
    engine_hist = test_labeled[test_labeled["engine_id"] == engine_id].sort_values("cycle").reset_index(drop=True)
    feat_full = system.feature_pipeline.transform_engine(engine_hist)
    return engine_hist, feat_full


# ------------------------------------------------------------- rendering --

def render_engine_timeline(engine_hist: pd.DataFrame, current_cycle: int, sensor_col: str):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6.5, 2.2), dpi=100)
    fig.patch.set_alpha(0)
    ax.set_facecolor("#0f172a")
    
    ax.grid(True, linestyle=":", alpha=0.25, color="#64748b")
    ax.plot(engine_hist["cycle"], engine_hist[sensor_col], color="#38bdf8", linewidth=1.8, label=sensor_col)
    ax.axvline(current_cycle, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Cycle {current_cycle}")
    
    curr_val = engine_hist.loc[engine_hist["cycle"] == current_cycle, sensor_col].values
    if len(curr_val) > 0:
        ax.scatter([current_cycle], [curr_val[0]], color="#ef4444", s=40, zorder=5)

    ax.set_xlabel("Cycle", color="#94a3b8", fontsize=8)
    ax.set_ylabel(sensor_col, color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7)
    
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
        
    ax.legend(facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#e2e8f0", fontsize=7, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close(fig)


def render_action_card(recommendation: dict):
    act = recommendation["action"]
    badge_color = {"CONTINUE": "✅", "INSPECT": "🔎", "STOP": "🛑"}[act]
    
    st.markdown(f"""
    <div class="mh-action-card mh-action-{act}">
        <div class="mh-action-title">
            <span>{badge_color}</span>
            <span>SYSTEM ACTION: {act}</span>
        </div>
        <div class="mh-action-details">
            <div><b>Trigger:</b> {recommendation['trigger_reason']}</div>
            <div><b>Confidence:</b> {recommendation['confidence']:.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_rul_card(rul_output: dict):
    st.markdown("""
    <div class="mh-card">
        <div class="mh-card-header">🛞 Remaining Useful Life (RUL)</div>
    """, unsafe_allow_html=True)
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown(f"""
        <div>
            <div class="metric-value-lg">{rul_output['point']:.0f} <span>cycles</span></div>
            <div class="metric-sub">95% CI: [{rul_output['lower']:.0f}, {rul_output['upper']:.0f}]</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        pct = min(1.0, max(0.0, rul_output['point'] / 150.0))
        st.write("")
        st.progress(pct)
    st.markdown("</div>", unsafe_allow_html=True)


def render_failure_risk_card(risk_output: dict):
    st.markdown("""
    <div class="mh-card">
        <div class="mh-card-header">⚠️ Failure Risk Horizon</div>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(risk_output))
    for idx, (h, proba) in enumerate(risk_output.items()):
        with cols[idx]:
            color = "#ef4444" if proba > 0.5 else ("#f59e0b" if proba > 0.2 else "#10b981")
            st.markdown(f"""
            <div style="text-align: center; background: #090d16; padding: 6px; border-radius: 6px; border: 1px solid #1e293b;">
                <div style="font-size: 9px; color: #94a3b8;">&le; {h} Cycles</div>
                <div style="font-size: 18px; font-weight: 800; color: {color};">{proba:.1%}</div>
            </div>
            """, unsafe_allow_html=True)


def render_anomaly_card(anomaly_output: dict):
    st.markdown("""
    <div class="mh-card">
        <div class="mh-card-header">📡 Anomaly Score & Percentile</div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown(f"""
        <div>
            <div class="metric-value-lg">{anomaly_output['percentile']:.0f}<span>th pct</span></div>
            <div class="metric-sub">Raw Score: {anomaly_output['raw']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.write("")
        st.progress(min(1.0, max(0.0, anomaly_output["percentile"] / 100.0)))
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------- main --

def main():
    st.set_page_config(
        page_title="Jet Engine Hospital",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()
    render_header()

    available = [d for d in KNOWN_DATASETS if _artifacts_ready(d)]
    missing = [d for d in KNOWN_DATASETS if d not in available]
    if not available:
        st.error(
            "No trained artifacts found under artifacts/<DATASET>/. Run the notebook "
            "(run_stage(...) for each dataset) first."
        )
        return

    st.sidebar.markdown("<h3 style='color: #f8fafc; font-size: 15px; margin-bottom: 8px;'>🎛️ Control Panel</h3>", unsafe_allow_html=True)
    if missing:
        st.sidebar.caption(f"Unavailable: {', '.join(missing)}")

    dataset = st.sidebar.selectbox("Dataset", available)
    system, metadata = load_artifacts(dataset)
    test_labeled = load_test_data(dataset)

    engine_ids = sorted(test_labeled["engine_id"].unique())
    engine_id = st.sidebar.selectbox("Engine ID", engine_ids)

    engine_hist, feat_full = compute_engine_features(dataset, engine_id)

    # Correct Cycle Slider Direction (Min -> Max)
    min_cycle = int(engine_hist["cycle"].min())
    max_cycle = int(engine_hist["cycle"].max())
    
    cycle = st.sidebar.slider(
        "Current Cycle",
        min_value=min_cycle,
        max_value=max_cycle,
        value=max_cycle,
        step=1
    )
    
    sensor_cols = [c for c in engine_hist.columns if c.startswith("sensor_")]
    sensor_col = st.sidebar.selectbox("Sensor Parameter", sensor_cols)

    # Automatically compute results based on current selections
    feat_row = feat_full[feat_full["cycle"] == cycle]
    feat_row_X = feat_row[system.feature_pipeline.feature_cols_]

    rul_out = system.predict_rul(feat_row_X)
    risk_out = system.failure_risk(feat_row_X)
    anomaly_out = system.anomaly_score(feat_row_X)
    rec = system.recommend({"rul": rul_out, "failure_risk": risk_out, "anomaly": anomaly_out})

    # Render Main Dashboard Panels
    render_action_card(rec)

    # 2-Column Dashboard Layout
    col_left, col_right = st.columns([1.4, 1], gap="small")

    with col_left:
        st.markdown("<div style='font-size: 12px; font-weight:600; color:#e2e8f0; margin-bottom: 2px;'>📈 Sensor History & Cycle Timeline</div>", unsafe_allow_html=True)
        render_engine_timeline(engine_hist, cycle, sensor_col)

    with col_right:
        render_rul_card(rul_out)
        render_failure_risk_card(risk_out)
        render_anomaly_card(anomaly_out)

    with st.expander("ℹ️ Model & System Metadata", expanded=False):
        st.json(metadata)


if __name__ == "__main__":
    main()