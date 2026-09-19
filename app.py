"""
AI-Powered Digital Twin for Sustainable Textile Manufacturing Process Intelligence.
Streamlit Application integrating real Kaggle data, Machine Learning, SimPy Digital Twin,
What-If Scenario Simulation, Sustainability Estimation, Eco Score, Bottleneck Detection,
and Gemini 3.6 Flash Process Intelligence.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Auto-download datasets if data/ folder is empty (cloud deployment)
from setup_data import ensure_data_exists
ensure_data_exists()

from src.data_preprocessing import (
    get_available_datasets,
    load_data,
    clean_and_preprocess,
    get_dataset_summary,
)
from src.ml_model import TextileMLSystem
from src.digital_twin import TextileFactoryDigitalTwin
from src.sustainability import calculate_sustainability_metrics
from src.recommendations import detect_bottlenecks, generate_recommendations
from src.gemini_assistant import query_gemini_process_intelligence

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Textile Digital Twin & Eco-Intelligence",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS with Glassmorphism, Animations, Google Fonts ───
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .stApp {
        background: linear-gradient(160deg, #0a0f1e 0%, #0d1528 40%, #101d35 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1222 0%, #111b30 50%, #0d1528 100%) !important;
        border-right: 1px solid rgba(99,130,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #c8d6e5 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #8fa8c8 !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }

    /* ── Metric Cards ── */
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #60a5fa, #a78bfa) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    div[data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #8fa8c8 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(99, 130, 255, 0.12) !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255,255,255,0.04) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(99, 130, 255, 0.3) !important;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.12), inset 0 1px 0 rgba(255,255,255,0.06) !important;
        transform: translateY(-2px) !important;
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 6px;
        border: 1px solid rgba(99, 130, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: #8fa8c8 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 10px 18px !important;
        transition: all 0.25s ease !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(167,139,250,0.12)) !important;
        color: #e0e7ff !important;
        border-bottom: none !important;
        box-shadow: 0 2px 12px rgba(59,130,246,0.15) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #c7d2fe !important;
        background: rgba(59,130,246,0.08) !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.5) !important;
        border-radius: 10px !important;
        color: #c8d6e5 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(99,130,255,0.08) !important;
    }

    /* ── DataFrame ── */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(99,130,255,0.1) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.35) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
    }

    /* ── Text Colors ── */
    h1, h2, h3, h4 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }
    p, span, label, .stMarkdown {
        color: #c8d6e5 !important;
    }
    .stSubheader {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    /* ── Alerts / Info ── */
    .stAlert, div[data-testid="stAlert"] {
        background: rgba(15, 23, 42, 0.5) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(99,130,255,0.12) !important;
        border-radius: 10px !important;
        color: #c8d6e5 !important;
    }

    /* ── Custom Components ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 35%, #1a365d 60%, #0d4040 100%);
        padding: 32px 36px;
        border-radius: 20px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.06);
        border: 1px solid rgba(99, 130, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99,130,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -20%;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(16,185,129,0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        margin: 0;
        font-size: 30px;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 50%, #86efac 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        margin: 8px 0 0 0;
        opacity: 0.75;
        font-size: 14px;
        color: #94a3b8;
        font-weight: 400;
        letter-spacing: 0.3px;
        position: relative;
        z-index: 1;
    }

    /* ── Pipeline ── */
    .pipeline-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(16px);
        padding: 20px 24px;
        border-radius: 16px;
        border: 1px solid rgba(99,130,255,0.08);
        margin-bottom: 24px;
        overflow-x: auto;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .pipeline-step {
        text-align: center;
        padding: 12px 18px;
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        border: 1px solid rgba(99,130,255,0.12);
        font-weight: 600;
        color: #e2e8f0;
        min-width: 130px;
        font-size: 13px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .pipeline-step:hover {
        border-color: rgba(99,130,255,0.3);
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(59,130,246,0.15);
    }
    .pipeline-arrow {
        font-size: 20px;
        color: #4f6b8a;
        font-weight: bold;
        padding: 0 4px;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 22px;
        border-radius: 16px;
        border: 1px solid rgba(99, 130, 255, 0.1);
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12), inset 0 1px 0 rgba(255,255,255,0.03);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(99, 130, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.18), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    /* ── Eco Badge ── */
    .eco-badge {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 6px 14px;
        border-radius: 20px;
        color: #6ee7b7 !important;
        font-weight: 600;
        display: inline-block;
        font-size: 13px;
    }

    /* ── Bottleneck Box ── */
    .bottleneck-box {
        background: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        padding: 18px 22px;
        border-radius: 12px;
        margin-bottom: 18px;
        border: 1px solid rgba(245, 158, 11, 0.15);
        backdrop-filter: blur(8px);
    }
    .bottleneck-box h3 { color: #fbbf24 !important; }
    .bottleneck-box p { color: #fcd34d !important; }

    /* ── Recommendation Cards ── */
    .rec-card {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(12px);
        padding: 20px 24px;
        border-radius: 14px;
        border: 1px solid rgba(99, 130, 255, 0.1);
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .rec-card:hover {
        transform: translateX(4px);
        border-color: rgba(99, 130, 255, 0.2);
    }

    /* ── Priority Badges ── */
    .priority-high {
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5 !important;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .priority-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #fcd34d !important;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .priority-low {
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7 !important;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Stat Pill ── */
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(8px);
        padding: 10px 16px;
        border-radius: 12px;
        border: 1px solid rgba(99,130,255,0.1);
        margin: 4px;
        font-size: 14px;
        color: #c8d6e5;
    }
    .stat-pill strong {
        color: #60a5fa !important;
    }

    /* ── Section Header ── */
    .section-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b !important;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99,130,255,0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99,130,255,0.35); }

    /* ── Plotly Chart Container ── */
    .stPlotlyChart {
        border-radius: 14px !important;
        overflow: hidden !important;
    }

    /* ── Text Input ── */
    .stTextInput input, .stNumberInput input {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(99,130,255,0.15) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: rgba(99,130,255,0.4) !important;
        box-shadow: 0 0 0 2px rgba(99,130,255,0.1) !important;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Plotly dark theme template for all charts
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(family="Inter, sans-serif", color="#c8d6e5"),
    colorway=["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#38bdf8", "#c084fc", "#4ade80"],
)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 4px;">
            <div style="font-size: 42px; margin-bottom: 2px;">🧵</div>
            <div style="font-size: 18px; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #a78bfa);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                Factory Control
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # 1. Dataset Selection
    st.markdown('<p class="section-label">📁 Dataset Source</p>', unsafe_allow_html=True)
    avail_datasets = get_available_datasets()
    if avail_datasets:
        selected_file = st.selectbox(
            "Select CSV Dataset:",
            avail_datasets,
            index=avail_datasets.index("weaving_dataset_full.csv") if "weaving_dataset_full.csv" in avail_datasets else 0,
            help="Real dataset from Mendeley stored in data/",
        )
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", selected_file
        )
    else:
        st.warning("No CSV found in data/ folder.")
        data_path = None

    uploaded_file = st.file_uploader(
        "Or Upload Custom CSV:", type=["csv"], help="Upload any textile manufacturing CSV"
    )
    if uploaded_file is not None:
        data_path = uploaded_file

    st.markdown("---")

    # 2. Gemini 3.6 Flash Frontend API Key
    st.markdown('<p class="section-label">⚡ Gemini 3.6 Flash AI</p>', unsafe_allow_html=True)
    gemini_key_input = st.text_input(
        "Gemini API Key:",
        type="password",
        value=st.session_state.get("gemini_api_key", ""),
        help="Enter your Google Gemini API Key for gemini-3.6-flash.",
    )
    if gemini_key_input:
        st.session_state["gemini_api_key"] = gemini_key_input
        st.caption("✅ Gemini 3.6 Flash Key active")
    else:
        st.caption("ℹ️ Offline expert mode active")

    st.markdown("---")

    # 3. Digital Twin Simulation Controls
    st.markdown('<p class="section-label">⚙️ SimPy Simulation</p>', unsafe_allow_html=True)
    sim_duration = st.slider("Simulation Horizon (Hours):", 12, 168, 48, step=12)
    batch_size = st.number_input("Batch Size (Yards):", 100, 5000, 500, step=100)
    arrival_interval = st.slider("Arrival Interval (Hrs):", 0.5, 5.0, 1.2, step=0.1)

    with st.expander("🛠️ Machine Config"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            warping_mc = st.number_input("Warping M/C:", 1, 10, 2)
            warping_time = st.number_input("Warping Time:", 0.1, 10.0, 1.2, step=0.1)
            weaving_mc = st.number_input("Weaving Looms:", 1, 30, 6)
            weaving_time = st.number_input("Weaving Time:", 0.5, 20.0, 4.0, step=0.5)
        with col_m2:
            sizing_mc = st.number_input("Sizing M/C:", 1, 10, 2)
            sizing_time = st.number_input("Sizing Time:", 0.1, 10.0, 1.5, step=0.1)
            inspection_mc = st.number_input("Inspect M/C:", 1, 10, 2)
            inspection_time = st.number_input("Inspect Time:", 0.1, 5.0, 0.8, step=0.1)

    defect_probability = st.slider("Defect Risk Factor:", 0.01, 0.30, 0.08, step=0.01)

# ---------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_processed_data(file_source):
    df_raw = load_data(file_source)
    df_clean = clean_and_preprocess(df_raw)
    return df_raw, df_clean

try:
    df_raw, df_clean = get_processed_data(data_path)
    summary_info = get_dataset_summary(df_clean)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ---------------------------------------------------------
# Run SimPy Digital Twin Simulation & Sustainability
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_digital_twin_simulation(
    sim_duration, batch_size, arrival_interval,
    warping_mc, warping_time, sizing_mc, sizing_time,
    weaving_mc, weaving_time, inspection_mc, inspection_time,
    defect_probability,
):
    dt = TextileFactoryDigitalTwin(
        sim_duration_hours=sim_duration,
        batch_size_yds=batch_size,
        arrival_interval_hours=arrival_interval,
        warping_machines=warping_mc,
        warping_time_hours=warping_time,
        sizing_machines=sizing_mc,
        sizing_time_hours=sizing_time,
        weaving_machines=weaving_mc,
        weaving_time_hours=weaving_time,
        inspection_machines=inspection_mc,
        inspection_time_hours=inspection_time,
        defect_prob=defect_probability,
    )
    sim_res = dt.run()
    sust_res = calculate_sustainability_metrics(sim_res)
    return sim_res, sust_res

sim_results, sust_results = run_digital_twin_simulation(
    sim_duration, batch_size, arrival_interval,
    warping_mc, warping_time, sizing_mc, sizing_time,
    weaving_mc, weaving_time, inspection_mc, inspection_time,
    defect_probability,
)

bottleneck_info = detect_bottlenecks(sim_results)
recommendations = generate_recommendations(sim_results, sust_results)

# ---------------------------------------------------------
# Hero Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1 class="hero-title">🧵 AI-Powered Digital Twin for Sustainable Textile Manufacturing</h1>
        <p class="hero-subtitle">Real-time Process Intelligence &nbsp;•&nbsp; SimPy Discrete-Event Simulation &nbsp;•&nbsp; Machine Learning &nbsp;•&nbsp; Sustainability & Eco Score</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Process Pipeline
st.markdown(
    """
    <div class="pipeline-container">
        <div class="pipeline-step">📦 Raw Material<br><span style="font-size:11px; color:#64748B;">Yarn Packages</span></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">🌀 Warping<br><span style="font-size:11px; color:#64748B;">Beam Prep</span></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">🧪 Sizing<br><span style="font-size:11px; color:#64748B;">Protective Coat</span></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">⚙️ Weaving<br><span style="font-size:11px; color:#64748B;">Loom Formation</span></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">🔍 Inspection<br><span style="font-size:11px; color:#64748B;">Defect Sorting</span></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step" style="border-color: rgba(16,185,129,0.4); box-shadow: 0 0 12px rgba(16,185,129,0.1);">✅ Finished<br><span style="font-size:11px; color:#6ee7b7;">Graded Rolls</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top KPI Metrics
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric(
        label="Total Production",
        value=f"{sim_results['completed_yds']:,.0f} yds",
        delta=f"{sim_results['throughput_yds_per_hr']:.1f} yds/hr",
    )
with kpi2:
    st.metric(
        label="Rejection Rate",
        value=f"{sim_results['rejection_rate_pct']:.1f}%",
        delta=f"-{sim_results['rejected_yds']:,.0f} yds scrap",
        delta_color="inverse",
    )
with kpi3:
    st.metric(
        label="Estimated Energy",
        value=f"{sust_results['total_energy_kwh']:,.1f} kWh",
        delta=f"{sust_results['energy_per_yard_kwh']:.2f} kWh/yd",
        delta_color="inverse",
    )
with kpi4:
    st.metric(
        label="Material Waste",
        value=f"{sust_results['total_waste_kg']:,.1f} kg",
        delta=f"{sust_results['waste_per_yard_kg']:.3f} kg/yd",
        delta_color="inverse",
    )
with kpi5:
    eco_val = sust_results["eco_score"]
    st.metric(
        label="Eco Score",
        value=f"{eco_val:.0f} / 100",
        delta="Optimal" if eco_val >= 75 else "Needs Improvement",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------
tab_eda, tab_ml, tab_twin, tab_whatif, tab_sust, tab_recs, tab_ai = st.tabs(
    [
        "📊 Dataset Overview",
        "🤖 ML Prediction",
        "🏭 Digital Twin",
        "🔄 What-If Sim",
        "🌿 Sustainability",
        "⚡ Bottlenecks",
        "💬 AI Copilot",
    ]
)

# =========================================================
# TAB 1: DATASET OVERVIEW & EDA
# =========================================================
with tab_eda:
    st.subheader("📊 Textile Weaving Dataset Exploration")

    # Stats pills
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:20px;">
            <div class="stat-pill">📋 <strong>{summary_info['rows']:,}</strong> Records</div>
            <div class="stat-pill">📐 <strong>{summary_info['cols']}</strong> Columns</div>
            <div class="stat-pill">🔧 <strong>{summary_info['missing_cells']:,}</strong> Missing Handled</div>
            <div class="stat-pill">🗑️ <strong>{summary_info.get('preprocessing_summary', {}).get('duplicates_removed', 0):,}</strong> Duplicates Removed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🔍 Dataset Preview")
    st.dataframe(df_clean.head(10), use_container_width=True)

    prep = summary_info.get("preprocessing_summary", {})
    with st.expander("Preprocessing audit trail"):
        st.write(
            f"{prep.get('original_rows', len(df_raw)):,} original rows → "
            f"{prep.get('aggregate_rows_removed', 0):,} aggregate rows removed → "
            f"{prep.get('duplicates_removed', 0):,} exact duplicates removed → "
            f"{prep.get('final_rows', len(df_clean)):,} final analytical rows."
        )
        st.caption(
            f"Index columns removed: {prep.get('index_columns_removed', 0)}. "
            f"Missing sentinels normalized: {prep.get('missing_values_normalized', 0):,}; "
            f"invalid rows removed: {prep.get('invalid_rows_removed', 0):,}."
        )

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("#### 🧵 EPI vs PPI Distribution")
        if "epi" in df_clean.columns and "ppi" in df_clean.columns:
            color_var = "Has_Rejection" if "Has_Rejection" in df_clean.columns else None
            fig_scatter = px.scatter(
                df_clean.sample(min(1500, len(df_clean))),
                x="epi", y="ppi", color=color_var,
                color_continuous_scale="Viridis",
                labels={"epi": "Ends Per Inch (EPI)", "ppi": "Picks Per Inch (PPI)"},
                title="EPI vs PPI (Colored by Rejection)",
            )
            fig_scatter.update_layout(**PLOTLY_THEME, height=380)
            st.plotly_chart(fig_scatter, use_container_width=True)

    with col_c2:
        st.markdown("#### 📉 Production Distribution")
        if "Total_Pdn(yds)" in df_clean.columns and "Rejection_Qty" in df_clean.columns:
            fig_hist = px.histogram(
                df_clean, x="Total_Pdn(yds)", nbins=40,
                color="Has_Rejection" if "Has_Rejection" in df_clean.columns else None,
                title="Total Production Yardage Distribution",
                labels={"Total_Pdn(yds)": "Production Output (yds)"},
            )
            fig_hist.update_layout(**PLOTLY_THEME, height=380)
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### 🌡️ Feature Correlation Heatmap")
    num_df = df_clean.select_dtypes(include=[np.number])
    top_num_cols = [c for c in num_df.columns if not c.startswith("Unnamed")][:10]
    if len(top_num_cols) > 1:
        corr_matrix = num_df[top_num_cols].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f",
            color_continuous_scale="Blues",
            title="Feature Correlation Matrix",
        )
        fig_corr.update_layout(**PLOTLY_THEME, height=440)
        st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================
# TAB 2: MACHINE LEARNING PREDICTION
# =========================================================
with tab_ml:
    st.subheader("🤖 Scikit-Learn Predictive Model")
    st.markdown("Trains **RandomForestClassifier** (rejection) and **RandomForestRegressor** (production) on the real dataset.")

    if (
        "ml_system" not in st.session_state
        or not hasattr(st.session_state["ml_system"], "dataset_fingerprint")
        or not hasattr(st.session_state["ml_system"], "metadata")
    ):
        st.session_state["ml_system"] = TextileMLSystem()

    ml_sys = st.session_state["ml_system"]
    dataset_label = selected_file if avail_datasets else getattr(uploaded_file, "name", "Custom CSV")
    current_fingerprint = TextileMLSystem.dataset_fingerprint(df_clean)
    if (
        ml_sys.metadata.get("dataset_fingerprint") != current_fingerprint
        or ml_sys.metadata.get("dataset_name") != dataset_label
    ):
        ml_sys = TextileMLSystem()
        st.session_state["ml_system"] = ml_sys

    col_btn, col_st = st.columns([1, 3])
    training_error = None
    with col_btn:
        if st.button("🚀 Train / Re-train Models", type="primary", use_container_width=True):
            with st.spinner("Training on real dataset..."):
                try:
                    ml_sys.train_models(df_clean, dataset_name=dataset_label)
                    st.success("Models trained!")
                except ValueError as exc:
                    ml_sys.is_trained = False
                    training_error = str(exc)

    if not ml_sys.is_trained:
        with st.spinner("Initializing ML model..."):
            try:
                ml_sys.train_models(df_clean, dataset_name=dataset_label)
            except ValueError as exc:
                ml_sys.is_trained = False
                training_error = str(exc)

    if training_error:
        st.warning(f"Model training is unavailable for this dataset: {training_error}")

    st.caption(
        f"Model dataset: {ml_sys.metadata.get('dataset_name', dataset_label)} | "
        f"Trained: {ml_sys.metadata.get('training_timestamp_utc', 'current run')} | "
        f"Preprocessing: {ml_sys.metadata.get('preprocessing_version', 'unknown')}"
    )

    st.markdown("---")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("### 🎯 Classification — Rejection Risk")
        cm_metrics = ml_sys.clf_metrics
        if cm_metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{cm_metrics['accuracy'] * 100:.1f}%")
            m2.metric("Precision", f"{cm_metrics['precision'] * 100:.1f}%")
            m3.metric("Recall", f"{cm_metrics['recall'] * 100:.1f}%")
            m4.metric("F1-Score", f"{cm_metrics['f1'] * 100:.1f}%")

            cm = np.array(cm_metrics["confusion_matrix"])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Samples"),
                x=["Accept", "Reject"], y=["Accept", "Reject"],
                color_continuous_scale=[[0, "#0f172a"], [0.5, "#3b82f6"], [1, "#a78bfa"]],
                title="Confusion Matrix",
            )
            fig_cm.update_layout(**PLOTLY_THEME, height=340)
            st.plotly_chart(fig_cm, use_container_width=True)

    with col_m2:
        st.markdown("### 📈 Regression — Production Volume")
        rm_metrics = ml_sys.reg_metrics
        if rm_metrics:
            r1, r2, r3 = st.columns(3)
            r1.metric("R² Score", f"{rm_metrics['r2']:.3f}")
            r2.metric("MAE", f"{rm_metrics['mae']:.1f} yds")
            r3.metric("RMSE", f"{rm_metrics['rmse']:.1f} yds")

        if ml_sys.feature_importances_clf:
            feat_df = pd.DataFrame(
                list(ml_sys.feature_importances_clf.items()),
                columns=["Feature", "Importance"],
            ).head(8)
            fig_feat = px.bar(
                feat_df, x="Importance", y="Feature", orientation="h",
                color="Importance",
                color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#a78bfa"]],
                title="Top Features — Rejection Risk",
            )
            fig_feat.update_layout(**PLOTLY_THEME, yaxis=dict(autorange="reversed"), height=340)
            st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Live Rejection & Production Predictor")
    st.markdown("Enter custom parameters for real-time ML risk prediction:")

    pred_c1, pred_c2, pred_c3, pred_c4 = st.columns(4)
    with pred_c1:
        inp_epi = st.number_input("EPI:", 50, 200, 110)
        inp_ppi = st.number_input("PPI:", 40, 150, 80)
    with pred_c2:
        inp_w_cnt = st.number_input("Warp Count (Ne):", 10, 100, 40)
        inp_weft_cnt = st.number_input("Weft Count (Ne):", 10, 100, 40)
    with pred_c3:
        inp_fab_allow = st.number_input("Fabric Allow (%):", 0.0, 20.0, 5.0, step=0.5)
        inp_shrink = st.number_input("Shrinkage (%):", 0.0, 15.0, 3.0, step=0.5)
    with pred_c4:
        inp_req_finish = st.number_input("Req. Finish (yds):", 500.0, 50000.0, 10000.0, step=500.0)
        inp_rec_beam = st.number_input("Beam Length (yds):", 500.0, 60000.0, 10500.0, step=500.0)

    input_sample = {
        "epi": inp_epi, "ppi": inp_ppi,
        "weft_count": inp_weft_cnt, "warp_count_num": inp_w_cnt,
        "Fabric_Allowance": inp_fab_allow, "Shrink_allow": inp_shrink,
        "Req_Finish_Fabrics": inp_req_finish,
        "Rec_Beam_length(yds)": inp_rec_beam,
        "Req_beam_length(yds)": inp_rec_beam,
        "Req_grey_fabric": inp_req_finish * (1 + inp_shrink / 100.0),
        "warp_cover_factor": inp_epi / np.sqrt(max(1.0, inp_w_cnt)),
        "weft_cover_factor": inp_ppi / np.sqrt(max(1.0, inp_weft_cnt)),
        "total_fabric_density": inp_epi + inp_ppi,
    }

    pred_res = (
        ml_sys.predict_single(input_sample)
        if ml_sys.is_trained
        else {"error": "Trainable model unavailable for the selected dataset."}
    )
    if "rejection_probability" in pred_res:
        p1, p2, p3 = st.columns(3)
        p1.metric("Rejection Probability", f"{pred_res['rejection_probability']}%")
        p2.metric("Risk Classification", pred_res['risk_tier'])
        if "predicted_production_yds" in pred_res:
            p3.metric("Est. Production", f"{pred_res['predicted_production_yds']:,.0f} yds")

    recommendations = generate_recommendations(
        sim_results,
        sust_results,
        ml_context=pred_res,
        feature_importances=ml_sys.feature_importances_clf,
    )

# =========================================================
# TAB 3: SIMPY DIGITAL TWIN
# =========================================================
with tab_twin:
    st.subheader("🏭 SimPy Discrete-Event Digital Twin")
    st.markdown(f"Simulates continuous manufacturing flow over **{sim_duration} hours** across all factory stages.")

    twin_c1, twin_c2 = st.columns(2)

    with twin_c1:
        st.markdown("#### 📊 Machine Utilization (%)")
        util_df = pd.DataFrame([
            {"Stage": k, "Utilization (%)": v["utilization_pct"],
             "Machines": v["machines"], "Batches": v["processed_count"]}
            for k, v in sim_results["stage_metrics"].items()
        ])
        fig_util = px.bar(
            util_df, x="Stage", y="Utilization (%)",
            color="Utilization (%)", text="Utilization (%)",
            color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#ef4444"]],
            title="Stage Utilization",
        )
        fig_util.add_hline(y=85, line_dash="dash", line_color="#ef4444",
                          annotation_text="Congestion Threshold (85%)",
                          annotation_font_color="#ef4444")
        fig_util.update_layout(**PLOTLY_THEME, height=380)
        st.plotly_chart(fig_util, use_container_width=True)

    with twin_c2:
        st.markdown("#### ⏳ Queue Wait Times (Hours)")
        wait_df = pd.DataFrame([
            {"Stage": k, "Avg Wait": v["avg_wait_hours"], "Max Wait": v["max_wait_hours"]}
            for k, v in sim_results["stage_metrics"].items()
        ])
        fig_wait = px.bar(
            wait_df, x="Stage", y=["Avg Wait", "Max Wait"], barmode="group",
            title="Stage Queue Delay",
        )
        fig_wait.update_layout(**PLOTLY_THEME, height=380)
        st.plotly_chart(fig_wait, use_container_width=True)

    st.markdown("#### 📜 Simulation Event Log")
    if not sim_results["event_log_df"].empty:
        st.dataframe(sim_results["event_log_df"].head(15), use_container_width=True)

# =========================================================
# TAB 4: WHAT-IF SCENARIO
# =========================================================
with tab_whatif:
    st.subheader("🔄 What-If Scenario Comparison")
    st.markdown("Simulate operational changes and compare against baseline.")

    with st.expander("⚙️ What-If Parameters", expanded=True):
        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            wi_weaving_mc = st.number_input("Weaving Looms:", 1, 40, weaving_mc + 2, key="wi_w_mc")
        with wc2:
            wi_weaving_time = st.number_input("Weaving Time (hrs):", 0.5, 10.0, max(0.5, weaving_time - 0.5), step=0.1, key="wi_w_time")
        with wc3:
            wi_defect_prob = st.slider("Defect Factor:", 0.01, 0.20, max(0.02, defect_probability - 0.03), step=0.01, key="wi_def")
        with wc4:
            wi_sizing_mc = st.number_input("Sizing M/C:", 1, 10, sizing_mc + 1, key="wi_sz_mc")

    wi_sim_res, wi_sust_res = run_digital_twin_simulation(
        sim_duration, batch_size, arrival_interval,
        warping_mc, warping_time, wi_sizing_mc, sizing_time,
        wi_weaving_mc, wi_weaving_time, inspection_mc, inspection_time,
        wi_defect_prob,
    )

    st.markdown("### ⚖️ Scenario Impact")
    delta_prod = wi_sim_res["completed_yds"] - sim_results["completed_yds"]
    delta_prod_pct = (delta_prod / max(1.0, sim_results["completed_yds"])) * 100.0
    delta_rej = wi_sim_res["rejection_rate_pct"] - sim_results["rejection_rate_pct"]
    delta_energy = wi_sust_res["total_energy_kwh"] - sust_results["total_energy_kwh"]
    delta_waste = wi_sust_res["total_waste_kg"] - sust_results["total_waste_kg"]
    delta_eco = wi_sust_res["eco_score"] - sust_results["eco_score"]

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Production", f"{wi_sim_res['completed_yds']:,.0f} yds", f"{delta_prod:+,.0f} ({delta_prod_pct:+.1f}%)")
    d2.metric("Rejection", f"{wi_sim_res['rejection_rate_pct']:.1f}%", f"{delta_rej:+.1f}%", delta_color="inverse")
    d3.metric("Energy", f"{wi_sust_res['total_energy_kwh']:,.1f} kWh", f"{delta_energy:+,.1f}", delta_color="inverse")
    d4.metric("Waste", f"{wi_sust_res['total_waste_kg']:,.1f} kg", f"{delta_waste:+,.1f}", delta_color="inverse")
    d5.metric("Eco Score", f"{wi_sust_res['eco_score']:.0f}/100", f"{delta_eco:+.1f} pts")

    comp_df = pd.DataFrame([
        {"Metric": "Output (yds/100)", "Baseline": sim_results["completed_yds"]/100, "What-If": wi_sim_res["completed_yds"]/100},
        {"Metric": "Rejection (%)", "Baseline": sim_results["rejection_rate_pct"], "What-If": wi_sim_res["rejection_rate_pct"]},
        {"Metric": "Energy (kWh/yd×10)", "Baseline": sust_results["energy_per_yard_kwh"]*10, "What-If": wi_sust_res["energy_per_yard_kwh"]*10},
        {"Metric": "Eco Score", "Baseline": sust_results["eco_score"], "What-If": wi_sust_res["eco_score"]},
    ])
    fig_comp = px.bar(
        comp_df, x="Metric", y=["Baseline", "What-If"], barmode="group",
        title="Baseline vs What-If",
    )
    fig_comp.update_layout(**PLOTLY_THEME, height=400)
    st.plotly_chart(fig_comp, use_container_width=True)

# =========================================================
# TAB 5: SUSTAINABILITY & ECO SCORE
# =========================================================
with tab_sust:
    st.subheader("🌿 Sustainability & Eco Score")
    st.markdown(
        "> ℹ️ Environmental metrics are transparent **Estimated Indicators** calculated from machine ratings, simulation runtime, and material scrap."
    )

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("### 🏆 Eco Score")
        eco_val = sust_results["eco_score"]
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=eco_val,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Project Eco Score", "font": {"size": 18, "color": "#c8d6e5"}},
                number={"font": {"color": "#60a5fa", "size": 48}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155"},
                    "bar": {"color": "#6366f1"},
                    "bgcolor": "#0f172a",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "rgba(239,68,68,0.15)"},
                        {"range": [50, 75], "color": "rgba(245,158,11,0.12)"},
                        {"range": [75, 100], "color": "rgba(16,185,129,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "#34d399", "width": 3},
                        "thickness": 0.75, "value": 85,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=340, margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8d6e5"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with s2:
        st.markdown("### 🧩 Score Breakdown")
        breakdown = sust_results["eco_score_breakdown"]
        b_df = pd.DataFrame([
            {"Pillar": "Energy Efficiency (Max 25)", "Score": breakdown["energy_efficiency"]},
            {"Pillar": "Waste Minimization (Max 25)", "Score": breakdown["waste_minimization"]},
            {"Pillar": "Carbon Intensity (Max 25)", "Score": breakdown["carbon_intensity"]},
            {"Pillar": "Throughput Health (Max 25)", "Score": breakdown["throughput_health"]},
        ])
        fig_break = px.bar(
            b_df, x="Score", y="Pillar", orientation="h",
            color="Score", text="Score", range_x=[0, 25],
            color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#34d399"], [1, "#6ee7b7"]],
        )
        fig_break.update_layout(**PLOTLY_THEME, height=340)
        st.plotly_chart(fig_break, use_container_width=True)

    st.markdown("### 📋 Environmental Telemetry")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("⚡ Energy", f"{sust_results['total_energy_kwh']:,.1f} kWh", f"{sust_results['energy_per_yard_kwh']:.2f} kWh/yd")
    sc2.metric("💧 Water", f"{sust_results['total_water_liters']:,.0f} L", f"{sust_results['water_per_yard_l']:.2f} L/yd")
    sc3.metric("🗑️ Scrap", f"{sust_results['total_waste_kg']:,.1f} kg", f"{sust_results['waste_per_yard_kg']:.3f} kg/yd")
    sc4.metric("🌍 Carbon", f"{sust_results['total_carbon_kg_co2e']:,.1f} kg CO₂e", f"{sust_results['carbon_per_yard_kg_co2e']:.3f} kg/yd")

    with st.expander("📐 Formulas & Assumptions"):
        st.markdown(
            f"""
            - **Energy:** $\\text{{Energy (kWh)}} = \\sum (\\text{{Machines}} \\times [(\\text{{Active}} \\times P) + (\\text{{Idle}} \\times 0.15P)])$
            - **Water:** $\\text{{Water (L)}} = \\text{{Output}} \\times {sust_results['assumptions']['water_consumption_rate']}$
            - **Waste:** $\\text{{Waste (kg)}} = (\\text{{Rejected}} + \\text{{Trim}}) \\times {sust_results['assumptions']['fabric_areal_weight']}$
            - **Carbon:** $\\text{{Carbon}} = \\text{{Energy}} \\times {sust_results['assumptions']['grid_emission_factor']}$
            """
        )

# =========================================================
# TAB 6: BOTTLENECKS & RECOMMENDATIONS
# =========================================================
with tab_recs:
    st.subheader("⚡ Bottleneck Detection & Recommendations")

    if bottleneck_info:
        st.markdown(
            f"""
            <div class="bottleneck-box">
                <h3 style="margin:0;">{bottleneck_info['severity']}</h3>
                <p style="margin:8px 0 0 0; font-size:15px;">
                    <strong>Stage:</strong> {bottleneck_info['stage']} &nbsp;|&nbsp;
                    <strong>Utilization:</strong> {bottleneck_info['utilization_pct']}% &nbsp;|&nbsp;
                    <strong>Avg Wait:</strong> {bottleneck_info['avg_wait_hours']} hrs
                </p>
                <p style="margin:4px 0 0 0; font-size:13px; opacity:0.85;">{bottleneck_info['summary']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🛠️ Actionable Recommendations")
    for rec in recommendations:
        p_class = "priority-high" if rec["priority"] == "High" else ("priority-medium" if rec["priority"] == "Medium" else "priority-low")
        st.markdown(
            f"""
            <div class="rec-card" style="border-left: 3px solid {'#ef4444' if rec['priority']=='High' else '#f59e0b' if rec['priority']=='Medium' else '#10b981'};">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <h4 style="margin:0; font-size:15px;">{rec['category']}: {rec['title']}</h4>
                    <span class="{p_class}">{rec['priority']}</span>
                </div>
                <p style="margin:0 0 10px 0; font-size:13px; opacity:0.8;">{rec['detail']}</p>
                <p style="margin:0 0 10px 0; font-size:12px; opacity:0.7;"><strong>Supporting metric:</strong> {rec.get('supporting_metric', 'Current simulation output')}</p>
                <span class="eco-badge">Expected: {rec['expected_gain']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# TAB 7: GEMINI 3.6 FLASH AI COPILOT
# =========================================================
with tab_ai:
    st.subheader("💬 Gemini 3.6 Flash AI Copilot")
    st.markdown("Ask the AI for root-cause analysis, energy management, and production optimization — powered by live factory telemetry.")

    st.markdown("##### ⚡ Quick Prompts:")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    quick_prompt = None
    if q_col1.button("🔍 Bottlenecks", use_container_width=True):
        quick_prompt = "What is the primary bottleneck in the weaving factory and how can we balance the workcenters?"
    if q_col2.button("🌿 Eco Score", use_container_width=True):
        quick_prompt = "How can we increase our Project Eco Score above 85 while minimizing carbon and material waste?"
    if q_col3.button("🧵 Rejection", use_container_width=True):
        quick_prompt = "What process parameters in sizing and weaving are most responsible for fabric rejections?"
    if q_col4.button("💡 Optimize", use_container_width=True):
        quick_prompt = "What would be the ROI and carbon benefit of adding 2 additional looms and reducing sizing cycle time?"

    ai_context = {
        "dataset_name": selected_file if avail_datasets else "Custom CSV",
        "completed_yds": sim_results["completed_yds"],
        "rejected_yds": sim_results["rejected_yds"],
        "rejection_rate_pct": sim_results["rejection_rate_pct"],
        "bottleneck_stage": bottleneck_info["stage"] if bottleneck_info else "None",
        "bottleneck_util": bottleneck_info["utilization_pct"] if bottleneck_info else 0,
        "total_energy_kwh": sust_results["total_energy_kwh"],
        "energy_per_yd": sust_results["energy_per_yard_kwh"],
        "total_waste_kg": sust_results["total_waste_kg"],
        "total_carbon_kg_co2e": sust_results["total_carbon_kg_co2e"],
        "eco_score": sust_results["eco_score"],
        "ml_accuracy": f"{ml_sys.clf_metrics.get('accuracy', 0)*100:.1f}%" if ml_sys.clf_metrics else "N/A",
        "ml_rejection_probability": pred_res.get("rejection_probability", "N/A"),
        "ml_predicted_production_yds": pred_res.get("predicted_production_yds", "N/A"),
        "ml_top_features": list(ml_sys.feature_importances_clf.items())[:5],
        "sustainability_status": "estimated from simulation assumptions",
    }

    user_ai_query = st.text_input(
        "Ask Gemini a question:",
        value=quick_prompt if quick_prompt else "",
        placeholder="e.g., How can we reduce sizing energy consumption without degrading yarn strength?",
    )

    if st.button("Submit to Gemini", type="primary") or quick_prompt:
        if user_ai_query:
            with st.spinner("Querying Gemini 3.6 Flash..."):
                response_text = query_gemini_process_intelligence(
                    prompt=user_ai_query,
                    context_data=ai_context,
                    api_key=st.session_state.get("gemini_api_key", ""),
                    model_name="gemini-3.6-flash",
                )
                st.markdown("### 💡 AI Response:")
                st.markdown(response_text)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding: 12px 0;">
        <p style="font-size:12px; color:#475569 !important; margin:0;">
            AI-Powered Digital Twin for Sustainable Textile Manufacturing &nbsp;•&nbsp;
            Streamlit &nbsp;•&nbsp; SimPy &nbsp;•&nbsp; Scikit-Learn &nbsp;•&nbsp; Plotly &nbsp;•&nbsp; Gemini 3.6 Flash
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
