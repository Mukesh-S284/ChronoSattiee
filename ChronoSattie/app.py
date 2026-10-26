"""
ChronoSattie: Main Dashboard & Page 1: Overview
AI/ML Satellite Clock and Ephemeris Time-Series Forecasting Prototype.
"""

import streamlit as st
import pandas as pd
from src.ui_state import init_session_state, render_sidebar_status
from src.config import DEFAULT_HORIZONS_HOURS
from src.models.forecasting import ForecastingEngine

st.set_page_config(
    page_title="ChronoSattie - GNSS Forecasting",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_sidebar_status()

# Hero Header
st.title("🛰️ ChronoSattie: Satellite Clock & Ephemeris Forecasting")
st.subheader("AI/ML Time-Series Forecasting System for GNSS Orbit & Clock Drift")

st.markdown("""
Welcome to **ChronoSattie**, a scientific time-series forecasting prototype developed to predict
satellite clock errors and orbital ephemeris parameters across forecast horizons of **1, 6, 12, and 24 hours**.

---
""")

# Persistent Real vs Synthetic Mode Alert
is_synthetic = st.session_state.get("is_synthetic", True)
dataset_name = st.session_state.get("dataset_name", "Unknown")

if is_synthetic:
    st.warning(
        "⚠️ **CURRENT STATUS: SYNTHETIC DEMONSTRATION MODE ACTIVE**\n\n"
        "The active dataset contains **simulated Keplerian orbits and atomic clock drift models** "
        "(with injected measurement noise and realistic gaps). It is provided for software demonstration, UI verification, "
        "and algorithmic testing. **Synthetic data is never presented as real satellite observations or operational performance.**\n\n"
        "👉 To ingest authentic IGS SP3 or RINEX CLK data, navigate to the **Data Explorer** page."
    )
else:
    st.success(
        f"🟢 **AUTHENTIC REAL GNSS OBSERVATIONS LOADED**\n\n"
        f"Active Dataset: **{dataset_name}** | Coordinates and clock corrections parsed from authentic IGS/RINEX archives."
    )

st.markdown("### 📊 Dataset Overview & Health Metrics")

raw_df: pd.DataFrame = st.session_state.get("raw_df", pd.DataFrame())

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Observations", f"{len(raw_df):,}" if not raw_df.empty else "0")

with col2:
    sats = sorted(raw_df["sat_id"].unique().tolist()) if not raw_df.empty else []
    st.metric("Active Satellites", len(sats))

with col3:
    time_span = "N/A"
    if not raw_df.empty:
        delta = raw_df["epoch"].max() - raw_df["epoch"].min()
        hours = delta.total_seconds() / 3600.0
        time_span = f"{hours:.1f} Hours ({delta.days}d {int(hours % 24)}h)"
    st.metric("Time Coverage", time_span)

with col4:
    st.metric("Supported Horizons", "1h, 6h, 12h, 24h")

st.markdown("---")

col_left, col_right = st.columns([1.2, 0.8])

with col_left:
    st.markdown("### 🛠️ Architecture & Forecasting Methodology")
    st.markdown("""
    The prototype provides a complete, leakage-free machine learning workflow:
    - **Data Ingestion Layer:** Support for **IGS SP3** (SP3-a, c, d) precise ephemeris and **RINEX 3.0x / 2.xx CLK** clock files.
    - **Gap-Aware Feature Engineering:** Strictly causal backward-looking lags, finite-difference velocity/drift estimates (suppressed across gaps), and orbital harmonics.
    - **Models Supported:**
      1. **Persistence:** Operational baseline predicting last observed value.
      2. **Linear Regression:** Baseline first-order linear trend estimator.
      3. **Random Forest Regressor:** Non-linear ensemble model capturing non-linear relativistic and orbital interactions.
      4. **HistGradientBoosting:** High-efficiency gradient-boosted decision trees for tabular time-series features.
    - **Leakage Prevention:** Chronological train/test splits. Future target observations are strictly withheld from training features.
    """)

with col_right:
    st.markdown("### 🛡️ Data Quality & Integrity Status")
    if not raw_df.empty:
        missing_count = int(raw_df.isna().sum().sum())
        dt_series = raw_df.groupby("sat_id")["epoch"].diff().dt.total_seconds().dropna()
        median_dt = float(dt_series.median()) if not dt_series.empty else 900.0
        gaps_count = int((dt_series > (median_dt * 1.5)).sum()) if not dt_series.empty else 0

        st.info(f"""
        - **Nominal Sampling Interval:** ~{median_dt:.0f} seconds ({median_dt / 60:.1f} min)
        - **Detected Observation Gaps:** {gaps_count}
        - **Missing / NaN Values:** {missing_count}
        - **Available Satellites:** {', '.join(sats[:10])}{' ...' if len(sats) > 10 else ''}
        - **Earliest Epoch:** {raw_df['epoch'].min():%Y-%m-%d %H:%M:%S}
        - **Latest Epoch:** {raw_df['epoch'].max():%Y-%m-%d %H:%M:%S}
        """)
    else:
        st.warning("No data is currently loaded in session state.")

st.markdown("### 🧭 Navigation Guide")
st.markdown("""
Use the sidebar pages to navigate through the project lifecycle:
1. **1_Data_Explorer:** Upload real SP3/CLK files, view data provenance, inspect raw/cleaned records, and render interactive 3D orbits and clock time-series.
2. **2_Model_Training:** Configure chronological splits, select features, train machine learning models, and review validation metrics.
3. **3_Satellite_Forecasting:** Generate future forecasts for 1h, 6h, 12h, and 24h horizons from any selected forecast origin.
4. **4_Evaluation:** Inspect predicted-vs-actual curves, residual distributions, and horizon performance comparisons when reference observations exist.
5. **5_Reports:** Download structured CSV packages and JSON data provenance manifests.
""")
