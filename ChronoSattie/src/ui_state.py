"""
Shared session state and helper utilities for Streamlit multi-page interface.
"""

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from src.data_ingestion.synthetic_generator import SyntheticGNSSGenerator


def init_session_state() -> None:
    """Initialize default session state keys if not already present."""
    if "raw_df" not in st.session_state:
        # Load synthetic demo data by default so the prototype is immediately interactive
        gen = SyntheticGNSSGenerator(random_seed=42)
        demo_df = gen.generate_orbit_and_clock(duration_hours=72)
        st.session_state["raw_df"] = demo_df
        st.session_state["dataset_name"] = "synthetic_demo (Default 72h)"
        st.session_state["is_synthetic"] = True
        st.session_state["data_manifest"] = {
            "source": "SYNTHETIC_SIMULATOR",
            "name": "synthetic_demo (Default 72h)",
            "is_synthetic": True,
            "satellites": sorted(demo_df["sat_id"].unique().tolist()),
            "start_time": demo_df["epoch"].min().isoformat(),
            "end_time": demo_df["epoch"].max().isoformat(),
            "total_rows": len(demo_df),
        }

    if "cleaned_df" not in st.session_state:
        st.session_state["cleaned_df"] = st.session_state["raw_df"].copy()

    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}

    if "forecast_history" not in st.session_state:
        st.session_state["forecast_history"] = []

    if "evaluation_results" not in st.session_state:
        st.session_state["evaluation_results"] = {}


def render_sidebar_status() -> None:
    """Render common sidebar status banner showing data provenance and mode."""
    st.sidebar.title("🛰️ ChronoSattie")
    st.sidebar.caption("Satellite Clock & Ephemeris Forecasting")

    is_synth = st.session_state.get("is_synthetic", True)
    dataset_name = st.session_state.get("dataset_name", "None")

    if is_synth:
        st.sidebar.warning("⚠️ **MODE: SYNTHETIC DEMO**\nSimulated Keplerian & Clock Drift Data. Not real GNSS observations.")
    else:
        st.sidebar.success(f"🟢 **MODE: REAL GNSS DATA**\nActive Product: {dataset_name}")

    st.sidebar.divider()
    raw_df = st.session_state.get("raw_df", pd.DataFrame())
    if not raw_df.empty:
        st.sidebar.write(f"**Records:** {len(raw_df):,}")
        sats = sorted(raw_df["sat_id"].unique().tolist())
        st.sidebar.write(f"**Satellites ({len(sats)}):** {', '.join(sats)}")
        st.sidebar.write(f"**Start:** {raw_df['epoch'].min():%Y-%m-%d %H:%M}")
        st.sidebar.write(f"**End:** {raw_df['epoch'].max():%Y-%m-%d %H:%M}")
    else:
        st.sidebar.info("No dataset loaded.")
