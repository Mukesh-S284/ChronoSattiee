"""
Page 2: Data Explorer
Upload authentic SP3/CLK files, generate synthetic data, inspect raw & cleaned observations,
and visualize interactive 3D orbits and clock bias time series.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.ui_state import init_session_state, render_sidebar_status
from src.data_ingestion.sp3_parser import SP3Parser
from src.data_ingestion.rinex_clk_parser import RINEXCLKParser
from src.data_ingestion.synthetic_generator import SyntheticGNSSGenerator
from src.preprocessing.cleaning import DataCleaner
from src.visualization.plots import (
    plot_clock_bias_series,
    plot_orbit_3d,
    plot_orbit_components,
)

st.set_page_config(page_title="Data Explorer - ChronoSattie", page_icon="🔍", layout="wide")
init_session_state()
render_sidebar_status()

st.title("🔍 Data Ingestion & Explorer")
st.markdown("Ingest authentic GNSS orbit (SP3) or clock (RINEX CLK) files, or generate controlled synthetic demonstrations.")

# Ingestion Controls Tabs
tab_upload, tab_synthetic, tab_clean = st.tabs(["📤 Upload Real GNSS Files", "🧪 Synthetic Data Generator", "🧹 Clean & Preprocess"])

with tab_upload:
    st.subheader("Upload Real SP3 or RINEX CLK Products")
    st.markdown(
        "Upload authentic IGS precise ephemeris (`.sp3`) or clock (`.clk`) files retrieved from "
        "[IGS](https://igs.org/) or [NASA CDDIS](https://cddis.nasa.gov/)."
    )

    col_sp3, col_clk = st.columns(2)

    with col_sp3:
        sp3_file = st.file_uploader("Upload IGS SP3 File (.sp3)", type=["sp3", "txt"])
        if sp3_file is not None:
            if st.button("Parse SP3 File", key="btn_parse_sp3"):
                with st.spinner("Parsing SP3 file..."):
                    try:
                        parser = SP3Parser()
                        df_sp3, meta = parser.parse_file(sp3_file)
                        st.session_state["raw_df"] = df_sp3
                        st.session_state["dataset_name"] = sp3_file.name
                        st.session_state["is_synthetic"] = False
                        st.session_state["data_manifest"] = {
                            "source": "REAL_IGS_SP3",
                            "name": sp3_file.name,
                            "is_synthetic": False,
                            "satellites": meta.satellites or sorted(df_sp3["sat_id"].unique().tolist()),
                            "start_time": df_sp3["epoch"].min().isoformat(),
                            "end_time": df_sp3["epoch"].max().isoformat(),
                            "total_rows": len(df_sp3),
                            "agency": meta.agency,
                            "coordinate_system": meta.coordinate_system,
                        }
                        st.success(f"Successfully loaded {len(df_sp3):,} records from '{sp3_file.name}'!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error parsing SP3: {e}")

    with col_clk:
        clk_file = st.file_uploader("Upload RINEX CLK File (.clk)", type=["clk", "txt"])
        if clk_file is not None:
            if st.button("Parse RINEX CLK File", key="btn_parse_clk"):
                with st.spinner("Parsing RINEX CLK file..."):
                    try:
                        parser = RINEXCLKParser()
                        df_clk, meta = parser.parse_file(clk_file)
                        # Add microsecond column if not present
                        if "clock_bias_us" not in df_clk.columns:
                            df_clk["clock_bias_us"] = df_clk["clock_bias_s"] * 1e6

                        st.session_state["raw_df"] = df_clk
                        st.session_state["dataset_name"] = clk_file.name
                        st.session_state["is_synthetic"] = False
                        st.session_state["data_manifest"] = {
                            "source": "REAL_RINEX_CLK",
                            "name": clk_file.name,
                            "is_synthetic": False,
                            "satellites": meta.satellites or sorted(df_clk["sat_id"].unique().tolist()),
                            "start_time": df_clk["epoch"].min().isoformat(),
                            "end_time": df_clk["epoch"].max().isoformat(),
                            "total_rows": len(df_clk),
                        }
                        st.success(f"Successfully loaded {len(df_clk):,} clock records from '{clk_file.name}'!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error parsing RINEX CLK: {e}")

    st.markdown("---")
    st.markdown("#### ⚡ Quick Demo: Pre-Downloaded Authentic IGS Dataset")
    st.markdown(
        "A real IGS final orbit product (`IGS0OPSFIN_20230500000_01D_15M_ORB.SP3.gz`) covering **32 GPS satellites** "
        "at 15-minute intervals across 24 hours is available in `data/raw/`."
    )
    if st.button("🚀 Load Pre-Downloaded Real IGS Data (32 Satellites, 3,072 Obs)", type="secondary"):
        from pathlib import Path
        sample_path = Path("data/raw/IGS0OPSFIN_20230500000_01D_15M_ORB.SP3.gz")
        if sample_path.exists():
            with st.spinner("Parsing authentic IGS orbit dataset..."):
                parser = SP3Parser()
                df_real, meta_real = parser.parse_file(sample_path)
                st.session_state["raw_df"] = df_real
                st.session_state["cleaned_df"] = df_real.copy()
                st.session_state["dataset_name"] = sample_path.name
                st.session_state["is_synthetic"] = False
                st.session_state["data_manifest"] = {
                    "source": "AUTHENTIC_BKG_IGS_MIRROR",
                    "name": sample_path.name,
                    "is_synthetic": False,
                    "satellites": sorted(df_real["sat_id"].unique().tolist()),
                    "start_time": df_real["epoch"].min().isoformat(),
                    "end_time": df_real["epoch"].max().isoformat(),
                    "total_rows": len(df_real),
                    "agency": meta_real.agency,
                    "coordinate_system": meta_real.coordinate_system,
                }
                st.success(f"Loaded {len(df_real):,} authentic observations across {df_real['sat_id'].nunique()} satellites!")
                st.rerun()
        else:
            st.error("Sample real file not found in data/raw/. Please use the manual file uploader above.")
    st.subheader("Generate Reproducible Synthetic GNSS Data")
    st.markdown("Generates a physically grounded Keplerian orbit and atomic clock drift dataset with realistic noise and gaps.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        synth_duration = st.slider("Duration (Hours)", min_value=24, max_value=168, value=72, step=12)
    with c2:
        synth_sats = st.multiselect("Satellites", ["G01", "G02", "G14", "E04", "R05", "C12"], default=["G01", "G02", "G14", "E04"])
    with c3:
        synth_missing = st.slider("Simulated Gap Rate", min_value=0.0, max_value=0.10, value=0.02, step=0.01)
    with c4:
        synth_seed = st.number_input("Random Seed", value=42, step=1)

    if st.button("Generate Synthetic Dataset", type="primary"):
        with st.spinner("Generating orbital trajectories and clock drift..."):
            gen = SyntheticGNSSGenerator(random_seed=int(synth_seed))
            df_synth = gen.generate_orbit_and_clock(
                sat_ids=synth_sats if synth_sats else ["G01", "G02"],
                duration_hours=synth_duration,
                missing_rate=synth_missing,
            )
            st.session_state["raw_df"] = df_synth
            st.session_state["dataset_name"] = f"synthetic_demo_{synth_duration}h_seed{synth_seed}"
            st.session_state["is_synthetic"] = True
            st.session_state["data_manifest"] = {
                "source": "SYNTHETIC_SIMULATOR",
                "name": st.session_state["dataset_name"],
                "is_synthetic": True,
                "satellites": sorted(df_synth["sat_id"].unique().tolist()),
                "start_time": df_synth["epoch"].min().isoformat(),
                "end_time": df_synth["epoch"].max().isoformat(),
                "total_rows": len(df_synth),
                "duration_hours": synth_duration,
                "missing_rate": synth_missing,
                "seed": synth_seed,
            }
            st.session_state["cleaned_df"] = df_synth.copy()
            st.success(f"Generated {len(df_synth):,} synthetic records!")
            st.rerun()

with tab_clean:
    st.subheader("Data Cleaning & Quality Pipeline")
    raw_df = st.session_state.get("raw_df", pd.DataFrame())

    if not raw_df.empty:
        c_filter, c_action = st.columns([1, 1])
        with c_filter:
            filter_outliers = st.checkbox("Replace statistical outliers with NaN", value=False)
        with c_action:
            if st.button("Run Preprocessing & Cleaning", key="btn_run_cleaning"):
                cleaner = DataCleaner()
                cleaned_df, report = cleaner.clean_dataset(raw_df, filter_outliers=filter_outliers)
                st.session_state["cleaned_df"] = cleaned_df
                st.session_state["cleaning_report"] = report
                st.success("Cleaning pipeline executed successfully!")

        report = st.session_state.get("cleaning_report", None)
        if report:
            st.markdown("#### Cleaning Diagnostics Report")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows Cleaned", report.final_rows)
            c2.metric("Duplicates Dropped", report.duplicates_removed)
            c3.metric("Observation Gaps", report.gap_count)
            c4.metric("Outliers Detected", report.outliers_detected)

            if report.warnings:
                with st.expander("Inspection Warnings", expanded=True):
                    for w in report.warnings:
                        st.warning(w)

st.markdown("---")

# Active Data Inspection & Interactive Visualizations
df = st.session_state.get("cleaned_df", st.session_state.get("raw_df", pd.DataFrame()))

if df.empty:
    st.info("No dataset is currently available. Please upload a file or generate synthetic data above.")
else:
    st.subheader("📊 Interactive Data Explorer & Visualizations")

    sat_list = sorted(df["sat_id"].unique().tolist())
    c_sat, c_date = st.columns([1, 2])

    with c_sat:
        selected_sat = st.selectbox("Select Satellite to Inspect", sat_list)
    with c_date:
        min_date = df["epoch"].min().date()
        max_date = df["epoch"].max().date()
        date_range = st.date_input("Filter Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    filtered_df = df[df["sat_id"] == selected_sat].copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["epoch"].dt.date >= date_range[0]) & (filtered_df["epoch"].dt.date <= date_range[1])
        ]

    # Visualizations
    v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs(["⏱️ Clock Bias", "🌐 3D Orbit (ECEF)", "📐 Coordinate Components", "📋 Raw Table Data"])

    with v_tab1:
        clk_col = "clock_bias_us" if "clock_bias_us" in filtered_df.columns else "clock_bias_ns"
        units = "μs" if clk_col == "clock_bias_us" else "ns"
        if clk_col in filtered_df.columns:
            fig_clk = plot_clock_bias_series(filtered_df, sat_id=selected_sat, target_col=clk_col, units=units)
            st.plotly_chart(fig_clk, use_container_width=True)
        else:
            st.warning("Clock bias column not found in dataset.")

    with v_tab2:
        if all(c in filtered_df.columns for c in ["x_km", "y_km", "z_km"]):
            fig_3d = plot_orbit_3d(filtered_df, sat_id=selected_sat)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("3D Orbit requires X, Y, Z coordinates (from SP3 files). Current dataset only has clock observations.")

    with v_tab3:
        if all(c in filtered_df.columns for c in ["x_km", "y_km", "z_km"]):
            fig_comp = plot_orbit_components(filtered_df, sat_id=selected_sat)
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Coordinate components not present in clock-only dataset.")

    with v_tab4:
        st.dataframe(filtered_df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 rows of {len(filtered_df):,} total observations for {selected_sat}.")
