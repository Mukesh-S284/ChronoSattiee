"""
Page 6: Reports & Data Exports
Downloadable CSV and JSON reports for scientific validation, model evaluation,
and data provenance tracking.
"""

from datetime import datetime
import json
import streamlit as st
import pandas as pd

from src.ui_state import init_session_state, render_sidebar_status
from src.reporting.exports import ExportManager

st.set_page_config(page_title="Reports & Exports - ChronoSattie", page_icon="📑", layout="wide")
init_session_state()
render_sidebar_status()

st.title("📑 Scientific Reports & Data Exports")
st.markdown(
    "Export processed datasets, forecast runs, model evaluations, and data provenance manifests "
    "in standard CSV and JSON formats."
)

export_mgr = ExportManager()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Processed Observations")
    st.markdown("Download the cleaned, chronologically sorted time-series observations.")

    df_cleaned = st.session_state.get("cleaned_df", st.session_state.get("raw_df", pd.DataFrame()))
    if not df_cleaned.empty:
        csv_obs = df_cleaned.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Observations (CSV)",
            data=csv_obs,
            file_name=f"chronosattie_processed_obs_{datetime.utcnow():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )
        st.caption(f"Includes {len(df_cleaned):,} rows across {df_cleaned['sat_id'].nunique()} satellites.")
    else:
        st.info("No observations available to export.")

    st.markdown("---")

    st.subheader("2. Forecast Predictions")
    st.markdown("Download generated forecast records including origin, target epoch, predicted values, and units.")

    forecast_history = st.session_state.get("forecast_history", [])
    if forecast_history:
        f_recs = []
        for f in forecast_history:
            f_recs.append({
                "satellite_id": f.sat_id,
                "target_name": f.target_name,
                "model_name": f.model_name,
                "horizon_hours": f.horizon_hours,
                "horizon_steps": f.horizon_steps,
                "origin_epoch": f.forecast_origin.strftime("%Y-%m-%d %H:%M:%S"),
                "target_epoch": f.target_epoch.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_value": f.prediction,
                "units": f.units,
                "is_synthetic": f.is_synthetic,
            })
        f_df = pd.DataFrame(f_recs)
        csv_forecasts = f_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast History (CSV)",
            data=csv_forecasts,
            file_name=f"chronosattie_forecasts_{datetime.utcnow():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )
        st.caption(f"Contains {len(f_recs)} forecast execution records.")
    else:
        st.info("No forecasts have been generated in this session yet (generate some in Satellite Forecasting).")

with col_right:
    st.subheader("3. Model Evaluation Metrics")
    st.markdown("Download quantitative evaluation results (MAE, RMSE, MBE, Persistence benchmark).")

    eval_results = st.session_state.get("evaluation_results", {})
    all_metrics = []
    for key, val in eval_results.items():
        for rec in val.get("records", []):
            rec_copy = rec.copy()
            rec_copy["evaluation_run"] = key
            all_metrics.append(rec_copy)

    if all_metrics:
        met_df = pd.DataFrame(all_metrics)
        csv_metrics = met_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Evaluation Metrics (CSV)",
            data=csv_metrics,
            file_name=f"chronosattie_metrics_{datetime.utcnow():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )
        st.caption(f"Contains {len(all_metrics)} evaluation horizon records.")
    else:
        st.info("No evaluation runs recorded yet (run evaluation in Evaluation page).")

    st.markdown("---")

    st.subheader("4. Data Provenance Manifest")
    st.markdown("Download verifiable dataset provenance metadata, source URLs, time coverage, and synthetic status.")

    manifest = st.session_state.get("data_manifest", {})
    if manifest:
        manifest_copy = manifest.copy()
        manifest_copy["export_timestamp_utc"] = datetime.utcnow().isoformat()
        manifest_json = json.dumps(manifest_copy, indent=2, default=str)
        st.download_button(
            label="📥 Download Data Manifest (JSON)",
            data=manifest_json,
            file_name=f"chronosattie_data_manifest_{datetime.utcnow():%Y%m%d_%H%M%S}.json",
            mime="application/json",
        )
        st.json(manifest_copy)
    else:
        st.info("Data manifest not found.")
