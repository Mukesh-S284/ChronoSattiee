"""
Page 5: Prediction Evaluation
Quantitative error analysis, predicted-versus-actual verification, residual diagnostics,
and multi-horizon benchmark comparisons against the Persistence baseline.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.ui_state import init_session_state, render_sidebar_status
from src.config import HORIZON_STEPS_SP3
from src.evaluation.validation import TimeSeriesValidator
from src.evaluation.metrics import MetricsCalculator
from src.visualization.plots import (
    plot_forecast_vs_actual,
    plot_residuals,
    plot_horizon_comparison,
)

st.set_page_config(page_title="Evaluation - ChronoSattie", page_icon="📉", layout="wide")
init_session_state()
render_sidebar_status()

st.title("📉 Prediction Evaluation & Residual Analysis")
st.markdown(
    "Perform rigorous quantitative error evaluation comparing model forecasts against authentic or reference observations. "
    "Features include predicted-vs-actual curves, residual distributions, and multi-horizon performance scaling."
)

df: pd.DataFrame = st.session_state.get("cleaned_df", st.session_state.get("raw_df", pd.DataFrame()))

if df.empty:
    st.warning("No dataset loaded. Please upload or generate data in Data Explorer.")
    st.stop()

# Configuration
sat_list = sorted(df["sat_id"].unique().tolist())
avail_targets = [c for c in ["clock_bias_us", "clock_bias_ns", "clock_bias_s", "x_km", "y_km", "z_km"] if c in df.columns]

col1, col2, col3 = st.columns(3)
with col1:
    eval_sat = st.selectbox("Select Satellite", sat_list)
with col2:
    eval_target = st.selectbox("Target Parameter", avail_targets)
with col3:
    eval_model = st.selectbox(
        "Forecasting Model",
        [
            ("random_forest", "Random Forest Regressor"),
            ("hist_gradient_boosting", "HistGradientBoosting"),
            ("linear_regression", "Linear Regression"),
        ],
        format_func=lambda x: x[1]
    )[0]

run_eval = st.button("📊 Run Full Multi-Horizon Evaluation (1h, 6h, 12h, 24h)", type="primary")

st.markdown("---")

# Horizon Evaluation Table & Diagnostics
if run_eval or f"{eval_sat}_{eval_target}_{eval_model}_multi" in st.session_state.get("evaluation_results", {}):
    cache_key = f"{eval_sat}_{eval_target}_{eval_model}_multi"

    if run_eval:
        with st.spinner("Executing walk-forward chronological evaluation across all 4 horizons..."):
            horizon_records = []
            detailed_results = {}

            for h in [1, 6, 12, 24]:
                h_steps = HORIZON_STEPS_SP3[h]
                res = TimeSeriesValidator.evaluate_model_at_horizon(
                    df=df,
                    sat_id=eval_sat,
                    target_col=eval_target,
                    model_name=eval_model,
                    horizon_hours=h,
                    horizon_steps=h_steps,
                    train_ratio=0.8,
                )

                if res["status"] == "SUCCESS":
                    m_met = res["model_metrics"]
                    b_met = res["baseline_metrics"]

                    # Compute relative improvement over persistence
                    improvement_pct = 0.0
                    if b_met.mae > 1e-12:
                        improvement_pct = ((b_met.mae - m_met.mae) / b_met.mae) * 100.0

                    horizon_records.append({
                        "horizon_hours": h,
                        "model_mae": m_met.mae,
                        "model_rmse": m_met.rmse,
                        "baseline_mae": b_met.mae,
                        "baseline_rmse": b_met.rmse,
                        "mean_bias_error": m_met.mean_bias_error,
                        "improvement_vs_persistence_pct": improvement_pct,
                        "test_samples": res["test_samples"],
                        "units": m_met.target_units,
                    })
                    detailed_results[h] = res
                else:
                    horizon_records.append({
                        "horizon_hours": h,
                        "model_mae": np.nan,
                        "model_rmse": np.nan,
                        "baseline_mae": np.nan,
                        "baseline_rmse": np.nan,
                        "mean_bias_error": np.nan,
                        "improvement_vs_persistence_pct": np.nan,
                        "test_samples": 0,
                        "units": "N/A",
                        "note": res.get("reason", "Unavailable"),
                    })

            st.session_state["evaluation_results"][cache_key] = {
                "records": horizon_records,
                "detailed": detailed_results,
            }

    cached = st.session_state["evaluation_results"].get(cache_key, {})
    records = cached.get("records", [])
    detailed = cached.get("detailed", {})

    if records:
        st.subheader("📋 Multi-Horizon Performance Summary")
        summary_df = pd.DataFrame(records)

        # Format display
        display_cols = ["horizon_hours", "model_mae", "model_rmse", "baseline_mae", "baseline_rmse", "mean_bias_error", "improvement_vs_persistence_pct", "test_samples"]
        st.dataframe(
            summary_df[display_cols].style.format({
                "model_mae": "{:.6f}",
                "model_rmse": "{:.6f}",
                "baseline_mae": "{:.6f}",
                "baseline_rmse": "{:.6f}",
                "mean_bias_error": "{:.6f}",
                "improvement_vs_persistence_pct": "{:+.1f}%",
            }),
            use_container_width=True
        )

        units = records[0]["units"] if records else "μs"

        # Horizon Bar Chart
        st.subheader("📊 Horizon Error Scaling")
        valid_recs = [r for r in records if not np.isnan(r["model_mae"])]
        if valid_recs:
            chart_df = pd.DataFrame(valid_recs)
            fig_bar = plot_horizon_comparison(chart_df, units=units)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed Inspection of a Specific Horizon
        st.subheader("🔬 Detailed Residual Diagnostics for a Selected Horizon")
        inspect_h = st.selectbox("Select Horizon to Inspect Curves", list(detailed.keys()), format_func=lambda x: f"{x}h Horizon")

        if inspect_h in detailed:
            h_res = detailed[inspect_h]
            y_test = h_res["y_test"]
            y_pred = h_res["y_pred"]
            y_base = h_res["y_pred_baseline"]
            test_epochs = h_res["test_epochs"]
            residuals = y_pred - y_test

            # Predicted vs Actual Plot
            fig_pva = plot_forecast_vs_actual(
                target_epochs=test_epochs,
                y_actual=y_test,
                y_predicted=y_pred,
                y_baseline=y_base,
                target_name=eval_target,
                units=units,
                sat_id=eval_sat,
                model_name=eval_model,
                horizon_hours=inspect_h,
            )
            st.plotly_chart(fig_pva, use_container_width=True)

            # Residual diagnostics
            fig_res = plot_residuals(target_epochs=test_epochs, residuals=residuals, units=units)
            st.plotly_chart(fig_res, use_container_width=True)

            # 3D Orbit Error in meters if coordinate targets are available
            if "km" in eval_target and len(y_test) > 0:
                pos_err_m = np.abs(residuals) * 1000.0  # 1D coordinate component in meters
                st.info(
                    f"**1D Position Component Error in Metres:**\n"
                    f"- Mean Error: **{np.mean(pos_err_m):.2f} m**\n"
                    f"- RMSE: **{np.sqrt(np.mean(pos_err_m**2)):.2f} m**\n"
                    f"- Max Error: **{np.max(pos_err_m):.2f} m**"
                )
else:
    st.info("Click 'Run Full Multi-Horizon Evaluation' above to compute chronological forecast metrics.")
