"""
Page 4: Satellite Forecasting
Generate forward-looking predictions for satellite clock bias and orbital coordinates
across 1, 6, 12, and 24-hour horizons from a selected forecast origin.
"""

from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.ui_state import init_session_state, render_sidebar_status
from src.config import HORIZON_STEPS_SP3
from src.models.forecasting import ForecastingEngine
from src.preprocessing.features import FeatureEngineer, FeaturePipelineConfig

st.set_page_config(page_title="Forecasting - ChronoSattie", page_icon="📡", layout="wide")
init_session_state()
render_sidebar_status()

st.title("📡 Satellite Real-Time Forecasting Engine")
st.markdown(
    "Generate forward predictions of satellite clock errors and orbital ephemeris coordinates "
    "for operational forecast horizons of **1h, 6h, 12h, and 24h**."
)

df: pd.DataFrame = st.session_state.get("cleaned_df", st.session_state.get("raw_df", pd.DataFrame()))

if df.empty:
    st.warning("No dataset loaded. Please upload or generate data in Data Explorer.")
    st.stop()

# Selection Controls
sat_list = sorted(df["sat_id"].unique().tolist())
avail_targets = [c for c in ["clock_bias_us", "clock_bias_ns", "clock_bias_s", "x_km", "y_km", "z_km"] if c in df.columns]

c1, c2, c3, c4 = st.columns(4)

with c1:
    selected_sat = st.selectbox("Satellite", sat_list)
with c2:
    selected_target = st.selectbox("Target Parameter", avail_targets)
with c3:
    model_name = st.selectbox(
        "Model",
        [
            ("random_forest", "Random Forest Regressor"),
            ("hist_gradient_boosting", "HistGradientBoosting"),
            ("linear_regression", "Linear Regression"),
            ("persistence", "Persistence Baseline"),
        ],
        format_func=lambda x: x[1]
    )[0]
with c4:
    horizon_hours = st.selectbox("Forecast Horizon", [1, 6, 12, 24], format_func=lambda h: f"{h} Hour{'s' if h > 1 else ''}")

horizon_steps = HORIZON_STEPS_SP3[horizon_hours]

sat_df = df[df["sat_id"] == selected_sat].sort_values(by="epoch").reset_index(drop=True)

# Forecast Origin Selection
st.subheader("⏱️ Forecast Origin Configuration")
c_origin1, c_origin2 = st.columns([2, 1])

with c_origin1:
    use_latest = st.checkbox("Use Most Recent Available Observation as Origin", value=True)
    if use_latest:
        origin_epoch = sat_df["epoch"].iloc[-1]
        origin_idx = len(sat_df) - 1
    else:
        origin_epoch = st.select_slider(
            "Select Historical Forecast Origin Epoch",
            options=sat_df["epoch"].tolist(),
            value=sat_df["epoch"].iloc[-max(horizon_steps + 10, len(sat_df) // 2)]
        )
        origin_idx = sat_df[sat_df["epoch"] == origin_epoch].index[0]

with c_origin2:
    st.info(f"**Origin Epoch:** {origin_epoch:%Y-%m-%d %H:%M:%S}\n\n**Target Epoch (+{horizon_hours}h):** {(origin_epoch + pd.Timedelta(hours=horizon_hours)):%Y-%m-%d %H:%M:%S}")

gen_forecast = st.button("🔮 Generate Satellite Forecast", type="primary")

st.markdown("---")

if gen_forecast:
    with st.spinner("Engineering features and predicting forward trajectory..."):
        # We only use observations strictly up to the origin epoch!
        hist_df = sat_df.iloc[: origin_idx + 1].copy()

        min_required = 30 + horizon_steps
        if len(hist_df) < min_required:
            st.error(f"Insufficient history up to origin epoch ({len(hist_df)} rows). Need at least {min_required} rows.")
            st.stop()

        feat_eng = FeatureEngineer(FeaturePipelineConfig(target_col=selected_target))
        try:
            X, y, feat_cols = feat_eng.create_features_and_targets(hist_df, horizon_steps=horizon_steps)
        except ValueError as e:
            st.error(f"Feature engineering failed: {e}")
            st.stop()

        if len(X) < 10:
            st.error(f"Too few feature samples ({len(X)}) to train forecasting model.")
            st.stop()

        # Fit forecasting model on data up to origin
        engine = ForecastingEngine(model_name=model_name)
        engine.fit(
            X_train=X,
            y_train=y,
            feature_cols=feat_cols,
            target_name=selected_target,
            sat_id=selected_sat,
            horizon_hours=horizon_hours,
            horizon_steps=horizon_steps,
        )

        # Generate forecast at latest row of X
        latest_feature_row = X.iloc[[-1]]
        forecast_res = engine.forecast_latest(
            latest_row=latest_feature_row,
            horizon_hours=horizon_hours,
        )

        # Store in forecast history
        st.session_state["forecast_history"].append(forecast_res)

        # Display Results Card
        st.success("✅ Forecast successfully computed!")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Satellite ID", forecast_res.sat_id)
        m_col2.metric("Target", f"{selected_target} ({forecast_res.units})")
        m_col3.metric("Predicted Value", f"{forecast_res.prediction:.6f} {forecast_res.units}")
        m_col4.metric("Horizon", f"{forecast_res.horizon_hours} Hour{'s' if forecast_res.horizon_hours > 1 else ''}")

        # Visualization of History + Forecast Curve
        st.subheader("📈 Historical Trajectory & Forecast Projection")

        # Plot last 48 hours of history leading to origin
        plot_window = hist_df.iloc[-min(len(hist_df), 192) :]

        fig = go.Figure()

        # 1. Historical observations (strictly known)
        fig.add_trace(
            go.Scatter(
                x=plot_window["epoch"],
                y=plot_window[selected_target],
                mode="lines+markers",
                marker=dict(size=3),
                line=dict(color="#1f77b4", width=2),
                name="Historical Known Observations",
            )
        )

        # 2. Forecast Origin marker
        fig.add_trace(
            go.Scatter(
                x=[origin_epoch],
                y=[plot_window[selected_target].iloc[-1]],
                mode="markers",
                marker=dict(size=10, color="orange", symbol="star"),
                name="Forecast Origin Epoch",
            )
        )

        # 3. Forecast Target marker and trajectory projection
        fig.add_trace(
            go.Scatter(
                x=[origin_epoch, forecast_res.target_epoch],
                y=[plot_window[selected_target].iloc[-1], forecast_res.prediction],
                mode="lines+markers",
                line=dict(color="#d62728", width=2.5, dash="dash"),
                marker=dict(size=8, color="#d62728", symbol="diamond"),
                name=f"Predicted Future ({model_name})",
            )
        )

        fig.update_layout(
            title=f"Forecast from Origin ({origin_epoch:%Y-%m-%d %H:%M}) to Target ({forecast_res.target_epoch:%Y-%m-%d %H:%M})",
            xaxis_title="Epoch",
            yaxis_title=f"{selected_target} ({forecast_res.units})",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scientific Disclaimer & Physics Note
        st.info(
            "⚠️ **Scientific & Operational Note:**\n"
            "- A machine learning prediction of satellite coordinates or clock drift is a statistical time-series estimate.\n"
            "- It does **not** replace a full numerical orbital propagator integrating gravitational geopotential harmonic coefficients, "
            "solar radiation pressure, or Earth tides.\n"
            "- For navigation-grade operational requirements, ML models should be combined with physical force models."
        )

# Historical Forecast Records
if st.session_state.get("forecast_history"):
    st.subheader("📋 Session Forecast History")
    hist_records = []
    for f in st.session_state["forecast_history"]:
        hist_records.append({
            "Satellite": f.sat_id,
            "Target": f.target_name,
            "Model": f.model_name,
            "Horizon": f"{f.horizon_hours}h",
            "Origin Epoch": f.forecast_origin.strftime("%Y-%m-%d %H:%M:%S"),
            "Target Epoch": f.target_epoch.strftime("%Y-%m-%d %H:%M:%S"),
            "Predicted Value": f"{f.prediction:.6f}",
            "Units": f.units,
            "Is Synthetic": f.is_synthetic,
        })
    st.dataframe(pd.DataFrame(hist_records), use_container_width=True)
