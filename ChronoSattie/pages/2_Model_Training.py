"""
Page 3: Model Training
Train time-series forecasting models using chronological, leakage-free validation splits.
"""

from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

from src.ui_state import init_session_state, render_sidebar_status
from src.config import HORIZON_STEPS_SP3, DEFAULT_HORIZONS_HOURS
from src.models.forecasting import ForecastingEngine
from src.models.registry import ModelMetadata, ModelRegistry
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.validation import TimeSeriesValidator
from src.preprocessing.features import FeatureEngineer, FeaturePipelineConfig

st.set_page_config(page_title="Model Training - ChronoSattie", page_icon="🧠", layout="wide")
init_session_state()
render_sidebar_status()

st.title("🧠 Model Training & Chronological Validation")
st.markdown(
    "Configure, train, and evaluate time-series forecasting models on strictly chronological splits. "
    "Future observations are completely withheld during training to prevent data leakage."
)

df: pd.DataFrame = st.session_state.get("cleaned_df", st.session_state.get("raw_df", pd.DataFrame()))

if df.empty:
    st.warning("No dataset is loaded. Please upload or generate a dataset in the Data Explorer.")
    st.stop()

# Satellite and Target Configuration
sat_list = sorted(df["sat_id"].unique().tolist())

# Detect available targets
avail_targets = []
for candidate in ["clock_bias_us", "clock_bias_ns", "clock_bias_s", "x_km", "y_km", "z_km"]:
    if candidate in df.columns:
        avail_targets.append(candidate)

col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

with col_cfg1:
    selected_sat = st.selectbox("Select Target Satellite", sat_list)
with col_cfg2:
    selected_target = st.selectbox("Select Prediction Target", avail_targets)
with col_cfg3:
    selected_horizon = st.selectbox(
        "Forecast Horizon",
        [1, 6, 12, 24],
        format_func=lambda h: f"{h} Hour{'s' if h > 1 else ''} ({HORIZON_STEPS_SP3[h]} steps @ 15m)"
    )

horizon_steps = HORIZON_STEPS_SP3[selected_horizon]

col_train1, col_train2, col_train3 = st.columns(3)

with col_train1:
    model_choice = st.selectbox(
        "Forecasting Model",
        [
            ("random_forest", "Random Forest Regressor"),
            ("hist_gradient_boosting", "HistGradientBoosting Regressor"),
            ("linear_regression", "Linear Regression Baseline"),
            ("persistence", "Persistence Baseline (Last Observed)"),
        ],
        format_func=lambda x: x[1],
    )[0]

with col_train2:
    train_ratio = st.slider("Chronological Train / Test Split Ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.05)

with col_train3:
    st.write("")
    st.write("")
    train_button = st.button("🚀 Train & Evaluate Model", type="primary")

st.markdown("---")

if train_button:
    with st.spinner(f"Training {model_choice} for {selected_sat} at {selected_horizon}h horizon..."):
        eval_result = TimeSeriesValidator.evaluate_model_at_horizon(
            df=df,
            sat_id=selected_sat,
            target_col=selected_target,
            model_name=model_choice,
            horizon_hours=selected_horizon,
            horizon_steps=horizon_steps,
            train_ratio=train_ratio,
        )

        if eval_result["status"] != "SUCCESS":
            st.error(f"Training could not proceed: {eval_result['reason']}")
        else:
            model_key = f"{selected_sat}_{selected_target}_{model_choice}_{selected_horizon}h"
            st.session_state["trained_models"][model_key] = eval_result
            st.success(f"Model successfully trained and evaluated on test split ({eval_result['test_samples']} test points)!")

# Show results if available
model_keys = list(st.session_state.get("trained_models", {}).keys())

if model_keys:
    st.subheader("📈 Model Performance & Comparison")

    records = []
    for k, res in st.session_state["trained_models"].items():
        m_met = res["model_metrics"]
        b_met = res["baseline_metrics"]
        records.append({
            "Model Key": k,
            "Satellite": res["sat_id"],
            "Target": res["target_col"],
            "Model": res["model_name"],
            "Horizon": f"{res['horizon_hours']}h",
            "Units": m_met.target_units,
            "Model MAE": f"{m_met.mae:.6f}",
            "Model RMSE": f"{m_met.rmse:.6f}",
            "Mean Bias Error": f"{m_met.mean_bias_error:.6f}",
            "Persistence MAE": f"{b_met.mae:.6f}",
            "Persistence RMSE": f"{b_met.rmse:.6f}",
            "Test Samples": res["test_samples"],
        })

    perf_df = pd.DataFrame(records)
    st.dataframe(perf_df, use_container_width=True)

    # Persistence of models
    st.subheader("💾 Persist Trained Model to Registry")
    c_save_key, c_save_btn = st.columns([3, 1])
    with c_save_key:
        save_model_key = st.selectbox("Select Model to Persist", model_keys)
    with c_save_btn:
        st.write("")
        st.write("")
        if st.button("Save Model to Disk"):
            res = st.session_state["trained_models"][save_model_key]
            m_met = res["model_metrics"]

            # Re-instantiate engine to save
            sat_df = df[df["sat_id"] == res["sat_id"]].sort_values(by="epoch")
            split = TimeSeriesValidator.chronological_train_test_split(sat_df, train_ratio=train_ratio)
            feat_eng = FeatureEngineer(FeaturePipelineConfig(target_col=res["target_col"]))
            X_tr, y_tr, feat_cols = feat_eng.create_features_and_targets(split.train_df, horizon_steps=res["horizon_steps"])

            engine = ForecastingEngine(model_name=res["model_name"])
            engine.fit(
                X_train=X_tr,
                y_train=y_tr,
                feature_cols=feat_cols,
                target_name=res["target_col"],
                sat_id=res["sat_id"],
                horizon_hours=res["horizon_hours"],
                horizon_steps=res["horizon_steps"],
            )

            meta = ModelMetadata(
                model_id=save_model_key,
                model_name=res["model_name"],
                target_name=res["target_col"],
                sat_id=res["sat_id"],
                horizon_hours=res["horizon_hours"],
                horizon_steps=res["horizon_steps"],
                feature_names=feat_cols,
                train_start=str(split.train_start),
                train_end=str(split.train_end),
                n_samples=len(X_tr),
                metrics={"mae": m_met.mae, "rmse": m_met.rmse, "mbe": m_met.mean_bias_error},
                is_synthetic=st.session_state.get("is_synthetic", True),
                created_at=datetime.utcnow().isoformat(),
            )

            registry = ModelRegistry()
            saved_path = registry.save_model(engine, meta)
            st.success(f"Model saved to registry at `{saved_path.name}`!")

# Saved Models Inventory
st.subheader("🗄️ Saved Model Registry")
registry = ModelRegistry()
saved_list = registry.list_models()

if saved_list:
    reg_df = pd.DataFrame(saved_list)
    cols_to_show = ["model_id", "model_name", "sat_id", "target_name", "horizon_hours", "created_at", "is_synthetic"]
    st.dataframe(reg_df[[c for c in cols_to_show if c in reg_df.columns]], use_container_width=True)
else:
    st.info("No persisted models found in `models/saved/`.")
