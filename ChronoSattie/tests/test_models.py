"""
Tests for Baseline and Machine Learning Forecasting Models (Points 12, 13, 14)
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.models.baselines import PersistenceForecaster, LinearDriftForecaster
from src.models.forecasting import ForecastingEngine
from src.preprocessing.features import FeatureEngineer, FeaturePipelineConfig


def test_persistence_forecaster() -> None:
    """Test 12: Verify persistence baseline predicts last observed value."""
    model = PersistenceForecaster()
    X = np.array([[1], [2], [3]])
    y = np.array([10.5, 11.2, 12.0])

    model.fit(X, y)
    preds = model.predict(np.array([[4], [5]]))

    assert len(preds) == 2
    assert all(p == 12.0 for p in preds)


def test_linear_drift_forecaster() -> None:
    """Test 12: Verify linear drift baseline computes correct extrapolation slope."""
    model = LinearDriftForecaster()
    # Linear ramp with slope = 2.0, intercept = 10.0
    y = np.array([10.0, 12.0, 14.0, 16.0])
    model.fit(np.zeros((4, 1)), y)

    assert pytest.approx(model.slope, 1e-4) == 2.0
    # Next step (t=4) -> 10 + 2*4 = 18.0
    assert pytest.approx(model.predict(4), 1e-4) == 18.0


@pytest.mark.parametrize("model_name", ["linear_regression", "random_forest", "hist_gradient_boosting"])
def test_ml_model_training_and_horizons(model_name: str) -> None:
    """Test 13 & 14: Verify ML models train and predict across horizons."""
    n_samples = 150
    start = datetime(2026, 1, 1, 0, 0)
    epochs = [start + timedelta(minutes=15 * i) for i in range(n_samples)]
    # Smooth signal with drift and sine harmonic
    t = np.arange(n_samples)
    values = 100.0 + 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 48)

    df = pd.DataFrame({"epoch": epochs, "sat_id": "G01", "clock_bias_us": values})

    for h_hours, h_steps in [(1, 4), (6, 24)]:
        feat_eng = FeatureEngineer(FeaturePipelineConfig(target_col="clock_bias_us", lags=[1, 2, 4]))
        X, y, feat_cols = feat_eng.create_features_and_targets(df, horizon_steps=h_steps)

        engine = ForecastingEngine(model_name=model_name)
        engine.fit(
            X_train=X.iloc[:80],
            y_train=y.iloc[:80],
            feature_cols=feat_cols,
            target_name="clock_bias_us",
            sat_id="G01",
            horizon_hours=h_hours,
            horizon_steps=h_steps,
        )

        assert engine.is_fitted
        preds = engine.predict(X.iloc[80:90])
        assert len(preds) == 10
        assert not np.isnan(preds).any()
