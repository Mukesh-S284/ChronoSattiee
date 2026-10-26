"""
Tests for Chronological Splitting and Leakage Prevention (Points 10, 11)
"""

from datetime import datetime, timedelta
import pandas as pd
import pytest

from src.evaluation.validation import TimeSeriesValidator
from src.preprocessing.features import FeatureEngineer, FeaturePipelineConfig


def test_chronological_splitting_no_shuffling() -> None:
    """Test 11: Verify train/test splitting maintains strict chronological order without shuffle."""
    n_samples = 100
    start = datetime(2026, 1, 1, 0, 0)
    epochs = [start + timedelta(minutes=15 * i) for i in range(n_samples)]
    df = pd.DataFrame({"epoch": epochs, "sat_id": "G01", "val": range(n_samples)})

    split = TimeSeriesValidator.chronological_train_test_split(df, train_ratio=0.8)

    assert len(split.train_df) == 80
    assert len(split.test_df) == 20

    # Strict chronological boundary: train_max < test_min
    assert split.train_end < split.test_start
    assert split.train_df["epoch"].max() < split.test_df["epoch"].min()

    # Values must remain ordered
    assert split.train_df["val"].tolist() == list(range(80))
    assert split.test_df["val"].tolist() == list(range(80, 100))


def test_scaler_leakage_prevention() -> None:
    """Test 10: Verify scaler is fit exclusively on training data."""
    start = datetime(2026, 1, 1, 0, 0)
    epochs = [start + timedelta(minutes=15 * i) for i in range(100)]
    # Values that have drastically different mean in test period
    values = [10.0] * 80 + [500.0] * 20
    df = pd.DataFrame({"epoch": epochs, "sat_id": "G01", "clock_bias_us": values})

    split = TimeSeriesValidator.chronological_train_test_split(df, train_ratio=0.8)

    feat_eng = FeatureEngineer(FeaturePipelineConfig(target_col="clock_bias_us", lags=[1, 2]))
    X_train, _, feat_cols = feat_eng.create_features_and_targets(split.train_df, horizon_steps=2)

    scaler = feat_eng.fit_scaler(X_train, feat_cols)

    # Scaler mean for lag_1 must match the training set mean (~10.0), completely ignorant of test values (500.0)
    lag1_idx = feat_cols.index("clock_bias_us_lag_1")
    assert pytest.approx(scaler.mean_[lag1_idx], 0.1) == 10.0
