"""
Tests for Data Cleaning and Feature Engineering (Points 4, 5, 6, 9, 10)
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaning import DataCleaner
from src.preprocessing.features import FeatureEngineer, FeaturePipelineConfig


def test_timestamp_ordering_and_duplicates() -> None:
    """Test 4 & 5: Verify chronological sorting and duplicate removal."""
    # Construct an unordered dataframe with intentional duplicates
    records = [
        {"epoch": "2026-01-01 00:30:00", "sat_id": "G01", "clock_bias_us": 102.0, "x_km": 20000, "y_km": 10000, "z_km": 15000},
        {"epoch": "2026-01-01 00:00:00", "sat_id": "G01", "clock_bias_us": 100.0, "x_km": 20000, "y_km": 10000, "z_km": 15000},
        {"epoch": "2026-01-01 00:00:00", "sat_id": "G01", "clock_bias_us": 100.0, "x_km": 20000, "y_km": 10000, "z_km": 15000}, # duplicate
        {"epoch": "2026-01-01 00:15:00", "sat_id": "G01", "clock_bias_us": 101.0, "x_km": 20000, "y_km": 10000, "z_km": 15000},
    ]
    df = pd.DataFrame(records)

    cleaner = DataCleaner()
    cleaned_df, report = cleaner.clean_dataset(df)

    assert len(cleaned_df) == 3
    assert report.duplicates_removed == 1
    # Check strictly monotonic increasing timestamps
    assert cleaned_df["epoch"].is_monotonic_increasing


def test_gap_detection() -> None:
    """Test 6: Verify observation gap detection when interval exceeds threshold."""
    # Data with a 2-hour gap between 00:15 and 02:15
    records = [
        {"epoch": "2026-01-01 00:00:00", "sat_id": "G01", "clock_bias_us": 100.0},
        {"epoch": "2026-01-01 00:15:00", "sat_id": "G01", "clock_bias_us": 101.0},
        {"epoch": "2026-01-01 02:15:00", "sat_id": "G01", "clock_bias_us": 109.0},  # 2 hour gap
    ]
    df = pd.DataFrame(records)

    cleaner = DataCleaner()
    _, report = cleaner.clean_dataset(df)

    assert report.gap_count >= 1
    assert report.max_gap_duration_s == 7200.0


def test_feature_generation_and_leakage_prevention() -> None:
    """Test 9 & 10: Verify strictly backward-looking features and zero future leakage."""
    # Create 50 consecutive 15-minute observations
    epochs = [datetime(2026, 1, 1, 0, 0) + timedelta(minutes=15 * i) for i in range(60)]
    clocks = [100.0 + 0.5 * i for i in range(60)]
    df = pd.DataFrame({"epoch": epochs, "sat_id": "G01", "clock_bias_us": clocks})

    feat_eng = FeatureEngineer(FeaturePipelineConfig(target_col="clock_bias_us", lags=[1, 2, 4]))
    X, y, feat_cols = feat_eng.create_features_and_targets(df, horizon_steps=4)

    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)

    # Check lag features look backward:
    # At row index 0 in X (which corresponds to some epoch t in df):
    # lag 1 must equal df['clock_bias_us'] at epoch t - 1
    # target y must equal df['clock_bias_us'] at epoch t + 4
    for idx in range(min(5, len(X))):
        epoch_t = X["epoch"].iloc[idx]
        target_ep = X["target_epoch"].iloc[idx]

        assert (target_ep - epoch_t).total_seconds() == 4 * 900.0  # exactly 4 steps forward

        # Confirm target y is from the future, but NOT present in feature cols
        assert "clock_bias_us" not in feat_cols  # current value is lagged, not raw feature
        assert "target" not in feat_cols

        # Confirm lag_1 matches previous epoch in original series
        row_in_df = df[df["epoch"] == epoch_t].index[0]
        assert X["clock_bias_us_lag_1"].iloc[idx] == df["clock_bias_us"].iloc[row_in_df - 1]
        assert X["clock_bias_us_lag_2"].iloc[idx] == df["clock_bias_us"].iloc[row_in_df - 2]
