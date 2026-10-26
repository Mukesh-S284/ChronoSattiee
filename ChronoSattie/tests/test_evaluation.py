"""
Tests for Quantitative Evaluation Metrics (Points 15, 16, 17)
"""

import numpy as np
import pytest

from src.evaluation.metrics import MetricsCalculator


def test_3d_position_error_calculation() -> None:
    """Test 15: Verify 3D position error in meters: norm(pred_xyz - ref_xyz) * 1000."""
    # Reference at (0, 0, 0) km
    ref_xyz_km = np.array([
        [10000.0, 20000.0, 15000.0],
        [10000.0, 20000.0, 15000.0],
    ])
    # Predicted with 3m, 4m errors:
    # Point 0: dx = 0.003 km (3m), dy = 0.004 km (4m), dz = 0 -> norm = 5 meters
    # Point 1: dx = 0, dy = 0, dz = 0.010 km (10m) -> norm = 10 meters
    pred_xyz_km = np.array([
        [10000.003, 20000.004, 15000.0],
        [10000.0, 20000.0, 15000.010],
    ])

    metrics = MetricsCalculator.calculate_3d_position_error(ref_xyz_km, pred_xyz_km)

    assert metrics["n_samples"] == 2
    # Mean of 5.0m and 10.0m is 7.5m
    assert pytest.approx(metrics["mean_position_error_m"], 1e-4) == 7.5
    # Max is 10.0m
    assert pytest.approx(metrics["max_position_error_m"], 1e-4) == 10.0
    # RMSE: sqrt((25 + 100)/2) = sqrt(62.5) ~= 7.9057m
    assert pytest.approx(metrics["rmse_position_error_m"], 1e-4) == np.sqrt(62.5)


def test_clock_error_conversion() -> None:
    """Test 16: Verify clock error metrics conversion from microseconds to nanoseconds."""
    y_true = np.array([100.0, 101.0, 102.0])
    # 0.005 us error (which is 5.0 ns)
    y_pred = np.array([100.005, 101.005, 102.005])

    metrics = MetricsCalculator.calculate_scalar_metrics(
        y_true, y_pred, target_name="clock_bias_us", target_units="us"
    )

    assert pytest.approx(metrics.mae, 1e-6) == 0.005
    assert pytest.approx(metrics.mae_ns, 1e-4) == 5.0  # 0.005 us = 5.0 ns


def test_metric_calculations_mae_rmse_mbe() -> None:
    """Test 17: Verify standard metric formulas (MAE, RMSE, Mean Bias Error)."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    # Residuals: [+2, -4, +6, -8]
    y_pred = np.array([12.0, 16.0, 36.0, 32.0])

    metrics = MetricsCalculator.calculate_scalar_metrics(y_true, y_pred, target_units="us")

    # MAE = (2 + 4 + 6 + 8) / 4 = 5.0
    assert pytest.approx(metrics.mae, 1e-6) == 5.0
    # MBE = (2 - 4 + 6 - 8) / 4 = -1.0
    assert pytest.approx(metrics.mean_bias_error, 1e-6) == -1.0
    # RMSE = sqrt((4 + 16 + 36 + 64)/4) = sqrt(120/4) = sqrt(30) ~= 5.4772
    assert pytest.approx(metrics.rmse, 1e-4) == np.sqrt(30.0)
