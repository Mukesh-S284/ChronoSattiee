"""
Scientific evaluation metrics for GNSS satellite orbit and clock forecasts.
Implements MAE, RMSE, Mean Bias Error, 3D Euclidean Position Error (meters), and Clock Error (ns/s).
Explicitly avoids misleading percentage metrics on zero-crossing signed quantities.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class ForecastMetrics:
    """Quantitative performance metrics with explicit units."""
    mae: float
    rmse: float
    mean_bias_error: float
    std_error: float
    n_samples: int
    target_units: str
    target_name: str
    # Specific orbital position error metrics (in meters) if target is 3D coordinates
    mean_3d_position_error_m: Optional[float] = None
    max_3d_position_error_m: Optional[float] = None
    p95_3d_position_error_m: Optional[float] = None
    # Clock error metrics in nanoseconds
    mae_ns: Optional[float] = None
    rmse_ns: Optional[float] = None


class MetricsCalculator:
    """
    Computes rigorous error metrics for satellite time-series forecasts.
    """

    @staticmethod
    def calculate_scalar_metrics(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        target_name: str = "clock_bias_us",
        target_units: str = "us",
    ) -> ForecastMetrics:
        """
        Calculate standard regression metrics: MAE, RMSE, MBE, and standard deviation of residuals.
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)

        # Filter out any NaN or Inf pairs
        valid = ~np.isnan(y_t) & ~np.isnan(y_p) & ~np.isinf(y_t) & ~np.isinf(y_p)
        y_t = y_t[valid]
        y_p = y_p[valid]

        n = len(y_t)
        if n == 0:
            return ForecastMetrics(
                mae=np.nan,
                rmse=np.nan,
                mean_bias_error=np.nan,
                std_error=np.nan,
                n_samples=0,
                target_units=target_units,
                target_name=target_name,
            )

        residuals = y_p - y_t
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mbe = float(np.mean(residuals))
        std_err = float(np.std(residuals))

        mae_ns = None
        rmse_ns = None
        if "us" in target_units:
            mae_ns = mae * 1000.0
            rmse_ns = rmse * 1000.0
        elif target_units == "s":
            mae_ns = mae * 1e9
            rmse_ns = rmse * 1e9
        elif target_units == "ns":
            mae_ns = mae
            rmse_ns = rmse

        return ForecastMetrics(
            mae=mae,
            rmse=rmse,
            mean_bias_error=mbe,
            std_error=std_err,
            n_samples=n,
            target_units=target_units,
            target_name=target_name,
            mae_ns=mae_ns,
            rmse_ns=rmse_ns,
        )

    @staticmethod
    def calculate_3d_position_error(
        true_xyz_km: np.ndarray,
        pred_xyz_km: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute 3D Euclidean distance error in meters:
        norm(predicted_xyz - reference_xyz) * 1000.0
        """
        t_xyz = np.asarray(true_xyz_km, dtype=float)
        p_xyz = np.asarray(pred_xyz_km, dtype=float)

        if t_xyz.shape != p_xyz.shape or t_xyz.shape[1] != 3:
            raise ValueError(f"Expected shape (N, 3), got {t_xyz.shape} and {p_xyz.shape}")

        valid = ~np.isnan(t_xyz).any(axis=1) & ~np.isnan(p_xyz).any(axis=1)
        t_xyz = t_xyz[valid]
        p_xyz = p_xyz[valid]

        if len(t_xyz) == 0:
            return {
                "mean_position_error_m": np.nan,
                "rmse_position_error_m": np.nan,
                "max_position_error_m": np.nan,
                "p95_position_error_m": np.nan,
                "n_samples": 0,
            }

        diff_km = p_xyz - t_xyz
        diff_m = diff_km * 1000.0  # Convert km to meters
        errors_m = np.linalg.norm(diff_m, axis=1)

        return {
            "mean_position_error_m": float(np.mean(errors_m)),
            "rmse_position_error_m": float(np.sqrt(np.mean(errors_m**2))),
            "max_position_error_m": float(np.max(errors_m)),
            "p95_position_error_m": float(np.percentile(errors_m, 95)),
            "n_samples": len(errors_m),
        }
