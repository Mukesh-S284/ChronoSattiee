"""
Time-series feature engineering and leakage-safe preprocessing.
Creates backward-looking lagged features, gap-aware velocities/drifts, and periodic orbital terms.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import GPS_ORBITAL_PERIOD_S, SP3_NOMINAL_INTERVAL_S


@dataclass
class FeaturePipelineConfig:
    """Configuration for feature extraction and lag creation."""
    target_col: str = "clock_bias_us"
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 8])
    include_derivatives: bool = True
    include_periodic: bool = True
    include_auxiliary_coords: bool = True
    max_gap_multiplier: float = 1.5
    nominal_dt_s: float = SP3_NOMINAL_INTERVAL_S


class FeatureEngineer:
    """
    Builds strictly causal, leakage-safe features for GNSS satellite time-series forecasting.
    """

    def __init__(self, config: Optional[FeaturePipelineConfig] = None) -> None:
        self.config = config or FeaturePipelineConfig()
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []

    def create_features_and_targets(
        self,
        df: pd.DataFrame,
        horizon_steps: int = 4,  # e.g. 4 steps = 1 hour at 15-min sampling
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Extract feature matrix X and target vector y for a specific forecast horizon.

        Args:
            df: Cleaned DataFrame for a single satellite, sorted chronologically.
            horizon_steps: Number of forward epochs for the target.

        Returns:
            X: DataFrame of feature inputs (at forecast origin t)
            y: Series of target values (at future epoch t + horizon_steps)
            feature_names: List of column names in X
        """
        if df.empty or len(df) <= max(self.config.lags) + horizon_steps:
            raise ValueError(
                f"Insufficient data ({len(df)} rows). Requires at least "
                f"{max(self.config.lags) + horizon_steps + 1} observations."
            )

        df_feat = df.copy().reset_index(drop=True)
        target_col = self.config.target_col

        if target_col not in df_feat.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe.")

        # Compute sampling delta t
        dt_seconds = df_feat["epoch"].diff().dt.total_seconds()
        median_dt = float(dt_seconds.median()) if not dt_seconds.dropna().empty else self.config.nominal_dt_s
        gap_limit = median_dt * self.config.max_gap_multiplier

        feature_cols: List[str] = []

        # 1. Backward Lags (Causal Only: t - lag)
        for lag in self.config.lags:
            col_name = f"{target_col}_lag_{lag}"
            df_feat[col_name] = df_feat[target_col].shift(lag)
            feature_cols.append(col_name)

        # 2. Derivative / Drift Estimates (Gap-Aware)
        if self.config.include_derivatives:
            diff_col = f"{target_col}_drift_1step"
            # Difference: y(t) - y(t-1)
            raw_diff = df_feat[target_col] - df_feat[target_col].shift(1)
            # Gap check: if dt > gap_limit, set drift to NaN
            is_gap = dt_seconds > gap_limit
            raw_diff[is_gap] = np.nan
            # Rate of change per second
            df_feat[diff_col] = raw_diff / dt_seconds.replace(0, np.nan)
            feature_cols.append(diff_col)

            # Acceleration / curvature: drift(t) - drift(t-1)
            accel_col = f"{target_col}_accel"
            raw_accel = df_feat[diff_col] - df_feat[diff_col].shift(1)
            df_feat[accel_col] = raw_accel
            feature_cols.append(accel_col)

        # 3. Auxiliary Coordinate Features (if forecasting clock, or position lags)
        if self.config.include_auxiliary_coords:
            for coord in ["x_km", "y_km", "z_km"]:
                if coord in df_feat.columns and coord != target_col:
                    lag_col = f"{coord}_lag_1"
                    df_feat[lag_col] = df_feat[coord].shift(1)
                    feature_cols.append(lag_col)
                    # Coordinate velocity
                    vel_col = f"{coord}_vel_1step"
                    raw_vel = (df_feat[coord] - df_feat[coord].shift(1)) / dt_seconds.replace(0, np.nan)
                    raw_vel[is_gap] = np.nan
                    df_feat[vel_col] = raw_vel
                    feature_cols.append(vel_col)

        # 4. Periodic Orbital and Diurnal Features
        if self.config.include_periodic:
            epoch_dt = pd.to_datetime(df_feat["epoch"])
            # Seconds since reference start
            start_epoch = epoch_dt.iloc[0]
            elapsed_s = (epoch_dt - start_epoch).dt.total_seconds()

            # MEO Orbital period harmonic (~11h 58m = 43082 s)
            orb_phase = (2.0 * np.pi * elapsed_s) / GPS_ORBITAL_PERIOD_S
            df_feat["sin_orbit_phase"] = np.sin(orb_phase)
            df_feat["cos_orbit_phase"] = np.cos(orb_phase)
            feature_cols.extend(["sin_orbit_phase", "cos_orbit_phase"])

            # 24-hour Diurnal cycle
            hour_val = epoch_dt.dt.hour + epoch_dt.dt.minute / 60.0
            diurnal_phase = (2.0 * np.pi * hour_val) / 24.0
            df_feat["sin_diurnal"] = np.sin(diurnal_phase)
            df_feat["cos_diurnal"] = np.cos(diurnal_phase)
            feature_cols.extend(["sin_diurnal", "cos_diurnal"])

        # 5. Target Forward Shift: y(t + horizon_steps)
        # Note: shifted negatively by horizon_steps so that row t contains y(t + H)
        target_series = df_feat[target_col].shift(-horizon_steps)
        future_epoch = df_feat["epoch"].shift(-horizon_steps)

        # Verification of temporal separation: ensure target epoch is actually horizon_steps ahead
        # and hasn't jumped across a massive gap
        expected_duration_s = horizon_steps * median_dt
        actual_duration_s = (future_epoch - df_feat["epoch"]).dt.total_seconds()
        # Allow +/- 30% tolerance for non-uniform sampling
        valid_horizon = (actual_duration_s >= 0.7 * expected_duration_s) & (
            actual_duration_s <= 1.3 * expected_duration_s
        )
        target_series[~valid_horizon] = np.nan

        # Combine and drop rows where features or target are NaN
        valid_mask = ~target_series.isna() & ~df_feat[feature_cols].isna().any(axis=1)

        X = df_feat.loc[valid_mask, feature_cols].copy()
        y = target_series.loc[valid_mask].copy()

        # Preserve metadata on the X dataframe for indexing and provenance
        X["epoch"] = df_feat.loc[valid_mask, "epoch"].values
        X["target_epoch"] = future_epoch.loc[valid_mask].values
        X["sat_id"] = df_feat.loc[valid_mask, "sat_id"].values

        self.feature_columns = feature_cols
        return X, y, feature_cols

    def fit_scaler(self, X_train: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
        """Fit standard scaler exclusively on training feature matrix."""
        self.scaler = StandardScaler()
        self.scaler.fit(X_train[feature_cols])
        return self.scaler

    def transform(self, X: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Transform feature matrix using fitted scaler."""
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted on training data yet.")
        return self.scaler.transform(X[feature_cols])
