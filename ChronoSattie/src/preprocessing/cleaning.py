"""
Data cleaning, gap identification, duplicate handling, and physical bounds validation.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import SP3_NOMINAL_INTERVAL_S


@dataclass
class DataCleaningReport:
    """Detailed summary of data quality, missing epochs, and cleaning actions."""
    initial_rows: int = 0
    final_rows: int = 0
    duplicates_removed: int = 0
    missing_value_rows: int = 0
    detected_nominal_dt_s: float = 900.0
    gap_count: int = 0
    max_gap_duration_s: float = 0.0
    outliers_detected: int = 0
    satellites_present: List[str] = field(default_factory=list)
    time_range_start: Optional[pd.Timestamp] = None
    time_range_end: Optional[pd.Timestamp] = None
    warnings: List[str] = field(default_factory=list)


class DataCleaner:
    """
    Cleans raw GNSS observations while preserving scientific physical integrity.
    """

    # Physical bounds for GNSS satellite orbits (MEO / GEO / IGSO)
    MIN_ORBIT_RADIUS_KM = 6000.0   # Must be higher than Earth surface (~6378 km, LEO excluded)
    MAX_ORBIT_RADIUS_KM = 45000.0  # Above GEO radius (~42164 km)
    MAX_CLOCK_BIAS_ABS_S = 0.1     # Max 100 ms for synchronized atomic clocks

    def __init__(
        self,
        nominal_dt_s: float = SP3_NOMINAL_INTERVAL_S,
        gap_multiplier_threshold: float = 1.5,
    ) -> None:
        self.nominal_dt_s = nominal_dt_s
        self.gap_multiplier_threshold = gap_multiplier_threshold

    def clean_dataset(
        self,
        df: pd.DataFrame,
        sat_id: Optional[str] = None,
        filter_outliers: bool = False,
    ) -> Tuple[pd.DataFrame, DataCleaningReport]:
        """
        Validate, chronologically sort, deduplicate, and clean GNSS observations.

        Args:
            df: Raw DataFrame with ['epoch', 'sat_id', ...]
            sat_id: Optional satellite to isolate. If None, process all satellites.
            filter_outliers: If True, replace statistical outliers with NaN; if False, only flag.

        Returns:
            Tuple of (cleaned_df, DataCleaningReport)
        """
        report = DataCleaningReport()
        if df.empty:
            report.warnings.append("Input DataFrame is empty.")
            return df.copy(), report

        df_work = df.copy()
        report.initial_rows = len(df_work)

        # 1. Satellite Filtering
        if sat_id is not None:
            df_work = df_work[df_work["sat_id"] == sat_id].copy()
            if df_work.empty:
                report.warnings.append(f"Satellite '{sat_id}' not found in dataset.")
                return df_work, report

        report.satellites_present = sorted(df_work["sat_id"].unique().tolist())

        # 2. Timestamp Validation & Chronological Sorting
        df_work["epoch"] = pd.to_datetime(df_work["epoch"])
        df_work = df_work.sort_values(by=["sat_id", "epoch"]).reset_index(drop=True)
        report.time_range_start = df_work["epoch"].min()
        report.time_range_end = df_work["epoch"].max()

        # 3. Deduplication
        dup_mask = df_work.duplicated(subset=["sat_id", "epoch"], keep="first")
        num_duplicates = int(dup_mask.sum())
        if num_duplicates > 0:
            report.duplicates_removed = num_duplicates
            report.warnings.append(f"Detected and dropped {num_duplicates} duplicate epoch records.")
            df_work = df_work[~dup_mask].reset_index(drop=True)

        # 4. Physical Bounds Validation
        # Check coordinates radius: r = sqrt(x^2 + y^2 + z^2)
        if all(col in df_work.columns for col in ["x_km", "y_km", "z_km"]):
            radius = np.sqrt(df_work["x_km"]**2 + df_work["y_km"]**2 + df_work["z_km"]**2)
            invalid_radius = (radius < self.MIN_ORBIT_RADIUS_KM) | (radius > self.MAX_ORBIT_RADIUS_KM)
            num_bad_radius = int(invalid_radius.sum())
            if num_bad_radius > 0:
                report.warnings.append(f"Found {num_bad_radius} coordinate records outside physical GNSS orbit bounds [6000, 45000] km.")
                df_work.loc[invalid_radius, ["x_km", "y_km", "z_km"]] = np.nan

        # Check clock bias
        if "clock_bias_s" in df_work.columns:
            invalid_clk = df_work["clock_bias_s"].abs() > self.MAX_CLOCK_BIAS_ABS_S
            num_bad_clk = int(invalid_clk.sum())
            if num_bad_clk > 0:
                report.warnings.append(f"Found {num_bad_clk} clock records exceeding {self.MAX_CLOCK_BIAS_ABS_S}s limit.")
                df_work.loc[invalid_clk, "clock_bias_s"] = np.nan

        # 5. Gap and Sampling Interval Analysis
        gap_count = 0
        max_gap = 0.0

        for sat in df_work["sat_id"].unique():
            sat_df = df_work[df_work["sat_id"] == sat]
            if len(sat_df) > 1:
                dt_series = sat_df["epoch"].diff().dt.total_seconds().dropna()
                median_dt = float(dt_series.median())
                if report.detected_nominal_dt_s == 900.0 and median_dt > 0:
                    report.detected_nominal_dt_s = median_dt

                gap_mask = dt_series > (median_dt * self.gap_multiplier_threshold)
                gap_count += int(gap_mask.sum())
                if not dt_series.empty:
                    max_gap = max(max_gap, float(dt_series.max()))

        report.gap_count = gap_count
        report.max_gap_duration_s = max_gap
        if gap_count > 0:
            report.warnings.append(f"Identified {gap_count} observation gaps (longest gap: {max_gap:.0f}s).")

        # 6. Statistical Outlier Detection (Hampel Filter / Rolling Median)
        outlier_count = 0
        for target_col in ["clock_bias_us", "clock_bias_ns", "x_km", "y_km", "z_km"]:
            if target_col in df_work.columns:
                series = df_work[target_col].dropna()
                if len(series) >= 10:
                    med = series.median()
                    mad = (series - med).abs().median()
                    if mad > 1e-9:
                        threshold = 4.5 * 1.4826 * mad
                        outliers = (series - med).abs() > threshold
                        n_out = int(outliers.sum())
                        outlier_count += n_out
                        if filter_outliers and n_out > 0:
                            df_work.loc[series[outliers].index, target_col] = np.nan

        report.outliers_detected = outlier_count
        report.final_rows = len(df_work)
        report.missing_value_rows = int(df_work.isna().any(axis=1).sum())

        return df_work, report
