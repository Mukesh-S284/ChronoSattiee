"""
Chronological time-series splitting and rolling-origin walk-forward validation.
Strictly prevents future-data leakage and handles multi-horizon evaluation.
"""

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.evaluation.metrics import ForecastMetrics, MetricsCalculator
from src.models.baselines import PersistenceForecaster
from src.models.forecasting import ForecastingEngine
from src.preprocessing.features import FeatureEngineer, FeaturePipelineConfig


@dataclass
class ValidationSplit:
    """Chronological dataset split with exact timestamp bounds."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_ratio: float


class TimeSeriesValidator:
    """
    Manages leakage-safe chronological validation and rolling-origin evaluation.
    """

    @staticmethod
    def chronological_train_test_split(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
    ) -> ValidationSplit:
        """
        Split a chronologically sorted DataFrame into training and testing partitions.
        Strictly forbids random shuffling.
        """
        if df.empty:
            raise ValueError("Cannot split empty DataFrame.")

        df_sorted = df.sort_values(by="epoch").reset_index(drop=True)
        n = len(df_sorted)
        split_idx = int(n * train_ratio)

        if split_idx == 0 or split_idx == n:
            raise ValueError(f"Invalid split index {split_idx} for dataset of length {n}.")

        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()

        return ValidationSplit(
            train_df=train_df,
            test_df=test_df,
            train_start=train_df["epoch"].min(),
            train_end=train_df["epoch"].max(),
            test_start=test_df["epoch"].min(),
            test_end=test_df["epoch"].max(),
            train_ratio=train_ratio,
        )

    @staticmethod
    def rolling_origin_splits(
        df: pd.DataFrame,
        n_splits: int = 4,
        test_horizon_steps: int = 4,
        min_train_steps: int = 24,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Yields successive walk-forward chronological splits (expanding window).
        """
        df_sorted = df.sort_values(by="epoch").reset_index(drop=True)
        n = len(df_sorted)

        available_test_room = n - min_train_steps
        if available_test_room < test_horizon_steps:
            raise ValueError(
                f"Insufficient data ({n} rows) for rolling origin with "
                f"min_train={min_train_steps} and test_steps={test_horizon_steps}."
            )

        step_size = max(1, (available_test_room - test_horizon_steps) // n_splits)

        for i in range(n_splits):
            origin_idx = min_train_steps + i * step_size
            if origin_idx + test_horizon_steps > n:
                break
            train_part = df_sorted.iloc[:origin_idx]
            test_part = df_sorted.iloc[origin_idx : origin_idx + test_horizon_steps]
            yield train_part, test_part

    @classmethod
    def evaluate_model_at_horizon(
        cls,
        df: pd.DataFrame,
        sat_id: str,
        target_col: str,
        model_name: str,
        horizon_hours: int,
        horizon_steps: int,
        train_ratio: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Run end-to-end chronological evaluation for a chosen model and horizon.
        Compares model against the Persistence baseline.
        """
        sat_df = df[df["sat_id"] == sat_id].copy().sort_values(by="epoch").reset_index(drop=True)

        min_required = 30 + horizon_steps
        if len(sat_df) < min_required:
            return {
                "status": "UNAVAILABLE",
                "reason": f"Insufficient observations for {sat_id} at {horizon_hours}h horizon (has {len(sat_df)}, needs >={min_required}).",
                "horizon_hours": horizon_hours,
            }

        split = cls.chronological_train_test_split(sat_df, train_ratio=train_ratio)

        # Feature engineering
        feat_engineer = FeatureEngineer(FeaturePipelineConfig(target_col=target_col))

        try:
            X_train, y_train, feat_cols = feat_engineer.create_features_and_targets(
                split.train_df, horizon_steps=horizon_steps
            )
            X_test, y_test, _ = feat_engineer.create_features_and_targets(
                split.test_df, horizon_steps=horizon_steps
            )
        except ValueError as e:
            return {
                "status": "UNAVAILABLE",
                "reason": str(e),
                "horizon_hours": horizon_hours,
            }

        if len(X_train) < 10 or len(X_test) < 2:
            return {
                "status": "UNAVAILABLE",
                "reason": f"Too few feature samples after lag creation (Train: {len(X_train)}, Test: {len(X_test)}).",
                "horizon_hours": horizon_hours,
            }

        # Fit model
        engine = ForecastingEngine(model_name=model_name)
        engine.fit(
            X_train=X_train,
            y_train=y_train,
            feature_cols=feat_cols,
            target_name=target_col,
            sat_id=sat_id,
            horizon_hours=horizon_hours,
            horizon_steps=horizon_steps,
        )

        # Predict
        y_pred = engine.predict(X_test)

        # Persistence Baseline
        persistence_engine = ForecastingEngine(model_name="persistence")
        persistence_engine.fit(
            X_train=X_train,
            y_train=y_train,
            feature_cols=feat_cols,
            target_name=target_col,
            sat_id=sat_id,
            horizon_hours=horizon_hours,
            horizon_steps=horizon_steps,
        )
        y_pred_pers = persistence_engine.predict(X_test)

        # Determine units
        units = "us" if "us" in target_col else ("km" if "km" in target_col else ("s" if "_s" in target_col else "ns"))

        model_metrics = MetricsCalculator.calculate_scalar_metrics(
            y_true=y_test, y_pred=y_pred, target_name=target_col, target_units=units
        )
        baseline_metrics = MetricsCalculator.calculate_scalar_metrics(
            y_true=y_test, y_pred=y_pred_pers, target_name=target_col, target_units=units
        )

        return {
            "status": "SUCCESS",
            "sat_id": sat_id,
            "target_col": target_col,
            "model_name": model_name,
            "horizon_hours": horizon_hours,
            "horizon_steps": horizon_steps,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "model_metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
            "y_test": y_test.values,
            "y_pred": y_pred,
            "y_pred_baseline": y_pred_pers,
            "test_epochs": X_test["target_epoch"].values,
            "origin_epochs": X_test["epoch"].values,
        }
