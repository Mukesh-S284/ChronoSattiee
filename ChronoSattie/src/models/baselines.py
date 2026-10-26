"""
Baseline forecasting models for satellite clock and ephemeris time-series.
Provides Persistence (last-value) and Linear Trend extrapolation baselines.
"""

from typing import Optional
import numpy as np
import pandas as pd


class PersistenceForecaster:
    """
    Persistence baseline: Assumes the future observation y(t + h) equals the last observation y(t).
    In GNSS operations, persistence is a mandatory reference benchmark.
    """

    def __init__(self) -> None:
        self.last_value: Optional[float] = None
        self.target_name: str = "target"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceForecaster":
        """
        Fit records the most recent training observation.
        """
        if len(y) > 0:
            self.last_value = float(y[-1])
        return self

    def predict(self, X: np.ndarray, last_observed_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict returns the last observed value for each sample.
        If last_observed_values array is provided, uses per-sample last observation.
        """
        n_samples = len(X)
        if last_observed_values is not None:
            return np.array(last_observed_values, dtype=float)
        if self.last_value is not None:
            return np.full(n_samples, self.last_value, dtype=float)
        return np.zeros(n_samples, dtype=float)


class LinearDriftForecaster:
    """
    Linear Drift baseline: Fits a first-order polynomial (slope and intercept)
    on the recent observation window to extrapolate future values.
    """

    def __init__(self) -> None:
        self.slope: float = 0.0
        self.intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearDriftForecaster":
        """Fit linear regression on time indices."""
        n = len(y)
        if n < 2:
            self.intercept = float(y[0]) if n == 1 else 0.0
            self.slope = 0.0
            return self

        t = np.arange(n, dtype=float)
        # First-order polynomial fit
        poly = np.polyfit(t, y, deg=1)
        self.slope = float(poly[0])
        self.intercept = float(poly[1])
        return self

    def predict(self, steps_ahead: int) -> float:
        """Extrapolate value at a specific forward step."""
        return self.intercept + self.slope * steps_ahead
