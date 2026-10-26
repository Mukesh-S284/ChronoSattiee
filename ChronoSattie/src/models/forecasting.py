"""
Unified forecasting engine supporting direct multi-step forecasting for GNSS satellite targets.
Implements Persistence, Linear Regression, Random Forest, and HistGradientBoosting models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.models.baselines import PersistenceForecaster
from src.preprocessing.features import FeatureEngineer, FeaturePipelineConfig


@dataclass
class ForecastResult:
    """Encapsulates the complete result of a forecast run."""
    sat_id: str
    target_name: str
    model_name: str
    horizon_hours: int
    horizon_steps: int
    forecast_origin: pd.Timestamp
    target_epoch: pd.Timestamp
    prediction: float
    actual: Optional[float] = None
    residual: Optional[float] = None
    units: str = "us"
    feature_importance: Optional[Dict[str, float]] = None
    is_synthetic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ForecastingEngine:
    """
    Orchestrates time-series model training and multi-horizon forecasting.
    """

    SUPPORTED_MODELS = [
        "persistence",
        "linear_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]

    def __init__(
        self,
        model_name: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {self.SUPPORTED_MODELS}")

        self.model_name = model_name
        self.model_params = model_params or {}
        self.random_state = random_state
        self.estimator: Any = None
        self.feature_names: List[str] = []
        self.target_name: str = "clock_bias_us"
        self.fitted_horizon_hours: int = 1
        self.fitted_horizon_steps: int = 4
        self.sat_id: str = "G01"
        self.is_fitted: bool = False

    def _init_estimator(self) -> Any:
        """Instantiate scikit-learn or baseline regressor."""
        params = self.model_params.copy()

        if self.model_name == "persistence":
            return PersistenceForecaster()

        elif self.model_name == "linear_regression":
            return LinearRegression(**params)

        elif self.model_name == "random_forest":
            default_rf = {"n_estimators": 100, "max_depth": 12, "random_state": self.random_state, "n_jobs": -1}
            default_rf.update(params)
            return RandomForestRegressor(**default_rf)

        elif self.model_name == "hist_gradient_boosting":
            default_hgb = {"max_iter": 100, "max_depth": 8, "random_state": self.random_state}
            default_hgb.update(params)
            return HistGradientBoostingRegressor(**default_hgb)

        raise ValueError(f"Unsupported model: {self.model_name}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: List[str],
        target_name: str = "clock_bias_us",
        sat_id: str = "G01",
        horizon_hours: int = 1,
        horizon_steps: int = 4,
    ) -> "ForecastingEngine":
        """
        Fit model on chronological training split.
        """
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training dataset cannot be empty.")

        self.target_name = target_name
        self.sat_id = sat_id
        self.fitted_horizon_hours = horizon_hours
        self.fitted_horizon_steps = horizon_steps
        self.feature_names = feature_cols
        self.estimator = self._init_estimator()

        X_mat = X_train[feature_cols].values
        y_vec = y_train.values

        if self.model_name == "persistence":
            self.estimator.fit(X_mat, y_vec)
        else:
            self.estimator.fit(X_mat, y_vec)

        self.is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate predictions for feature rows.
        """
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError("ForecastingEngine must be fitted before predict() is called.")

        X_mat = X[self.feature_names].values

        if self.model_name == "persistence":
            # For persistence, the last observed value is lag_1 of the target
            lag1_col = f"{self.target_name}_lag_1"
            if lag1_col in X.columns:
                return self.estimator.predict(X_mat, last_observed_values=X[lag1_col].values)
            return self.estimator.predict(X_mat)

        return self.estimator.predict(X_mat)

    def forecast_latest(
        self,
        latest_row: pd.DataFrame,
        horizon_hours: int,
        nominal_dt_s: float = 900.0,
    ) -> ForecastResult:
        """
        Generate a future forecast from the single most recent observation origin.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        preds = self.predict(latest_row)
        pred_val = float(preds[0])

        origin_epoch = pd.to_datetime(latest_row["epoch"].iloc[0])
        # Compute future target epoch based on horizon hours
        target_epoch = origin_epoch + pd.Timedelta(hours=horizon_hours)

        # Determine target units
        units = "us"
        if "km" in self.target_name:
            units = "km"
        elif "ns" in self.target_name:
            units = "ns"
        elif "s" in self.target_name:
            units = "s"

        feat_imp = None
        if hasattr(self.estimator, "feature_importances_"):
            feat_imp = dict(zip(self.feature_names, [float(x) for x in self.estimator.feature_importances_]))

        return ForecastResult(
            sat_id=self.sat_id,
            target_name=self.target_name,
            model_name=self.model_name,
            horizon_hours=horizon_hours,
            horizon_steps=self.fitted_horizon_steps,
            forecast_origin=origin_epoch,
            target_epoch=target_epoch,
            prediction=pred_val,
            units=units,
            feature_importance=feat_imp,
            is_synthetic=bool(latest_row.get("is_synthetic", [False])[0]),
        )
