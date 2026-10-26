"""
Model persistence and registry management using joblib.
Stores trained estimators, metadata, feature column configurations, and performance metrics.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

from src.config import MODELS_DIR


@dataclass
class ModelMetadata:
    """Metadata recorded for each persisted model."""
    model_id: str
    model_name: str
    target_name: str
    sat_id: str
    horizon_hours: int
    horizon_steps: int
    feature_names: List[str]
    train_start: str
    train_end: str
    n_samples: int
    metrics: Dict[str, float]
    is_synthetic: bool
    created_at: str


class ModelRegistry:
    """
    Registry for saving, loading, listing, and inspecting forecasting models.
    """

    def __init__(self, registry_dir: Optional[Path] = None) -> None:
        self.registry_dir = registry_dir or MODELS_DIR
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        engine: Any,
        metadata: ModelMetadata,
        scaler: Optional[Any] = None,
    ) -> Path:
        """
        Persist trained ForecastingEngine, scaler, and metadata.
        """
        model_filename = f"{metadata.model_id}.joblib"
        save_path = self.registry_dir / model_filename

        bundle = {
            "estimator": engine.estimator,
            "model_name": engine.model_name,
            "target_name": engine.target_name,
            "sat_id": engine.sat_id,
            "fitted_horizon_hours": engine.fitted_horizon_hours,
            "fitted_horizon_steps": engine.fitted_horizon_steps,
            "feature_names": engine.feature_names,
            "metadata": asdict(metadata),
            "scaler": scaler,
        }

        joblib.dump(bundle, save_path)
        return save_path

    def load_model(self, model_id: str) -> Dict[str, Any]:
        """
        Load persisted model bundle by ID.
        """
        path = self.registry_dir / f"{model_id}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        bundle = joblib.load(path)
        return bundle

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models and their metadata.
        """
        results: List[Dict[str, Any]] = []
        for file in self.registry_dir.glob("*.joblib"):
            try:
                bundle = joblib.load(file)
                meta = bundle.get("metadata", {})
                meta["file_path"] = str(file)
                results.append(meta)
            except Exception:
                continue

        # Sort by creation date descending
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results

    def delete_model(self, model_id: str) -> bool:
        """Delete a model file from registry."""
        path = self.registry_dir / f"{model_id}.joblib"
        if path.exists():
            path.unlink()
            return True
        return False
