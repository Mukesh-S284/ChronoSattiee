"""
Report and data export utilities for forecasts, metrics, and data provenance manifests.
Generates structured CSV and JSON outputs preserving timestamps, units, and satellite IDs.
"""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import REPORTS_DIR


class ExportManager:
    """
    Manages export generation for scientific forecasts, evaluation reports, and manifests.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_forecasts_csv(
        self,
        forecast_records: List[Dict[str, Any]],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export generated predictions to CSV.
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"satellite_forecasts_{ts}.csv"

        path = self.output_dir / filename
        df = pd.DataFrame(forecast_records)
        df.to_csv(path, index=False)
        return path

    def export_metrics_csv(
        self,
        metrics_records: List[Dict[str, Any]],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export model evaluation metrics to CSV.
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_metrics_{ts}.csv"

        path = self.output_dir / filename
        df = pd.DataFrame(metrics_records)
        df.to_csv(path, index=False)
        return path

    def export_processed_data_csv(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export processed GNSS time series observations to CSV.
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_observations_{ts}.csv"

        path = self.output_dir / filename
        df.to_csv(path, index=False)
        return path

    def export_provenance_manifest(
        self,
        dataset_info: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export dataset provenance and verification manifest as JSON.
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"data_manifest_{ts}.json"

        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, default=str)
        return path
