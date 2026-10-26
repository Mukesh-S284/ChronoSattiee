"""
Tests for Synthetic Data Labeling, CSV Exports, and Application Imports (Points 18, 19, 20)
"""

import json
from pathlib import Path
import pytest

from src.data_ingestion.synthetic_generator import SyntheticGNSSGenerator
from src.reporting.exports import ExportManager


def test_synthetic_data_labeling_and_reproducibility() -> None:
    """Test 18: Verify synthetic generator explicitly labels all records and is reproducible."""
    gen1 = SyntheticGNSSGenerator(random_seed=123)
    df1 = gen1.generate_orbit_and_clock(sat_ids=["G01"], duration_hours=24)

    gen2 = SyntheticGNSSGenerator(random_seed=123)
    df2 = gen2.generate_orbit_and_clock(sat_ids=["G01"], duration_hours=24)

    # Must be identical with same seed
    assert len(df1) == len(df2)
    assert df1["clock_bias_us"].iloc[0] == df2["clock_bias_us"].iloc[0]

    # Must be explicitly labeled as synthetic
    assert bool(df1["is_synthetic"].all()) is True
    assert all(df1["data_source"] == "SYNTHETIC_SIMULATOR")


def test_csv_and_manifest_exports(tmp_path: Path) -> None:
    """Test 19: Verify CSV and JSON manifest exports generate well-formed files."""
    mgr = ExportManager(output_dir=tmp_path)

    # Forecasts CSV
    f_records = [{
        "satellite_id": "G01",
        "target_name": "clock_bias_us",
        "horizon_hours": 1,
        "predicted_value": 123.456,
        "units": "us",
    }]
    f_path = mgr.export_forecasts_csv(f_records, filename="test_forecasts.csv")
    assert f_path.exists()
    assert "G01" in f_path.read_text()

    # Manifest JSON
    manifest_data = {"dataset": "test", "is_synthetic": True, "records": 100}
    m_path = mgr.export_provenance_manifest(manifest_data, filename="test_manifest.json")
    assert m_path.exists()
    loaded = json.loads(m_path.read_text())
    assert loaded["is_synthetic"] is True


def test_streamlit_app_imports() -> None:
    """Test 20: Verify core application and page modules import without error."""
    import app
    from src.ui_state import init_session_state
    from src.visualization import plots
    from src.models import baselines, forecasting, registry

    assert hasattr(app, "st")
    assert hasattr(plots, "plot_orbit_3d")
    assert hasattr(forecasting, "ForecastingEngine")
