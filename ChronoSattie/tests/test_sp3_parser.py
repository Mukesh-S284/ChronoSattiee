"""
Tests for SP3 Parser (Points 1, 3, 6, 8)
"""

from pathlib import Path
import numpy as np
import pytest

from src.data_ingestion.sp3_parser import SP3Parser

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_test.sp3"


def test_sp3_parsing_fixture() -> None:
    """Test 1: Verify correct parsing of SP3 format test fixture."""
    parser = SP3Parser()
    df, meta = parser.parse_file(FIXTURE_PATH)

    assert not df.empty
    assert len(df) == 6  # 3 satellites * 2 epochs
    assert set(df["sat_id"].unique()) == {"G01", "G02", "G03"}
    assert meta.agency == "IGS"
    assert meta.coordinate_system == "IGS14"
    assert meta.epoch_interval_s == 900.0

    # Check G01 coordinates at first epoch
    g01_first = df[(df["sat_id"] == "G01") & (df["epoch"] == "2026-04-28 00:00:00")].iloc[0]
    assert pytest.approx(g01_first["x_km"], 1e-4) == -13421.432981
    assert pytest.approx(g01_first["clock_bias_us"], 1e-4) == 123.456789
    assert bool(g01_first["is_synthetic"]) is False


def test_sp3_invalid_file_handling() -> None:
    """Test 3: Verify graceful handling and informative errors on invalid or missing files."""
    parser = SP3Parser()

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        parser.parse_file("non_existent_file_path_12345.sp3")

    # Empty content
    import io
    empty_stream = io.StringIO("")
    with pytest.raises(ValueError, match="empty or unreadable"):
        parser.parse_file(empty_stream)

    # Malformed non-SP3 header
    invalid_stream = io.StringIO("RANDOM TEXT HEADER WITHOUT HASH\nMORE TEXT")
    with pytest.raises(ValueError, match="First line does not start with '#'"):
        parser.parse_file(invalid_stream)


def test_sp3_missing_value_sentinel() -> None:
    """Test 6: Verify 999999.999999 sentinel in SP3 is converted to NaN."""
    parser = SP3Parser()
    df, _ = parser.parse_file(FIXTURE_PATH)

    # G03 at first epoch has clock = 999999.999999
    g03_first = df[(df["sat_id"] == "G03") & (df["epoch"] == "2026-04-28 00:00:00")].iloc[0]
    assert np.isnan(g03_first["clock_bias_us"])


def test_sp3_satellite_filtering() -> None:
    """Test 8: Verify satellite selection and filtering."""
    parser = SP3Parser()
    df, _ = parser.parse_file(FIXTURE_PATH)

    g01_only = df[df["sat_id"] == "G01"]
    assert len(g01_only) == 2
    assert all(g01_only["sat_id"] == "G01")
