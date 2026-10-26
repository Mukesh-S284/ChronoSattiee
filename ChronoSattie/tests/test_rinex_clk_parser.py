"""
Tests for RINEX CLK Parser (Points 2, 7)
"""

from pathlib import Path
import pytest

from src.data_ingestion.rinex_clk_parser import RINEXCLKParser

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_test.clk"


def test_rinex_clk_parsing_fixture() -> None:
    """Test 2: Verify correct parsing of RINEX CLK test fixture."""
    parser = RINEXCLKParser()
    df, meta = parser.parse_file(FIXTURE_PATH)

    assert not df.empty
    assert len(df) == 4
    assert set(df["sat_id"].unique()) == {"G01", "G02"}
    assert meta.time_system == "GPS"
    assert meta.file_type == "C"

    # Verify first row
    first_row = df[(df["sat_id"] == "G01") & (df["epoch"] == "2026-04-28 00:00:00")].iloc[0]
    assert pytest.approx(first_row["clock_bias_s"], 1e-12) == 0.123456789012e-3


def test_rinex_clk_unit_conversion() -> None:
    """Test 7: Verify clock bias unit conversions between seconds and nanoseconds."""
    parser = RINEXCLKParser()
    df, _ = parser.parse_file(FIXTURE_PATH)

    first_row = df.iloc[0]
    expected_ns = first_row["clock_bias_s"] * 1e9
    assert pytest.approx(first_row["clock_bias_ns"], 1e-3) == expected_ns
