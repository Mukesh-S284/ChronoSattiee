"""
Parser for RINEX Clock (CLK) format files (version 2.xx and 3.0x).
Extracts satellite clock biases (seconds and nanoseconds), sigmas, epochs, and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class RINEXCLKMetadata:
    """Metadata extracted from the RINEX CLK file header."""
    version: str = "3.00"
    file_type: str = "C"
    agency: str = "UNKNOWN"
    time_system: str = "GPS"
    satellites: List[str] = field(default_factory=list)


class RINEXCLKParser:
    """
    Parser for RINEX Clock files (focusing on 'AS' satellite clock records).
    """

    MAX_REASONABLE_CLOCK_BIAS_S = 0.5  # GNSS satellite clock biases are typically < a few milliseconds

    def __init__(self) -> None:
        self.metadata = RINEXCLKMetadata()

    def parse_file(self, file_path_or_buffer: Union[str, Path, object]) -> Tuple[pd.DataFrame, RINEXCLKMetadata]:
        """
        Parse a RINEX CLK file or string/buffer into a DataFrame and metadata.

        Returns:
            Tuple of:
            - DataFrame with columns: ['epoch', 'sat_id', 'clock_bias_s', 'clock_bias_ns', 'sigma_s', 'is_synthetic', 'data_source']
            - RINEXCLKMetadata dataclass
        """
        lines: List[str] = []

        if isinstance(file_path_or_buffer, (str, Path)):
            path = Path(file_path_or_buffer)
            if not path.exists():
                raise FileNotFoundError(f"RINEX CLK file does not exist: {path}")
            data_source = path.name
            if path.suffix.lower() == ".gz" or str(path).lower().endswith(".clk.gz"):
                import gzip
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            else:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
        else:
            data_source = getattr(file_path_or_buffer, "name", "uploaded_clk")
            content = file_path_or_buffer.read()
            if isinstance(content, bytes) and content.startswith(b"\x1f\x8b"):
                import gzip
                content = gzip.decompress(content).decode("utf-8", errors="replace")
            elif isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            lines = content.splitlines()

        if not lines:
            raise ValueError(f"RINEX CLK file '{data_source}' is empty.")

        self.metadata = self._parse_header(lines)

        records: List[Dict[str, Union[datetime, str, float, bool]]] = []
        satellites_seen = set()

        in_data = False
        for line in lines:
            if "END OF HEADER" in line:
                in_data = True
                continue

            if not in_data and not any("RINEX" in l for l in lines[:5]):
                # If no formal header marker is found, begin parsing data records directly
                in_data = True

            if in_data:
                line_str = line.strip()
                if not line_str or not line_str.startswith("AS"):
                    continue

                record = self._parse_as_record(line_str, data_source)
                if record is not None:
                    records.append(record)
                    satellites_seen.add(record["sat_id"])

        if not records:
            raise ValueError(f"No satellite clock records ('AS') found in RINEX CLK file '{data_source}'.")

        self.metadata.satellites = sorted(list(satellites_seen))

        df = pd.DataFrame(records)
        df["epoch"] = pd.to_datetime(df["epoch"])
        df["sat_id"] = df["sat_id"].astype(str)
        df["clock_bias_s"] = pd.to_numeric(df["clock_bias_s"], errors="coerce")
        df["clock_bias_ns"] = pd.to_numeric(df["clock_bias_ns"], errors="coerce")
        df["sigma_s"] = pd.to_numeric(df["sigma_s"], errors="coerce")
        df["is_synthetic"] = df["is_synthetic"].astype(bool)
        df["data_source"] = df["data_source"].astype(str)

        # Sort chronologically by satellite and epoch
        df = df.sort_values(by=["sat_id", "epoch"]).reset_index(drop=True)

        return df, self.metadata

    def _parse_header(self, lines: List[str]) -> RINEXCLKMetadata:
        """Parse header lines to extract version and agency."""
        meta = RINEXCLKMetadata()
        for line in lines[:40]:
            if "RINEX VERSION / TYPE" in line:
                parts = line[:60].split()
                if len(parts) >= 1:
                    meta.version = parts[0]
                if len(parts) >= 2:
                    meta.file_type = parts[1]
            elif "SYS / # / DATA TYPE" in line or "TIME SYSTEM ID" in line:
                if "GPS" in line:
                    meta.time_system = "GPS"
                elif "UTC" in line:
                    meta.time_system = "UTC"
                elif "GLO" in line:
                    meta.time_system = "GLO"
                elif "GAL" in line:
                    meta.time_system = "GAL"
            elif "END OF HEADER" in line:
                break
        return meta

    def _parse_as_record(
        self, line: str, data_source: str
    ) -> Optional[Dict[str, Union[datetime, str, float, bool]]]:
        """
        Parse an 'AS' record line.
        Format: AS  SatID  YYYY MM DD HH MM SS.ssssssss  N  ClockBias [Sigma]
        """
        parts = line.split()
        if len(parts) < 9:
            return None

        # parts[0] is 'AS'
        sat_id = parts[1]
        if sat_id.isdigit():
            sat_id = f"G{int(sat_id):02d}"

        try:
            year = int(parts[2])
            month = int(parts[3])
            day = int(parts[4])
            hour = int(parts[5])
            minute = int(parts[6])
            second_float = float(parts[7])
            second_int = int(second_float)
            microsecond = int(round((second_float - second_int) * 1_000_000))
            epoch = datetime(year, month, day, hour, minute, second_int, microsecond)
        except Exception:
            return None

        # parts[8] is number of data values (typically 1 or 2)
        try:
            # parts[9] is clock bias in seconds
            clock_bias_s = float(parts[9].replace("D", "E").replace("d", "e"))
        except (ValueError, IndexError):
            clock_bias_s = np.nan

        sigma_s = np.nan
        if len(parts) >= 11:
            try:
                sigma_s = float(parts[10].replace("D", "E").replace("d", "e"))
            except ValueError:
                sigma_s = np.nan

        # Sanity check on clock bias
        if abs(clock_bias_s) > self.MAX_REASONABLE_CLOCK_BIAS_S or np.isnan(clock_bias_s):
            clock_bias_s = np.nan
            clock_bias_ns = np.nan
        else:
            clock_bias_ns = clock_bias_s * 1e9  # seconds to nanoseconds

        return {
            "epoch": epoch,
            "sat_id": sat_id,
            "clock_bias_s": clock_bias_s,
            "clock_bias_ns": clock_bias_ns,
            "sigma_s": sigma_s,
            "is_synthetic": False,
            "data_source": data_source,
        }
