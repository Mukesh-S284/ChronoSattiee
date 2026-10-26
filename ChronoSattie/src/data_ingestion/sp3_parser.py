"""
Parser for IGS Precise Orbit Products in SP3 (SP3-a, SP3-c, SP3-d) format.
Extracts satellite coordinates (km), clock corrections (microseconds), timestamps, and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class SP3Metadata:
    """Metadata extracted from the SP3 file header."""
    version: str = "c"
    coordinate_system: str = "UNKNOWN"
    orbit_type: str = "PRE"
    agency: str = "UNKNOWN"
    start_time: Optional[datetime] = None
    number_of_epochs: int = 0
    epoch_interval_s: float = 900.0
    satellites: List[str] = field(default_factory=list)
    time_system: str = "GPS"


class SP3Parser:
    """
    Robust SP3 parser supporting standard IGS precise ephemeris formats.
    """

    BAD_VALUE_THRESHOLD = 999990.0

    def __init__(self) -> None:
        self.metadata = SP3Metadata()

    def parse_file(self, file_path_or_buffer: Union[str, Path, object]) -> Tuple[pd.DataFrame, SP3Metadata]:
        """
        Parse an SP3 file or string/stream into a Pandas DataFrame and metadata.

        Returns:
            Tuple of:
            - DataFrame with columns: ['epoch', 'sat_id', 'x_km', 'y_km', 'z_km', 'clock_bias_us', 'is_synthetic', 'data_source']
            - SP3Metadata dataclass
        """
        lines: List[str] = []

        if isinstance(file_path_or_buffer, (str, Path)):
            path = Path(file_path_or_buffer)
            if not path.exists():
                raise FileNotFoundError(f"SP3 file does not exist: {path}")
            data_source = path.name
            if path.suffix.lower() == ".gz" or str(path).lower().endswith(".sp3.gz"):
                import gzip
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            else:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
        else:
            # Assume file-like object or StringIO or UploadedFile
            data_source = getattr(file_path_or_buffer, "name", "uploaded_sp3")
            content = file_path_or_buffer.read()
            # Check for gzip magic number 0x1f 0x8b
            if isinstance(content, bytes) and content.startswith(b"\x1f\x8b"):
                import gzip
                content = gzip.decompress(content).decode("utf-8", errors="replace")
            elif isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            lines = content.splitlines()

        if not lines:
            raise ValueError(f"SP3 file '{data_source}' is empty or unreadable.")

        # Validate SP3 header marker '#'
        first_line = lines[0].strip()
        if not first_line.startswith("#"):
            raise ValueError(f"Invalid SP3 file format: First line does not start with '#' (found '{first_line[:10]}')")

        self.metadata = self._parse_header(lines)

        records: List[Dict[str, Union[datetime, str, float, bool]]] = []
        current_epoch: Optional[datetime] = None

        for line_num, line in enumerate(lines):
            line_str = line.strip()
            if not line_str:
                continue

            # Epoch header line: starts with '*'
            if line_str.startswith("*"):
                current_epoch = self._parse_epoch_line(line_str)
                continue

            # Position record: starts with 'P' (e.g. 'PG01', 'P G01', 'P  G01')
            if line.startswith("P") or (line.startswith(" ") and line.strip().startswith("P")):
                if current_epoch is None:
                    continue  # Position record before first epoch marker, skip

                record = self._parse_position_line(line, current_epoch, data_source)
                if record is not None:
                    records.append(record)

        if not records:
            raise ValueError(f"No valid satellite position records ('P') found in SP3 file '{data_source}'.")

        df = pd.DataFrame(records)

        # Ensure correct column types
        df["epoch"] = pd.to_datetime(df["epoch"])
        df["sat_id"] = df["sat_id"].astype(str)
        df["x_km"] = pd.to_numeric(df["x_km"], errors="coerce")
        df["y_km"] = pd.to_numeric(df["y_km"], errors="coerce")
        df["z_km"] = pd.to_numeric(df["z_km"], errors="coerce")
        df["clock_bias_us"] = pd.to_numeric(df["clock_bias_us"], errors="coerce")
        df["is_synthetic"] = df["is_synthetic"].astype(bool)
        df["data_source"] = df["data_source"].astype(str)

        # Sort chronologically by satellite and epoch
        df = df.sort_values(by=["sat_id", "epoch"]).reset_index(drop=True)

        return df, self.metadata

    def _parse_header(self, lines: List[str]) -> SP3Metadata:
        """Parse SP3 header lines to extract metadata."""
        meta = SP3Metadata()
        satellites: List[str] = []

        for line in lines[:30]:  # Headers are in the first ~25-30 lines
            line_strip = line.strip()
            if line.startswith("#"):
                # Line 1: version, start time, agency, etc.
                parts = line.split()
                if len(parts) >= 2:
                    meta.version = parts[0][1:]  # e.g. 'c' or 'd'
                if len(parts) >= 7:
                    try:
                        year = int(parts[1])
                        month = int(parts[2])
                        day = int(parts[3])
                        hour = int(parts[4])
                        minute = int(parts[5])
                        second = float(parts[6])
                        meta.start_time = datetime(year, month, day, hour, minute, int(second))
                    except (ValueError, IndexError):
                        pass
                if len(parts) >= 8:
                    try:
                        meta.number_of_epochs = int(parts[7])
                    except ValueError:
                        pass
                if len(parts) >= 10:
                    meta.coordinate_system = parts[9]
                if len(parts) >= 12:
                    meta.agency = parts[11]

            elif line.startswith("##"):
                # Line 2: interval
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        meta.epoch_interval_s = float(parts[3])
                    except (ValueError, IndexError):
                        pass

            elif line.startswith("+"):
                # Satellite ID list (either space-separated or fixed 3-character columns)
                raw_sats = line[9:60] if len(line) >= 60 else line[1:].strip()
                tokens = line[1:].strip().split()
                for token in tokens:
                    if len(token) > 3 and not token.isdigit():
                        # Handle concatenated 3-char IDs e.g. G01G02G03
                        for i in range(0, len(token), 3):
                            chunk = token[i : i + 3].strip()
                            if len(chunk) >= 2 and chunk not in satellites:
                                satellites.append(chunk)
                    elif len(token) >= 2 and (token[0].isalpha() or token.isdigit()):
                        sat_id = token if token[0].isalpha() else f"G{int(token):02d}"
                        if sat_id not in satellites and not sat_id.isdigit():
                            satellites.append(sat_id)

            elif line.startswith("*"):
                # End of header, first epoch begins
                break

        meta.satellites = satellites
        return meta

    def _parse_epoch_line(self, line: str) -> Optional[datetime]:
        """
        Parse epoch line: '*  YYYY MM DD HH MM SS.ssssssss'
        """
        try:
            parts = line.replace("*", "").strip().split()
            if len(parts) >= 6:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3])
                minute = int(parts[4])
                second = float(parts[5])
                microsecond = int(round((second - int(second)) * 1_000_000))
                return datetime(year, month, day, hour, minute, int(second), microsecond)
        except Exception:
            return None
        return None

    def _parse_position_line(
        self, line: str, epoch: datetime, data_source: str
    ) -> Optional[Dict[str, Union[datetime, str, float, bool]]]:
        """
        Parse position record 'P  SatID   X   Y   Z   Clock'
        """
        tokens = line.strip().split()
        if not tokens or len(tokens) < 5:
            return None

        # Determine satellite ID and numeric indices
        # Case A: tokens = ['PG01', '-12345.678', '...', '...', '...']
        # Case B: tokens = ['P', 'G01', '-12345.678', '...', '...', '...']
        if tokens[0] == "P":
            if len(tokens) < 6:
                return None
            sat_id = tokens[1]
            x_str, y_str, z_str, clk_str = tokens[2], tokens[3], tokens[4], tokens[5]
        elif tokens[0].startswith("P"):
            sat_id = tokens[0][1:].strip()
            if not sat_id and len(tokens) >= 6:
                sat_id = tokens[1]
                x_str, y_str, z_str, clk_str = tokens[2], tokens[3], tokens[4], tokens[5]
            else:
                x_str, y_str, z_str, clk_str = tokens[1], tokens[2], tokens[3], tokens[4]
        else:
            return None

        # Standardize sat_id (e.g. if '1' -> 'G01')
        if sat_id.isdigit():
            sat_id = f"G{int(sat_id):02d}"

        try:
            x_val = float(x_str)
            y_val = float(y_str)
            z_val = float(z_str)
            clk_val = float(clk_str)
        except ValueError:
            return None

        # Check for SP3 bad value sentinels (999999.999999)
        if abs(x_val) >= self.BAD_VALUE_THRESHOLD:
            x_val = np.nan
        if abs(y_val) >= self.BAD_VALUE_THRESHOLD:
            y_val = np.nan
        if abs(z_val) >= self.BAD_VALUE_THRESHOLD:
            z_val = np.nan
        if abs(clk_val) >= self.BAD_VALUE_THRESHOLD:
            clk_val = np.nan

        return {
            "epoch": epoch,
            "sat_id": sat_id,
            "x_km": x_val,
            "y_km": y_val,
            "z_km": z_val,
            "clock_bias_us": clk_val,
            "is_synthetic": False,
            "data_source": data_source,
        }
