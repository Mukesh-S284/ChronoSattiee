"""
GNSS Product Downloader for IGS and NASA CDDIS archives.
Handles authentication via environment variables (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
without hardcoded secrets, and provides fallback and manual file handling.
"""

from datetime import datetime
import gzip
import io
import os
from pathlib import Path
from typing import Optional, Tuple
import requests
from dotenv import load_dotenv

from src.config import CDDIS_BASE_URL, EARTHDATA_PASSWORD, EARTHDATA_USERNAME, RAW_DATA_DIR

load_dotenv()


class AuthenticationRequiredError(Exception):
    """Raised when NASA CDDIS or Earthdata credentials are missing or invalid."""
    pass


class GNSSDownloader:
    """
    Downloads GNSS precise orbit (SP3) and satellite clock (CLK) products
    from NASA CDDIS or IGS data distribution servers.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        self.username = username or os.getenv("EARTHDATA_USERNAME", "")
        self.password = password or os.getenv("EARTHDATA_PASSWORD", "")
        self.save_dir = save_dir or RAW_DATA_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calendar_to_gps_week(date: datetime) -> Tuple[int, int]:
        """
        Convert calendar datetime to GPS Week and Day of Week (0=Sunday ... 6=Saturday).
        GPS epoch starts 1980-01-06 00:00:00 UTC.
        """
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        delta_days = (date - gps_epoch).days
        gps_week = delta_days // 7
        day_of_week = delta_days % 7
        return gps_week, day_of_week

    def check_credentials(self) -> bool:
        """Verify whether Earthdata credentials are provided in the environment."""
        return bool(self.username and self.password)

    def download_product(
        self,
        date: datetime,
        product_type: str = "final_sp3",
        timeout_s: int = 30,
    ) -> Path:
        """
        Attempt download of a specified GNSS product for a target date.

        product_type:
            - 'final_sp3': IGS Final Orbit (igsWWWWd.sp3)
            - 'rapid_sp3': IGS Rapid Orbit (igrWWWWd.sp3)
            - 'final_clk': IGS Final 30s Clock (igsWWWWd.clk)
        """
        if not self.check_credentials():
            raise AuthenticationRequiredError(
                "NASA Earthdata credentials are required to download from CDDIS.\n"
                "Please set EARTHDATA_USERNAME and EARTHDATA_PASSWORD in your .env file or environment.\n"
                "Register for a free Earthdata account at: https://urs.earthdata.nasa.gov/\n"
                "Alternatively, you can manually upload SP3 / RINEX CLK files via the Streamlit dashboard."
            )

        gps_week, dow = self.calendar_to_gps_week(date)

        if product_type == "final_sp3":
            filename = f"igs{gps_week}{dow}.sp3.Z"
            remote_path = f"{CDDIS_BASE_URL}/{gps_week}/{filename}"
        elif product_type == "rapid_sp3":
            filename = f"igr{gps_week}{dow}.sp3.Z"
            remote_path = f"{CDDIS_BASE_URL}/{gps_week}/{filename}"
        elif product_type == "final_clk":
            filename = f"igs{gps_week}{dow}.clk.Z"
            remote_path = f"{CDDIS_BASE_URL}/{gps_week}/{filename}"
        else:
            raise ValueError(f"Unsupported product type: {product_type}")

        dest_file = self.save_dir / filename

        session = requests.Session()
        session.auth = (self.username, self.password)

        try:
            response = session.get(remote_path, timeout=timeout_s, stream=True)
            if response.status_code == 401 or response.status_code == 403:
                raise AuthenticationRequiredError(
                    f"Authentication failed (HTTP {response.status_code}). Verify your Earthdata login."
                )
            response.raise_for_status()

            with open(dest_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return dest_file

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error while downloading {remote_path}: {e}")
