"""
Configuration settings, path constants, and physical GNSS reference constants.
"""

from pathlib import Path
import os
from typing import Dict, List, Tuple

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo"
MODELS_DIR = BASE_DIR / "models" / "saved"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DEMO_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# GNSS Physical Constants
SPEED_OF_LIGHT_M_S = 299792458.0  # Speed of light in vacuum (m/s)
EARTH_GM = 3.986004418e14         # Earth's gravitational parameter (m^3/s^2)
EARTH_ROTATION_RATE_RAD_S = 7.2921151467e-5  # Earth rotation rate (rad/s)
GPS_ORBIT_RADIUS_M = 26560000.0   # Nominal semi-major axis for GPS MEO (m)
GPS_ORBITAL_PERIOD_S = 43082.0    # ~11 hours 58 minutes (half a sidereal day)

# Forecast Horizons (in hours and equivalent 15-minute SP3 steps)
DEFAULT_HORIZONS_HOURS = [1, 6, 12, 24]
SP3_NOMINAL_INTERVAL_S = 900  # 15 minutes = 900 seconds
CLK_NOMINAL_INTERVAL_S = 30   # 30 seconds for standard RINEX high-rate clock

# Mapping from forecast horizon in hours to number of SP3 epochs (assuming 15-min sampling)
HORIZON_STEPS_SP3: Dict[int, int] = {
    1: 4,     # 1 hour = 4 * 15 min
    6: 24,    # 6 hours = 24 * 15 min
    12: 48,   # 12 hours = 48 * 15 min
    24: 96,   # 24 hours = 96 * 15 min
}

# Supported Satellite Constellation Prefixes
CONSTELLATION_NAMES: Dict[str, str] = {
    "G": "GPS (USA)",
    "R": "GLONASS (Russia)",
    "E": "Galileo (EU)",
    "C": "BeiDou (China)",
    "J": "QZSS (Japan)",
    "I": "NavIC (India)",
}

# CDDIS & NASA Earthdata Authentication
EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME", "")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD", "")
CDDIS_BASE_URL = "https://cddis.nasa.gov/archive/gnss/products"
IGS_FTP_FALLBACK_URL = "https://files.igs.org/pub/product"

# Special / Invalid Data Sentinels in SP3
SP3_BAD_VALUE_SENTINEL = 999999.999999
SP3_CLOCK_BAD_VALUE = 999999.999999
