"""
Script to download authentic IGS SP3 and CLK data from the open BKG IGS Data Center mirror.
"""

import gzip
import re
import sys
from pathlib import Path
import requests

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR

BKG_BASE_URL = "https://igs.bkg.bund.de/root_ftp/IGS/products"


def download_sample_real_igs(target_dir: Path = RAW_DATA_DIR) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to open BKG IGS repository at: {BKG_BASE_URL} ...")
    r = requests.get(BKG_BASE_URL + "/", timeout=15)
    r.raise_for_status()

    # Find week numbers (4 digits)
    matches = re.findall(r'href="(\d{4})/?', r.text)
    if not matches:
        print("Could not parse week listings from BKG mirror.")
        return

    recent_weeks = sorted(list(set(matches)), key=int)
    print(f"Found {len(recent_weeks)} GPS weeks on BKG. Most recent: {recent_weeks[-5:]}")

    # Try several weeks to find where standard SP3 files are stored
    for target_week in ["2250", "2200", "2150", "2100", "2000", "1950"]:
        week_url = f"{BKG_BASE_URL}/{target_week}/"
        print(f"Checking GPS Week {target_week}: {week_url}")
        try:
            r_week = requests.get(week_url, timeout=10)
            if r_week.status_code == 200:
                # Find all links
                links = [l.strip() for l in re.findall(r'href="([^"]+)"', r_week.text)]
                sp3_matches = [l for l in links if "sp3" in l.lower()]
                if sp3_matches:
                    print(f"Found {len(sp3_matches)} SP3 products in week {target_week}!")
                    print(f"Sample SP3 products: {sp3_matches[:5]}")
                    chosen_file = sp3_matches[0]
                    file_url = f"{week_url}{chosen_file}"
                    dest_path = target_dir / chosen_file
                    print(f"Downloading: {file_url} -> {dest_path}")
                    with requests.get(file_url, stream=True, timeout=30) as resp:
                        resp.raise_for_status()
                        with open(dest_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=8192):
                                f.write(chunk)
                    print(f"Downloaded real IGS product ({dest_path.stat().st_size:,} bytes) to {dest_path}")
                    return
        except Exception as e:
            print(f"Error checking week {target_week}: {e}")


if __name__ == "__main__":
    download_sample_real_igs()
