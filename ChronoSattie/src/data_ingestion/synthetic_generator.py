"""
Synthetic GNSS satellite data generator for testing and demonstration.
Generates physically realistic Keplerian/ECEF satellite trajectories and atomic clock drift models.
Explicitly labels all generated records with is_synthetic=True.
"""

from datetime import datetime, timedelta
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    DEMO_DATA_DIR,
    EARTH_GM,
    EARTH_ROTATION_RATE_RAD_S,
    GPS_ORBIT_RADIUS_M,
    GPS_ORBITAL_PERIOD_S,
    SPEED_OF_LIGHT_M_S,
)


class SyntheticGNSSGenerator:
    """
    Generates physically plausible synthetic GNSS orbit and satellite clock time series.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def generate_orbit_and_clock(
        self,
        sat_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        duration_hours: int = 72,
        sampling_interval_s: int = 900,  # 15 minutes
        missing_rate: float = 0.02,       # 2% simulated missing epochs
    ) -> pd.DataFrame:
        """
        Generate time-series observations for specified satellites.

        Returns:
            DataFrame with columns:
            ['epoch', 'sat_id', 'x_km', 'y_km', 'z_km', 'clock_bias_us', 'clock_bias_ns', 'is_synthetic', 'data_source']
        """
        if sat_ids is None:
            sat_ids = ["G01", "G02", "G14", "E04"]

        if start_time is None:
            # Fixed reference epoch for reproducibility
            start_time = datetime(2026, 1, 1, 0, 0, 0)

        total_epochs = int((duration_hours * 3600) / sampling_interval_s) + 1
        records: List[Dict] = []

        # Nominal Keplerian elements for MEO satellites
        # GPS semi-major axis in km: ~26,560 km
        a_km = GPS_ORBIT_RADIUS_M / 1000.0
        n_rad_s = 2.0 * math.pi / GPS_ORBITAL_PERIOD_S
        inclination_rad = math.radians(55.0)

        for sat_idx, sat_id in enumerate(sat_ids):
            # Unique orbital plane parameters per satellite
            raan_0 = math.radians((sat_idx * 60.0) % 360.0)  # Ascending node
            arg_perigee = math.radians(30.0 + sat_idx * 45.0)
            eccentricity = 0.005 + 0.002 * (sat_idx % 3)

            # Clock polynomial coefficients (unique per satellite)
            # a0 (offset in microseconds), a1 (drift in us/s), a2 (drift rate in us/s^2)
            a0_us = 100.0 + 35.0 * sat_idx + self.rng.normal(0, 5.0)
            a1_us_s = (1.5 + 0.3 * sat_idx) * 1e-6  # ~1.5 ps/s drift
            a2_us_s2 = 1.0e-12

            # Random walk phase noise accumulator
            rw_phase_us = 0.0

            for step in range(total_epochs):
                current_epoch = start_time + timedelta(seconds=step * sampling_interval_s)
                t_seconds = step * sampling_interval_s

                # Simulated missing epoch check
                if self.rng.random() < missing_rate:
                    continue  # Epoch gap simulated

                # 1. Keplerian Orbit Computation
                # Mean anomaly
                M = (n_rad_s * t_seconds) % (2.0 * math.pi)
                # Solve Kepler's equation M = E - e*sin(E) by first-order approximation
                E = M + eccentricity * math.sin(M) + 0.5 * (eccentricity**2) * math.sin(2.0 * M)
                # True anomaly
                sin_nu = (math.sqrt(1.0 - eccentricity**2) * math.sin(E)) / (1.0 - eccentricity * math.cos(E))
                cos_nu = (math.cos(E) - eccentricity) / (1.0 - eccentricity * math.cos(E))
                nu = math.atan2(sin_nu, cos_nu)

                # Distance from Earth center in km
                r_km = a_km * (1.0 - eccentricity * math.cos(E))

                # Position in orbital plane
                u = arg_perigee + nu  # Argument of latitude
                x_orb = r_km * math.cos(u)
                y_orb = r_km * math.sin(u)

                # Nodal precession and Greenwich Sidereal Time rotation into ECEF
                theta_earth = EARTH_ROTATION_RATE_RAD_S * t_seconds
                omega_current = raan_0 - theta_earth

                # Transform to ECEF coordinates
                x_ecef = x_orb * math.cos(omega_current) - y_orb * math.cos(inclination_rad) * math.sin(omega_current)
                y_ecef = x_orb * math.sin(omega_current) + y_orb * math.cos(inclination_rad) * math.cos(omega_current)
                z_ecef = y_orb * math.sin(inclination_rad)

                # Add tiny measurement noise (mm-scale in km)
                x_ecef += self.rng.normal(0, 1e-5)
                y_ecef += self.rng.normal(0, 1e-5)
                z_ecef += self.rng.normal(0, 1e-5)

                # 2. Satellite Atomic Clock Model
                # Relativistic periodic variation: delta_t_r = (2 * sqrt(GM*a) / c^2) * e * sin(E)
                # In microseconds:
                rel_amplitude_s = (2.0 * math.sqrt(EARTH_GM * (a_km * 1000.0)) / (SPEED_OF_LIGHT_M_S**2)) * eccentricity
                rel_amplitude_us = rel_amplitude_s * 1e6
                clock_rel_us = rel_amplitude_us * math.sin(E)

                # Thermal/orbital periodic variations
                thermal_us = 0.002 * math.sin(2.0 * M + 0.5)

                # Random walk drift step
                rw_phase_us += self.rng.normal(0, 0.0002)

                # White measurement noise
                white_noise_us = self.rng.normal(0, 0.0001)

                clock_bias_us = (
                    a0_us
                    + a1_us_s * t_seconds
                    + a2_us_s2 * (t_seconds**2)
                    + clock_rel_us
                    + thermal_us
                    + rw_phase_us
                    + white_noise_us
                )

                clock_bias_ns = clock_bias_us * 1000.0

                records.append(
                    {
                        "epoch": current_epoch,
                        "sat_id": sat_id,
                        "x_km": x_ecef,
                        "y_km": y_ecef,
                        "z_km": z_ecef,
                        "clock_bias_us": clock_bias_us,
                        "clock_bias_ns": clock_bias_ns,
                        "clock_bias_s": clock_bias_us * 1e-6,
                        "is_synthetic": True,
                        "data_source": "SYNTHETIC_SIMULATOR",
                    }
                )

        df = pd.DataFrame(records)
        df["epoch"] = pd.to_datetime(df["epoch"])
        df = df.sort_values(by=["sat_id", "epoch"]).reset_index(drop=True)
        return df

    def save_demo_sp3(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export synthetic DataFrame to valid SP3 format for parser and end-to-end testing.
        """
        if output_path is None:
            output_path = DEMO_DATA_DIR / "synthetic_demo.sp3"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sat_ids = sorted(df["sat_id"].unique().tolist())
        epochs = sorted(df["epoch"].unique().tolist())

        first_epoch = pd.to_datetime(epochs[0])

        with open(output_path, "w", encoding="utf-8") as f:
            # Line 1: Header
            f.write(
                f"#c {first_epoch.year:4d} {first_epoch.month:2d} {first_epoch.day:2d} "
                f"{first_epoch.hour:2d} {first_epoch.minute:2d} {first_epoch.second:11.8f} "
                f"{len(epochs):7d}   SYNTHETIC IGS14   PRE   CHRONOSATTIE\n"
            )
            # Line 2: Interval
            f.write("##  2200 432000.00000000    900.00000000  59332 0.0000000000000\n")
            # Satellite list line
            sat_str = " ".join(sat_ids)
            f.write(f"+   {len(sat_ids):2d}   {sat_str}\n")
            f.write("/* SYNTHETIC DEMONSTRATION DATASET - CHRONOSATTIE */\n")

            # Data records grouped by epoch
            for ep in epochs:
                ep_dt = pd.to_datetime(ep)
                f.write(
                    f"*  {ep_dt.year:4d} {ep_dt.month:2d} {ep_dt.day:2d} "
                    f"{ep_dt.hour:2d} {ep_dt.minute:2d} {ep_dt.second:11.8f}\n"
                )
                ep_sub = df[df["epoch"] == ep]
                for _, row in ep_sub.iterrows():
                    f.write(
                        f"P{row['sat_id']:>3s} {row['x_km']:14.6f} {row['y_km']:14.6f} "
                        f"{row['z_km']:14.6f} {row['clock_bias_us']:14.6f}\n"
                    )

            f.write("EOF\n")

        return output_path

    def save_demo_clk(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export synthetic DataFrame to valid RINEX CLK format.
        """
        if output_path is None:
            output_path = DEMO_DATA_DIR / "synthetic_demo.clk"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("     3.00           C                   CLK                 RINEX VERSION / TYPE\n")
            f.write("CHRONOSATTIE        SYNTHETIC DATA GEN  20261026 230000 UTC PGM / RUN BY / DATE\n")
            f.write("     1    AS                                                # / TYPES OF DATA   \n")
            f.write("   GPS                                                      TIME SYSTEM ID      \n")
            f.write("                                                            END OF HEADER       \n")

            for _, row in df.iterrows():
                ep_dt = pd.to_datetime(row["epoch"])
                clk_s = row["clock_bias_us"] * 1e-6
                f.write(
                    f"AS {row['sat_id']:>3s} {ep_dt.year:4d} {ep_dt.month:02d} {ep_dt.day:02d} "
                    f"{ep_dt.hour:02d} {ep_dt.minute:02d} {ep_dt.second:11.8f}  1  {clk_s:19.12E}  0.100000E-09\n"
                )

        return output_path
