"""
Data ingestion module for SP3, RINEX CLK, synthetic demo generator, and downloader.
"""

from src.data_ingestion.sp3_parser import SP3Parser, SP3Metadata
from src.data_ingestion.rinex_clk_parser import RINEXCLKParser, RINEXCLKMetadata
from src.data_ingestion.synthetic_generator import SyntheticGNSSGenerator
from src.data_ingestion.downloader import GNSSDownloader, AuthenticationRequiredError

__all__ = [
    "SP3Parser",
    "SP3Metadata",
    "RINEXCLKParser",
    "RINEXCLKMetadata",
    "SyntheticGNSSGenerator",
    "GNSSDownloader",
    "AuthenticationRequiredError",
]
