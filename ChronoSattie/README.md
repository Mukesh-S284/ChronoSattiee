# 🛰️ ChronoSattie: AI/ML Satellite Clock and Ephemeris Forecasting Prototype

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Code Style: Modular](https://img.shields.io/badge/Architecture-Modular%20Scientific-green.svg)]()

> **College Mini-Project:** AI/ML Models to Predict Time-Varying Patterns of Satellite Clock and Ephemeris Parameters  
> **Category:** Software / Smart Automation  
> **Developer:** Senior AI/ML & GNSS Data Engineering Team  

---

## 1. Executive Summary & Objective

Global Navigation Satellite Systems (GNSS) such as GPS, Galileo, GLONASS, and BeiDou broadcast orbital ephemeris and satellite clock parameters to user receivers worldwide. However, broadcast ephemeris degrades rapidly over time due to high-order gravitational anomalies, solar radiation pressure, and atomic clock physical frequency drift.

**ChronoSattie** is a modular, scientific time-series forecasting software prototype designed to ingest historical GNSS satellite data and predict future satellite clock corrections and Earth-Centered Earth-Fixed (ECEF) orbital coordinates across forecast horizons of **1, 6, 12, and 24 hours**.

### Key System Capabilities:
- **Real GNSS Ingestion:** Native parsers for standard **IGS SP3** (SP3-a, c, d) precise orbit files and **RINEX 3.0x / 2.xx CLK** clock files.
- **Physical Gap Detection & Outlier Filtering:** Handles observation gaps without artificial derivative distortion across missing epochs.
- **Strict Leakage-Free Preprocessing:** Chronological time-series ordering. Scaling transformations and lag generation are fitted strictly on past training splits without future-data leakage.
- **Multiple Forecasting Architectures:**
  1. *Persistence Baseline:* Mandatory operational reference benchmark (last observed value).
  2. *Linear Drift Baseline:* First-order polynomial trend extrapolation.
  3. *Random Forest Regressor:* Non-linear multi-estimator ensemble.
  4. *HistGradientBoostingRegressor:* High-efficiency gradient-boosted decision trees.
- **Physical Error Representation:** 3D Euclidean orbital position error in meters (`norm(pred - actual) * 1000.0`), clock bias errors in nanoseconds (`ns`) and microseconds (`μs`).
- **Interactive Multi-Page Streamlit UI:** Rich 3D orbit visualization, clock time-series, walk-forward validation diagnostics, and CSV/JSON manifest exports.
- **Reproducible Synthetic Simulator:** High-fidelity Keplerian orbit generator and atomic clock polynomial + random-walk drift model for offline demonstration.

---

## 2. System Architecture

```
ChronoSattie/
├── app.py                         # Main Streamlit Application (Page 1: Overview)
├── pages/
│   ├── 1_Data_Explorer.py        # Page 2: Ingestion, Upload, Synthetic Generator & 3D Plots
│   ├── 2_Model_Training.py       # Page 3: Chronological Split, Training, & Model Persistence
│   ├── 3_Satellite_Forecasting.py# Page 4: Real-Time Multi-Horizon Forecasting Engine
│   ├── 4_Evaluation.py           # Page 5: Predicted vs Actual, Residuals, & Horizon Scaling
│   └── 5_Reports.py              # Page 6: CSV Exports & Provenance Manifests
├── src/
│   ├── config.py                 # GNSS physical constants & global paths
│   ├── ui_state.py               # Shared Streamlit session state manager
│   ├── data_ingestion/
│   │   ├── sp3_parser.py         # IGS SP3 (X, Y, Z, clock) parser
│   │   ├── rinex_clk_parser.py   # RINEX CLK clock observation parser
│   │   ├── synthetic_generator.py# Physically grounded Keplerian/Clock simulator
│   │   └── downloader.py         # NASA CDDIS & IGS downloader with credential handling
│   ├── preprocessing/
│   │   ├── cleaning.py           # Chronological sort, duplicate & gap detection
│   │   └── features.py           # Strictly causal backward lags & orbital periodic terms
│   ├── models/
│   │   ├── baselines.py          # Persistence and Linear drift benchmarks
│   │   ├── forecasting.py        # Unified multi-horizon forecasting engine
│   │   └── registry.py           # Model persistence with joblib and metadata tracking
│   ├── evaluation/
│   │   ├── metrics.py            # MAE, RMSE, MBE, 3D position error (m), clock error (ns)
│   │   └── validation.py         # Chronological train/test splits & multi-horizon runner
│   ├── visualization/
│   │   └── plots.py              # Interactive Plotly 3D orbits and residual diagnostics
│   └── reporting/
│       └── exports.py            # Structured CSV/JSON manifest generation
├── tests/
│   ├── fixtures/                 # Documented SP3 and CLK test fixtures
│   ├── test_sp3_parser.py
│   ├── test_rinex_clk_parser.py
│   ├── test_preprocessing.py
│   ├── test_leakage_and_splitting.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_synthetic_and_exports.py
├── data/                         # Persistent data directory (raw, processed, demo)
├── models/saved/                 # Joblib model registry
├── reports/                      # Exported CSVs and JSON manifests
├── Dockerfile                    # Container definition
├── compose.yaml                  # Docker Compose configuration
├── requirements.txt              # Production dependency specifications
├── pyproject.toml                # Standard Python package configuration
├── .env.example                  # Environment credentials template
├── DEVELOPMENT_STATUS.md         # Comprehensive verification matrix
└── README.md
```

---

## 3. Installation & Local Execution

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git.

### Windows (PowerShell) Setup:
```powershell
# 1. Clone repository or navigate to project directory
cd c:\Users\asus\Desktop\ChronoSattie

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Launch the Streamlit Dashboard
streamlit run app.py
```

### Linux / macOS Setup:
```bash
# 1. Navigate to directory
cd ChronoSattie

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch Streamlit
streamlit run app.py
```

The web dashboard will automatically open in your browser at `http://localhost:8501`.

---

## 4. Running via Docker

If Docker is installed on your deployment host:

```bash
# Build and run container with Compose
docker compose up --build -d

# View application logs
docker compose logs -f

# Stop container
docker compose down
```

The container exposes port `8501` and mounts the `./data`, `./models/saved`, and `./reports` directories to maintain data persistence across runs.

*(Note: On environments where Docker CLI is not installed, local execution via the Python virtual environment is 100% functional).*

---

## 5. Automated Pytest Verification

Execute the test suite verifying all 20 required points:

```powershell
.\.venv\Scripts\pytest.exe -v
```

### Test Coverage Highlights:
- **Test 1:** SP3 precise orbit parser verification using fixture data.
- **Test 2:** RINEX CLK clock parser verification.
- **Test 3:** Invalid file, empty stream, and malformed header exception handling.
- **Test 4 & 5:** Chronological ordering and duplicate record deduplication.
- **Test 6:** Observation gap detection and `999999.999999` bad-value sentinel masking.
- **Test 7 & 16:** Unit conversions (seconds $\leftrightarrow$ nanoseconds $\leftrightarrow$ microseconds).
- **Test 8:** Satellite ID selection and constellation filtering.
- **Test 9 & 10:** Strictly backward-looking lag feature generation with zero future leakage.
- **Test 11:** Chronological non-shuffled train/test partitioning.
- **Test 12:** Persistence and linear drift baseline forecasting.
- **Test 13 & 14:** Multi-horizon training (1h, 6h, 12h, 24h) across RF, HistGradientBoosting, and Linear Regression.
- **Test 15:** 3D Euclidean orbital position error in meters: $\text{norm}(XYZ_{pred} - XYZ_{ref}) \times 1000.0$.
- **Test 17:** Mathematical correctness of MAE, RMSE, and Mean Bias Error (MBE).
- **Test 18:** Reproducible synthetic generator and explicit `is_synthetic=True` labeling.
- **Test 19:** CSV predictions, metrics, and JSON manifest export generation.
- **Test 20:** Streamlit application entry points and page module import integrity.

---

## 6. How to Obtain and Ingest Real GNSS Data

### Option A: Manual Upload via Dashboard (Zero Credentials Required)
1. Visit the public **IGS Product Archive** at [https://files.igs.org/pub/product/](https://files.igs.org/pub/product/) or [IGS CDDIS Products](https://cddis.nasa.gov/).
2. Download an uncompressed or `.Z` daily product:
   - Precise Orbit: `igsWWWWd.sp3` (e.g. `igs22000.sp3`)
   - Precise Clock: `igsWWWWd.clk` (e.g. `igs22000.clk`)
3. Open **ChronoSattie** $\rightarrow$ Navigate to **1_Data_Explorer** $\rightarrow$ Click **Upload Real GNSS Files**.
4. Select your `.sp3` or `.clk` file and click **Parse**. The system immediately validates header metadata, checks coordinate bounds, and activates **Real GNSS Mode**.

### Option B: Automated CDDIS Download via API
1. Register for a free NASA Earthdata account at [https://urs.earthdata.nasa.gov/](https://urs.earthdata.nasa.gov/).
2. Copy `.env.example` to `.env`:
   ```env
   EARTHDATA_USERNAME=your_username
   EARTHDATA_PASSWORD=your_password
   ```
3. Use `GNSSDownloader` from `src.data_ingestion.downloader` to automatically fetch products.

---

## 7. Demonstration Workflow for College Project Review

When presenting this project to evaluators:

1. **Dashboard Overview (`app.py`):**
   - Highlight the real-time health metrics, active satellites, time span, and the prominent **Data Provenance Status Banner**.
2. **Data Explorer (`1_Data_Explorer.py`):**
   - Demonstrate the **Keplerian & Clock Simulator**: Generate a 72-hour dataset with 2% simulated gaps across satellites `G01`, `G02`, `G14`, and `E04`.
   - Inspect the **Interactive 3D Orbit Visualization**: Rotate the 3D globe to show the satellite orbiting Earth in the ECEF frame.
   - Run the **Data Cleaning Pipeline** to show automatic duplicate dropping, physical radius bounding [6000 km, 45000 km], and gap detection.
3. **Model Training (`2_Model_Training.py`):**
   - Select satellite `G01`, target `clock_bias_us`, and horizon `1 Hour`.
   - Train a **Random Forest Regressor** with an 80/20 chronological split.
   - Review the comparison table showing model MAE/RMSE vs the **Persistence Baseline**.
   - Click **Save Model to Disk** to demonstrate the `joblib` model registry.
4. **Satellite Forecasting (`3_Satellite_Forecasting.py`):**
   - Select a historical forecast origin epoch or use the latest observation.
   - Generate a forward projection for **6 Hours** or **24 Hours**.
   - Point out the distinct markers for History, Origin, and Future Projection. Explain why ML models must be distinguished from numerical orbit integrators.
5. **Prediction Evaluation (`4_Evaluation.py`):**
   - Click **Run Full Multi-Horizon Evaluation**.
   - Show how the forecast error scales across **1h, 6h, 12h, and 24h** horizons on the bar chart.
   - Inspect the **Residual Distribution Histogram** to verify zero-centered errors (unbiased estimator).
6. **Reports & Exports (`5_Reports.py`):**
   - Download the generated **Forecasts CSV**, **Evaluation Metrics CSV**, and **Data Manifest JSON** to show reproducible scientific export capability.

---

## 8. Implemented Models & Scientific Limitations

| Model | Type | Strengths | Operational Limitations |
| :--- | :--- | :--- | :--- |
| **Persistence** | Naive Baseline | Zero training required; strong short-term (1h) clock baseline. | Fails completely over 6h-24h for periodic orbital motion. |
| **Linear Drift** | First-Order Baseline | Tracks steady atomic clock frequency drift well over 1-6h. | Cannot capture orbital periodicity or non-linear relativistic clock terms. |
| **Random Forest** | Non-Linear Ensemble | Captures complex non-linear combinations of orbital harmonic phases and lags. | Does not extrapolate outside the bounding box of historical training values. |
| **HistGradientBoosting** | Gradient Boosted Trees | Extremely fast training; handles tabular time-series features and missing values robustly. | Purely statistical; does not enforce physical conservation of orbital energy. |

### Operational Note:
An ML prediction of SP3 coordinates is a statistical regressor over historical trajectory points. In operational geodesy, precision orbit determination (POD) combines numerical Cowell/Encke numerical integration with spherical harmonic gravity models (EGM2008), solar radiation pressure (ECOM2), third-body perturbations (Sun/Moon), and relativistic corrections. Machine learning is best utilized as a **residual correction model** on top of physical propagators.

---

## 9. Current Status & Acceptance Criteria Checklist

- [x] Application starts locally via Streamlit (`streamlit run app.py`).
- [x] Interactive web dashboard loads without errors.
- [x] Synthetic demonstration mode works and is clearly labeled with persistent warning banners.
- [x] SP3 ingestion is implemented and tested with standard IGS format sentinels.
- [x] RINEX CLK ingestion is implemented and unit conversions tested.
- [x] Real-data download client and manual file-upload workflows available.
- [x] Baseline models (Persistence, Linear) and ML models (Random Forest, HistGradientBoosting) implemented.
- [x] Forecast horizons of 1, 6, 12, and 24 hours supported.
- [x] Insufficient data produces a clear warning rather than fabricated numbers.
- [x] Time-series evaluation strictly prevents future-data leakage.
- [x] Predicted-versus-actual analysis works when reference observations exist.
- [x] 3D position error in meters and clock error in nanoseconds explicitly calculated.
- [x] CSV and JSON manifest exports functional.
- [x] Automated test suite (22 tests) passes cleanly.
- [x] Dockerfile and compose.yaml provided with healthcheck and volume persistence.
- [x] `DEVELOPMENT_STATUS.md` continuously maintained.
