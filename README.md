# Spornado Disease Risk Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

A production-grade machine learning system for crop disease risk prediction built on the **agricultural disease triangle**: weather conditions × spore pressure × crop variety susceptibility.

---

## Overview

The Spornado trap device records spore counts in the field.  This system fuses those trap readings with historical weather data to train per-crop **Logistic Regression** models (Corn, Soybean, Potato) that classify each observation as **LOW / MEDIUM / HIGH** disease risk.

Key design decisions that separate this from a notebook experiment:

| Decision | Why |
|---|---|
| **TimeSeriesSplit CV** | Preserves chronological order; prevents future data leaking into training folds |
| **Recall-constrained threshold** | In agriculture a missed outbreak (False Negative) costs a crop; the deployment threshold is set to catch ≥ 90 % of real outbreaks rather than maximise accuracy at the default 0.5 cut-off |
| **SimpleImputer in Pipeline** | Handles partial-weather rows at inference without crashing |
| **Threshold JSON beside each model** | Decouples threshold tuning from retraining; `predict.py` loads it at runtime |
| **Spore count excluded from features** | Spore count determines the test result — including it would be target leakage.  The model must predict risk from *environmental* signals alone so it is useful *before* a trap is read |

---

## Disease Triangle Framework

```
              DISEASE RISK
                   △
                  /│\
                 / │ \
           WEATHER │ CROP
            DATA   │ VARIETY
                  \ │ /
                   \│/
             SPORE PRESSURE
```

| Phase | Components | Status |
|---|---|---|
| **Baseline** | Weather + Spore | ✅ Trained and ready |
| **Enhanced** | + Crop variety tolerance | ⏳ Run when farmer variety records are available |

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/yourusername/spornado-risk-model.git
cd spornado-risk-model
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Build the merged dataset (requires raw data files and weather CSVs)
python -m src.build_weather_spore_data

# 3. Train per-crop models
python -m src.build_crop_models

# 4. Predict risk for a single observation
python -m src.predict \
    --crop Corn \
    --temperature_max_c 28.5 \
    --humidity_max_percent 83.0 \
    --precipitation_mm 2.1 \
    --month 8 --day_of_year 220 --week_of_year 32
```

### Python API

```python
from src.predict import Predictor

p = Predictor()
result = p.predict(
    crop="Corn",
    temperature_max_c=28.5,
    humidity_max_percent=83.0,
    precipitation_mm=2.1,
    month=8,
    day_of_year=220,
    week_of_year=32,
)
# {'crop': 'Corn', 'risk_probability': 0.73, 'risk_label': 'HIGH', ...}
```

---

## Pipeline: Step by Step

```
data/raw/All data.xlsx  ──┐
data/raw/More Data.csv  ──┤  src/build_weather_spore_data.py
data/weather/*.csv      ──┘         │
                                    ▼
                    data/raw/spornado_weather_spore_data.csv
                                    │
                      src/build_crop_models.py
                      (features → TimeSeriesSplit CV
                       → Pipeline fit → threshold optimisation)
                                    │
                    ┌───────────────┴───────────────┐
             models/corn_model.pkl          models/corn_thresholds.json
             models/soybean_model.pkl       models/soybean_thresholds.json
             models/potato_model.pkl        models/potato_thresholds.json
                                    │
                            src/predict.py
                       (load model + thresholds → LOW/MEDIUM/HIGH)

── Optional Phase 2 ──────────────────────────────────────────────────
data/external/*.pdf  →  src/extract_pdf_to_csv.py
                                    │
                    data/processed/*_disease_tolerance.csv
                                    │
farmer variety records  →  src/integrate_variety_data.py
                                    │
                    data/processed/spornado_with_variety_features.csv
                                    │
                   src/build_crop_models_with_variety.py
                                    │
                   models/*_model_variety.pkl  (overrides baseline)
```

---

## Project Structure

```
spornado-risk-model/
│
├── src/                                   # All production code
│   ├── __init__.py
│   ├── config_loader.py                   # Load + resolve config.yaml
│   ├── features.py                        # Feature engineering pipeline
│   ├── build_weather_spore_data.py        # Step 1 — merge raw data with weather
│   ├── build_crop_models.py               # Step 2 — train per-crop models
│   ├── predict.py                         # Step 3 — load model & predict risk
│   ├── extract_pdf_to_csv.py              # Phase 2 — extract tolerance from PDFs
│   ├── integrate_variety_data.py          # Phase 2 — merge variety features
│   └── build_crop_models_with_variety.py  # Phase 2 — retrain with variety data
│
├── tests/
│   ├── test_features.py                   # Unit tests for feature engineering
│   ├── test_models.py                     # Unit tests for model training & thresholds
│   └── test_pipeline.py                   # Integration tests
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb # Data understanding and visualisations
│   └── 02_disease_prediction_model.ipynb  # Model demo using src/ modules
│
├── data/
│   ├── raw/                               # Input data (git-ignored except .gitkeep)
│   │   ├── spornado_weather_spore_data.csv  ← output of build_weather_spore_data.py
│   │   ├── All data.xlsx
│   │   └── More Data.csv
│   ├── processed/                         # Cleaned / variety-enriched data
│   │   ├── corn_disease_tolerance.csv
│   │   ├── soybean_disease_tolerance.csv
│   │   └── potato_disease_tolerance.csv
│   └── external/                          # Seed-company PDF catalogs
│       ├── Syngenta_NK-EAST_SeedGuide_2026.pdf
│       └── BASF_Xitavo_Catalog_2025.pdf
│
├── models/                                # Trained models (git-ignored; .gitkeep only)
│
├── outputs/                               # Plots and dashboard images
│
├── docs/
│   ├── ARCHITECTURE.md                    # System design overview
│   ├── DATA_GUIDE.md                      # Data formats and requirements
│   └── DEPLOYMENT_GUIDE.md               # Production deployment steps
│
├── config/
│   └── config.yaml                        # All tunable parameters
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Configuration

All parameters live in `config/config.yaml`.  The most important ones:

```yaml
model:
  min_recall:         0.90   # "catch at least 9 in 10 real outbreaks"
  medium_band_factor: 0.70   # MEDIUM zone = [optimal × 0.70, optimal)
  cv_folds:           5
  n_jobs:             1      # set to -1 on a multi-core server
```

Change `min_recall` to tighten or relax the false-negative constraint.  A higher value (e.g. `0.95`) catches more outbreaks at the cost of more false alarms.

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## Phase 2: Variety-Aware Models

When farmer records include the seed variety planted:

```bash
# Extract tolerance ratings from seed-company PDFs
python -m src.extract_pdf_to_csv \
    --pdf data/external/Syngenta_NK-EAST_SeedGuide_2026.pdf \
    --company Syngenta --crop Corn

# Merge variety features into the dataset
python -m src.integrate_variety_data \
    --input data/raw/farmer_variety_records.csv \
    --spornado data/raw/spornado_weather_spore_data.csv

# Retrain with all three disease-triangle components
python -m src.build_crop_models_with_variety
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `No trained model found for crop` | Models not yet trained | Run `python -m src.build_crop_models` |
| `Insufficient data for crop` | Fewer than 100 samples | Check data quality; lower `min_samples_per_crop` in config |
| `Cannot achieve X% recall` | Insufficient positive samples | Lower `min_recall` or collect more positive-class data |
| `No weather files loaded` | `data/weather/` is empty | Provide per-location `weather_<lat>_<lon>.csv` files |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
