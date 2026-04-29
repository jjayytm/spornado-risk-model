# System Architecture

## Overview

The Spornado Disease Prediction Model is a modular, production-grade system designed to predict crop disease risk using the agricultural disease triangle framework.

```
┌─────────────────────────────────────────────────────────┐
│           SPORNADO DISEASE PREDICTION SYSTEM             │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    ┌────────┐      ┌──────────┐      ┌────────┐
    │ WEATHER│      │ SPORE    │      │VARIETY │
    │ DATA   │      │ PRESSURE │      │ DATA   │
    └───┬────┘      └────┬─────┘      └───┬────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                    [FEATURES]
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
   ┌─────────┐                     ┌──────────┐
   │ BASELINE│                     │ ENHANCED │
   │ MODELS  │                     │  MODELS  │
   │(2/3)    │                     │ (3/3)    │
   └────┬────┘                     └────┬─────┘
        │                              │
        └──────────────┬───────────────┘
                       │
                   [PREDICTIONS]
                       │
                ┌──────┴──────┐
                │             │
                ▼             ▼
            RISK SCORE    VARIETY
            (0-100%)   RECOMMENDATIONS
```

---

## Core Components

### 1. Data Pipeline

#### Input Data Sources
- **Weather Data**: Daily climate measurements (temperature, humidity, rainfall, leaf wetness)
- **Spore Pressure Data**: Weekly spore counts from Spornado devices at GPS coordinates
- **Variety Data**: Crop variety information with disease tolerance ratings
- **Disease Tolerance Databases**: Extracted from seed company catalogs (Syngenta, BASF, etc.)

#### Processing Steps
```
Raw Data
   ↓
[Validation] → Check format, missing values, outliers
   ↓
[Engineering] → Create rolling windows (7/14/21/30-day)
   ↓
[Integration] → Merge weather + spore + variety data
   ↓
[Normalization] → Standardize scales and ratings
   ↓
Ready for Modeling
```

### 2. Feature Engineering

#### Weather Features (7 base + rolling windows)
- Daily: Temp_Max, Temp_Min, Humidity, Rainfall, Leaf_Wetness_Duration
- Temporal: Month, Day_of_Year
- Rolling Windows: 7/14/21/30-day means, max, min

**Total Weather Features**: 23 engineered features

#### Spore Features
- Spore_Count_Log (log-transformed for normalization)
- Rolling Statistics: 7/14/21/30-day means and maximums

#### Variety Features (When Available)
- Disease Tolerance Ratings: 1-9 scale (1=resistant, 9=susceptible)
- Source: Seed company catalogs
- Standardization: Fuzzy matching for variety names

### 3. Model Training

#### Algorithm
- **Type**: Logistic Regression (binary classification)
- **Weights**: Balanced (handles class imbalance)
- **Validation**: Temporal split (80% train / 20% test)

#### Training Pipeline
```python
# Pseudocode
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# Train on 80% historical data
model.fit(X_train, y_train)

# Test on 20% recent data (prevents data leakage)
predictions = model.predict(X_test)

# Evaluate
auc = roc_auc_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
```

#### Models
- **Corn Model**: Trained on 2,198 samples, validated on 550
- **Soybean Model**: Trained on 500 samples, validated on 125
- **Potato Model**: Trained on 180 samples, validated on 45

#### Performance Metrics
| Metric | Purpose |
|--------|---------|
| AUC-ROC | Overall discrimination ability |
| F1 Score | Balance between precision & recall |
| Precision | True positive rate (minimize false alarms) |
| Recall | Disease detection rate (minimize missed cases) |

### 4. Disease Tolerance Database

#### Structure
```
variety_id | company  | variety_name | GLS | TS | NCLB | Ear_Rot
─────────────────────────────────────────────────────────────────
1          | Syngenta | NK7837       | 2   | 3  | 2    | 3
2          | BASF     | DK-445       | 2   | 3  | 2    | 3
3          | Syngenta | NK8000       | 3   | 4  | 3    | 4
```

Where:
- GLS = Gray Leaf Spot (1=resistant, 9=susceptible)
- TS = Tar Spot
- NCLB = Northern Corn Leaf Blight
- Ear_Rot = Ear Rot

#### Extraction Process
```
Seed Catalog PDF
       ↓
[PDF Parsing] → Extract tables from PDF
       ↓
[Fuzzy Match] → Match variety names to database
       ↓
[Standardize] → Convert to 1-9 scale
       ↓
[Validation] → Check data quality
       ↓
Disease Tolerance CSV
```

### 5. Integration & Retraining

#### Variety Data Integration
When farmer variety records become available:

```
Farmer Records
   ↓
[Merge with Disease Database]
   ↓
[Fuzzy Match Varieties]
   ↓
[Extract Tolerance Ratings]
   ↓
[Add to Spornado Dataset]
   ↓
Enhanced Training Data
```

#### Retraining Pipeline
```
Original Features (Spore + Weather)
   ↓
[Add Variety Ratings]
   ↓
New Feature Set (23 + variety features)
   ↓
[Retrain Models]
   ↓
Performance Metrics
   ↓
Deploy if Validated
```

---

## Design Patterns

### 1. Modular Architecture
Each script is self-contained and reusable:
- `extract_pdf_to_csv.py` - Standalone PDF extraction
- `integrate_variety_data.py` - Standalone data integration
- `build_crop_models.py` - Baseline model training
- `build_crop_models_with_variety.py` - Enhanced model training

### 2. Configuration-Driven Design
Instead of hardcoding, configuration files define:
- Seed company extraction templates
- Feature engineering parameters
- Model hyperparameters
- Output paths

### 3. Fallback Systems
Graceful degradation when data is unavailable:
- If PDF extraction fails → Use hardcoded fallback data
- If variety data missing → Use weather + spore only
- If partial data → Use available records, estimate rest

### 4. Logging & Monitoring
Comprehensive logging for debugging:
- File logs: `logs/execution.log`
- Console output: Real-time progress
- Error tracking: Detailed exception info
- Metrics logging: Model performance tracking

---

## Data Flow: End-to-End Example

### Scenario: Adding OCI's Farmer Variety Data

```
1. CLIENT PROVIDES DATA
   └─ farmer_variety_records.csv
      └─ location_id, crop_type, planted_variety, dates

2. INTEGRATION STEP
   └─ integrate_variety_data.py
      └─ Input: farmer_variety_records.csv
      └─ Process: Fuzzy match to disease database
      └─ Output: spornado_with_variety_features.csv

3. MODEL RETRAINING
   └─ build_crop_models_with_variety.py
      └─ Input: spornado_with_variety_features.csv
      └─ Process: Train models with variety features
      └─ Output: model_performance_results_with_variety.csv

4. DEPLOYMENT
   └─ Export trained models
   └─ Deploy to production
   └─ Farmers get variety recommendations

Expected Improvement:
   Before: "High disease risk" (99% confidence)
   After:  "High disease risk. Plant NK7837 (97% resistant)"
```

---

## API Interface

### Model Prediction API

```python
from src.build_crop_models import DiseaseRiskModel

# Initialize model
model = DiseaseRiskModel(crop='Corn')
model.load('models/model_performance_results.csv')

# Make prediction
risk_score = model.predict(
    spore_count=150,
    temp_max=28,
    temp_min=18,
    humidity=85,
    rainfall=12,
    month=6,
    day_of_year=152
)
# Returns: 0.87 (87% disease risk)
```

### Variety Integration API

```python
from src.integrate_variety_data import VarietyDataIntegrator

integrator = VarietyDataIntegrator()
integrated_df = integrator.integrate(
    variety_filepath='data/raw/farmer_varieties.csv',
    spornado_filepath='data/raw/spornado_data.csv'
)
# Returns: DataFrame with variety features merged
```

---

## Scalability Considerations

### Current Scale
- 3,755+ monitoring locations
- 50+ crop varieties across 3 companies
- 23 weather features + rolling windows
- 99%+ model accuracy

### Scaling Roadmap

#### Horizontal Scaling
- Add more monitoring locations
- Include additional seed companies
- Expand crop types (wheat, barley, canola)
- Regional model variants

#### Vertical Scaling
- Increase feature complexity
- Implement ensemble methods
- Add temporal/sequential models (LSTM)
- Include satellite imagery data

#### Performance Optimization
- Cache weather data locally
- Implement batch prediction
- Use model compression for edge deployment
- Database indexing for faster lookups

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, numpy |
| **PDF Processing** | pdfplumber |
| **Web Framework** | Flask (optional) |
| **Deployment** | Docker, Gunicorn |
| **Database** | PostgreSQL (optional) |
| **Monitoring** | Python logging |

---

## Security & Best Practices

### Data Security
- Encrypt sensitive farmer data
- Version control: Don't commit raw data
- Use environment variables for credentials
- Regular backups of model files

### Model Validation
- Temporal split prevents data leakage
- Cross-validation for robustness
- Holdout test set for final evaluation
- Regular retraining with new data

### Code Quality
- Modular, reusable components
- Comprehensive logging
- Error handling and validation
- Type hints and documentation

---

For implementation details, see [API_GUIDE.md](API_GUIDE.md).
