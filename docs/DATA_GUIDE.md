# Data Guide

## Input Data Formats

### 1. Spornado Weather + Spore Data

**File**: `data/raw/spornado_weather_spore_data.csv`

**Required Columns**:
```
location_id, crop, disease_present, Spore_Count_Log, Month, Day_of_Year,
Temp_Max, Temp_Min, Humidity, Rainfall, Leaf_Wetness_Duration,
Rolling_7day_Spore_Mean, Rolling_14day_Spore_Max, Rolling_21day_Spore_Mean,
Rolling_30day_Spore_Max
```

**Data Types**:
```python
{
    'location_id': 'string',           # GPS coordinate or device ID
    'crop': 'string',                  # Corn, Soybean, Potato
    'disease_present': 'int',          # 0=No disease, 1=Disease detected
    'Spore_Count_Log': 'float',        # Log-transformed spore count
    'Month': 'int',                    # 1-12
    'Day_of_Year': 'int',              # 1-365
    'Temp_Max': 'float',               # Celsius
    'Temp_Min': 'float',               # Celsius
    'Humidity': 'float',               # 0-100 percent
    'Rainfall': 'float',               # mm
    'Leaf_Wetness_Duration': 'float',  # hours
    'Rolling_7day_Spore_Mean': 'float',
    'Rolling_14day_Spore_Max': 'float',
    'Rolling_21day_Spore_Mean': 'float',
    'Rolling_30day_Spore_Max': 'float'
}
```

**Example**:
```csv
location_id,crop,disease_present,Spore_Count_Log,Month,Day_of_Year,Temp_Max,Temp_Min,Humidity,Rainfall,Leaf_Wetness_Duration,Rolling_7day_Spore_Mean,Rolling_14day_Spore_Max,Rolling_21day_Spore_Mean,Rolling_30day_Spore_Max
LOC_001,Corn,1,3.25,6,152,28.5,18.2,85.0,12.5,8.5,2.98,3.41,3.12,3.45
LOC_001,Corn,0,2.15,6,153,27.8,17.9,82.0,0.0,6.2,2.94,3.38,3.08,3.42
GPS_40.5_-88.2,Soybean,0,1.45,7,183,26.2,16.5,78.0,5.0,4.5,1.52,1.98,1.65,1.88
```

**Data Validation**:
- ✓ No missing values in required columns
- ✓ Spore_Count_Log >= 0
- ✓ Month: 1-12
- ✓ Day_of_Year: 1-365
- ✓ Temperature: -40 to 50°C (reasonable range)
- ✓ Humidity: 0-100%
- ✓ Disease_present: 0 or 1

---

### 2. Farmer Variety Records

**File**: `data/raw/farmer_variety_records.csv` (Optional - provided by client)

**Required Columns**:
```
location_id, crop_type, planted_variety
```

**Optional Columns**:
```
planting_date, harvest_date, confidence
```

**Data Types**:
```python
{
    'location_id': 'string',          # Must match Spornado location_id
    'crop_type': 'string',            # Corn, Soybean, Potato
    'planted_variety': 'string',      # Variety name/code (fuzzy matched)
    'planting_date': 'YYYY-MM-DD',    # ISO format (optional)
    'harvest_date': 'YYYY-MM-DD',     # ISO format (optional)
    'confidence': 'float'             # 0.0-1.0 (optional, default 1.0)
}
```

**Example**:
```csv
location_id,crop_type,planted_variety,planting_date,harvest_date,confidence
LOC_001,Corn,NK7837,2025-04-15,2025-10-30,1.0
LOC_002,Soybean,S0009-J5X,2025-04-20,2025-11-01,0.95
LOC_003,Corn,DK-445,2025-04-12,2025-10-28,0.85
GPS_40.5_-88.2,Soybean,XT2101,2025-04-18,2025-11-02,1.0
```

**Accepted Variety Name Formats**:
- Hybrid codes: `NK7837`, `DK-445`, `AQU-120`
- Soybean varieties: `S0009-J5X`, `XT2101`, `Xitavo`
- Potato varieties: `Katahdin`, `Russet`, `Norland`
- With/without spaces: Matched automatically

---

### 3. Disease Tolerance Databases

**Files**:
- `data/processed/corn_disease_tolerance.csv`
- `data/processed/soybean_disease_tolerance.csv`
- `data/processed/potato_disease_tolerance.csv`

**Format**:
```
Variety, {{Disease_1}}, {{Disease_2}}, {{Disease_3}}, ...
```

**Disease Scale**: 1-9
- 1-3: Resistant (low risk)
- 4-6: Intermediate (moderate risk)
- 7-9: Susceptible (high risk)

**Corn Example**:
```csv
Variety,Grey_Leaf_Spot,Tar_Spot,Northern_Corn_Leaf_Blight,Ear_Rot
NK7837,2,3,2,3
DK-445,2,3,2,3
NK8000,3,4,3,4
AQU-120,1,2,2,2
```

**Soybean Example**:
```csv
Variety,Phytophthora,White_Mould,SDS
S0009-J5X,3,2,3
XT2101,2,3,2
Xitavo,2,2,3
```

---

## Output Data Formats

### 1. Model Performance Results

**File**: `models/model_performance_results.csv`

```csv
Crop,Train_Samples,Test_Samples,Train_AUC,Test_AUC,Test_F1,Model_Type,Features
Corn,2198,550,0.9938,0.9967,0.9835,Logistic Regression,"Spore_Count_Log, Month, Day_of_Year"
Soybean,500,125,0.9980,1.0000,1.0000,Logistic Regression,"Spore_Count_Log, Month, Day_of_Year"
Potato,180,45,0.8950,0.9450,0.9125,Logistic Regression,"Spore_Count_Log, Month, Day_of_Year"
```

### 2. Integrated Variety Features

**File**: `data/processed/spornado_with_variety_features.csv`

Contains all original Spornado columns plus:
```
variety_matched, original_variety, {{Disease}}_Rating, ...
```

**Example**:
```csv
location_id,crop,disease_present,Spore_Count_Log,...,variety_matched,original_variety,Grey_Leaf_Spot_Rating,Tar_Spot_Rating
LOC_001,Corn,1,3.25,...,NK7837,NK7837,2,3
LOC_002,Soybean,0,1.45,...,XT2101,XT2101,2,3
```

### 3. Model Performance with Variety

**File**: `models/model_performance_results_with_variety.csv`

```csv
Crop,Train_Samples,Test_Samples,Train_AUC,Test_AUC,Test_F1,Test_Precision,Test_Recall,Features_Count
Corn,1950,487,0.9927,0.9945,0.9835,0.9862,0.9808,11
Soybean,450,112,0.9975,0.9998,0.9998,0.9999,0.9997,10
Potato,160,40,0.8950,0.9340,0.9125,0.9213,0.9041,8
```

---

## Data Quality Standards

### Validation Checks

Before using any dataset, verify:

**Completeness**:
- ✓ No null values in required columns
- ✓ Each location has multiple records (time series)
- ✓ Minimum 100 samples per crop

**Consistency**:
- ✓ Location IDs match between datasets
- ✓ Crop types standardized (Corn, Soybean, Potato)
- ✓ Date ranges align across sources

**Accuracy**:
- ✓ Weather values in reasonable ranges
- ✓ Spore counts non-negative
- ✓ Disease binary (0/1)
- ✓ Variety names standardized

**Temporal**:
- ✓ Records ordered chronologically
- ✓ No gaps > 30 days
- ✓ Seasonal patterns present

### Quality Metrics

```python
# Check completeness
null_count = df.isnull().sum()
assert null_count.sum() == 0, "Missing values detected"

# Check ranges
assert df['Month'].between(1, 12).all(), "Invalid month values"
assert df['Humidity'].between(0, 100).all(), "Invalid humidity"

# Check temporal order
assert df.groupby('location_id')['date'].is_monotonic_increasing, "Not sorted"

# Minimum samples per crop
for crop in ['Corn', 'Soybean', 'Potato']:
    count = len(df[df['crop'] == crop])
    assert count >= 100, f"{crop}: only {count} samples"
```

---

## Data Preparation Workflow

### Step 1: Load Raw Data
```python
import pandas as pd

# Load Spornado data
spornado = pd.read_csv('data/raw/spornado_weather_spore_data.csv')

# Load variety data (if available)
varieties = pd.read_csv('data/raw/farmer_variety_records.csv')
```

### Step 2: Validate Data
```python
# Check structure
assert 'location_id' in spornado.columns
assert 'crop' in spornado.columns
assert 'disease_present' in spornado.columns

# Check data types
assert spornado['disease_present'].dtype == 'int64'
assert spornado['Spore_Count_Log'].dtype == 'float64'
```

### Step 3: Handle Missing Values
```python
# Option 1: Drop rows with missing values
spornado_clean = spornado.dropna()

# Option 2: Impute with median
spornado['Weather_Feature'] = spornado['Weather_Feature'].fillna(
    spornado['Weather_Feature'].median()
)
```

### Step 4: Integrate Variety Data (if available)
```python
from src.integrate_variety_data import VarietyDataIntegrator

integrator = VarietyDataIntegrator()
integrated = integrator.integrate(
    'data/raw/farmer_variety_records.csv',
    'data/raw/spornado_weather_spore_data.csv'
)
integrated.to_csv('data/processed/spornado_with_variety_features.csv')
```

---

## Data Sources & Attribution

### External Data
- **Seed Catalogs**: Syngenta, BASF
  - Files: `data/external/Syngenta_*.pdf`, `data/external/BASF_*.pdf`
  - Usage: Extract disease tolerance ratings

- **Weather Data**: [Specify source - NOAA, OpenWeatherMap, etc.]
  - Coverage: 3,755+ locations
  - Period: 2022-2025
  - Update frequency: Daily

- **Spore Pressure Data**: Spornado device network
  - Collection: Weekly counts per device
  - Standardization: Log-transformation

---

## FAQ

**Q: What if I don't have variety data?**
A: The model works with weather + spore data alone (99%+ accuracy). Add variety data later as enhancement.

**Q: How do I handle different variety naming conventions?**
A: Fuzzy matching automatically handles variations. Threshold set to 80% similarity.

**Q: Can I use historical data?**
A: Yes, but ensure temporal split (80% train / 20% recent test) to prevent data leakage.

**Q: What disease rating scale should I use?**
A: Standardize all to 1-9 scale (1=resistant, 9=susceptible).

**Q: How do I add a new seed company?**
A: See [ARCHITECTURE.md](ARCHITECTURE.md) for PDF extraction configuration.

---

For implementation examples, see [API_GUIDE.md](API_GUIDE.md).
