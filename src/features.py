"""
Feature engineering pipeline for the Spornado disease risk model.

Responsibilities
----------------
* Normalise raw column names to snake_case
* Parse dates
* Encode binary target  (positive → 1 / negative → 0)
* Coerce weather columns to numeric
* Add temporal features  (month, day_of_year, week_of_year)
* Compute per-location rolling weather statistics
* Return the final feature-column list that the model will train on

Design note
-----------
Spore count is deliberately *not* a model feature.  The Spornado device
classifies a trap as "positive" based on whether spores were detected —
using the count to predict the label is target leakage.  The model must
estimate disease risk from weather and calendar signals alone so that
predictions can be made before a trap is read.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

_COL_MAP: dict[str, str] = {
    "Customer name":    "customer_name",
    "Laboratory name":  "laboratory_name",
    "Spornado serial":  "spornado_serial",
    "Ref. number":      "ref_number",
    "Crop type":        "crop_type",
    "Result CQ":        "result_cq",
    "Spore count":      "spore_count",     # kept for EDA; excluded from model features
    "Cassette":         "cassette",
    "Test":             "test",
    "Result":           "result",
    "Start Date":       "start_date",
    "End Date":         "end_date",
    "Location Ref":     "location_ref",
    "GPS latitude":     "gps_latitude",
    "GPS longitude":    "gps_longitude",
    "Created by":       "created_by",
    "Created At":       "created_at",
    "Updated By":       "updated_by",
    "Updated At":       "updated_at",
    "_source":          "_source",
}

WEATHER_COLS: List[str] = [
    "temperature_min_c",
    "temperature_max_c",
    "temperature_mean_c",
    "humidity_max_percent",
    "precipitation_mm",
    "wind_speed_max_kmh",
    "dew_point_min_c",
]

# Identity / location features — numeric, no scaling required for tree models.
# Always included when present in the data; NaN-safe via the median imputer.
IDENTITY_NUMERIC_COLS: List[str] = [
    "gps_latitude",
    "gps_longitude",
    "deployment_duration_days",
]

TARGET_COL = "disease_present"


# ---------------------------------------------------------------------------
# Individual transformation steps
# ---------------------------------------------------------------------------

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw column names to their snake_case equivalents."""
    rename = {k: v for k, v in _COL_MAP.items() if k in df.columns}
    return df.rename(columns=rename)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("start_date", "end_date", "created_at", "updated_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map 'positive' → 1 and 'negative' → 0 into a new column TARGET_COL.
    Any other value (blank, unknown) becomes NaN and will be dropped.
    """
    df = df.copy()
    df[TARGET_COL] = (
        df["result"]
        .str.strip()
        .str.lower()
        .map({"positive": 1, "negative": 0})
    )
    n_unknown = df[TARGET_COL].isna().sum()
    if n_unknown:
        logger.warning(
            "%d rows have unrecognised 'result' values and will be dropped.", n_unknown
        )
    return df


def coerce_weather_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in WEATHER_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"]        = df["start_date"].dt.month
    df["day_of_year"]  = df["start_date"].dt.dayofyear
    df["week_of_year"] = df["start_date"].dt.isocalendar().week.astype("Int64").astype(float)
    return df


def add_deployment_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trap deployment duration in calendar days (end_date - start_date).

    Agronomic rationale
    -------------------
    A trap deployed for 14 days samples a broader weather window and accumulates
    more spores than one deployed for 5 days.  Duration is therefore an
    independent signal that is always available (start_date and end_date come
    from the trap record, not from GPS), making it suitable for both the full
    weather model and the temporal fallback.

    NaN is set when either date is missing — the downstream median imputer
    handles this without dropping the row.
    """
    df = df.copy()
    if "start_date" not in df.columns or "end_date" not in df.columns:
        logger.warning(
            "Cannot compute deployment_duration_days — start_date or end_date missing."
        )
        df["deployment_duration_days"] = np.nan
        return df

    duration = (df["end_date"] - df["start_date"]).dt.days
    df["deployment_duration_days"] = pd.to_numeric(duration, errors="coerce")

    n_valid = int(df["deployment_duration_days"].notna().sum())
    if n_valid > 0:
        logger.info(
            "Deployment duration: %d valid rows  (mean %.1f d | min %g d | max %g d).",
            n_valid,
            df["deployment_duration_days"].mean(),
            df["deployment_duration_days"].min(),
            df["deployment_duration_days"].max(),
        )
    return df


# ---------------------------------------------------------------------------
# Data quality & availability guards
# ---------------------------------------------------------------------------

def filter_corrupted_dates(
    df: pd.DataFrame,
    min_year: int = 2010,
    max_year: int = 2030,
) -> pd.DataFrame:
    """
    Drop rows whose ``start_date`` falls outside [min_year, max_year].

    Why this matters
    ----------------
    Excel date serials can silently produce nonsense dates such as year 0204
    or 1900 when the underlying cell contains a formula error or a legacy
    two-digit-year.  ``pd.to_datetime`` happily parses them without raising —
    they then corrupt rolling statistics (mis-sorted time windows) and leak
    through the chronological train/test split as phantom early observations.

    Rows with a NaT start_date are kept as-is; they are handled downstream
    by the NaN-feature drop.

    Parameters
    ----------
    min_year, max_year : int
        Valid date range.  Configure in config.yaml under data.valid_year_min
        and data.valid_year_max.
    """
    if "start_date" not in df.columns:
        return df

    before = len(df)
    year   = df["start_date"].dt.year
    valid  = year.between(min_year, max_year, inclusive="both") | df["start_date"].isna()
    df     = df[valid].reset_index(drop=True)

    dropped = before - len(df)
    if dropped:
        logger.warning(
            "Dropped %d rows with start_date outside [%d, %d] "
            "(likely corrupted Excel date serials).",
            dropped, min_year, max_year,
        )
    return df


def flag_weather_availability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean column ``weather_available``: True only when all required
    weather columns contain real (non-NaN) values for that row.

    Two-model training rationale
    ----------------------------
    The merged dataset contains rows from two fundamentally different sources:

    1. **CSV rows (GPS-enabled)** — each trap has GPS coordinates; the pipeline
       matched a nearby weather station and populated the weather columns with
       real, location-specific observations.

    2. **XLSX rows (no GPS)** — trap records exported without coordinates; no
       weather station could be matched, so all seven weather columns are NaN.

    Filling XLSX weather NaNs with the cross-location mean (what SimpleImputer
    does by default) is agronomically meaningless: "average temperature across
    all farms in the dataset" carries zero signal for a specific trap's disease
    risk.  Using those imputed rows in the weather model would dilute the signal
    and produce misleading feature importances.

    ``build_crop_models.py`` uses this flag to train two separate per-crop models:

    - ``_model_weather.pkl``  — GPS rows only, full weather + temporal features
                                (XGBoost; captures nonlinear weather interactions)
    - ``_model.pkl``          — ALL rows, temporal features only
                                (LogisticRegression; 3 features, no imputation noise)

    ``predict.py`` enforces the same policy at inference time: weather/variety
    models require complete weather inputs.  This keeps train/inference policy
    aligned and avoids routing partial-weather rows into the full-triangle path.
    """
    present = [c for c in WEATHER_COLS if c in df.columns]
    df      = df.copy()

    if not present:
        logger.warning(
            "No weather columns found in DataFrame — 'weather_available' will be "
            "False for all rows.  Check that build_weather_spore_data.py ran first."
        )
        df["weather_available"] = False
        return df

    # Full-triangle consistency: require complete weather feature vector.
    df["weather_available"] = df[present].notna().all(axis=1)

    n_with    = int(df["weather_available"].sum())
    n_without = len(df) - n_with
    logger.info(
        "Weather availability: %d rows have GPS-matched weather data, "
        "%d rows do not (no GPS coordinates).",
        n_with, n_without,
    )
    return df


def add_rolling_weather_features(
    df: pd.DataFrame,
    base_cols: List[str],
    windows: List[int],
    group_col: str = "location_ref",
    date_col: str = "start_date",
) -> pd.DataFrame:
    """
    Per-location rolling statistics over the N most recent trap deployments.

    Sorting is done chronologically within each location so that the rolling
    window only looks backward in time — no future leakage.  min_periods=1
    ensures every row gets a value even at the start of a location's history.
    """
    df = df.copy().sort_values([group_col, date_col]).reset_index(drop=True)

    for col in base_cols:
        if col not in df.columns:
            logger.warning("Rolling base column '%s' not in DataFrame — skipping.", col)
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        df[col] = numeric
        for w in windows:
            df[f"{col}_roll{w}mean"] = (
                df.groupby(group_col)[col]
                  .transform(lambda s, _w=w: s.rolling(_w, min_periods=1).mean())
            )
            df[f"{col}_roll{w}max"] = (
                df.groupby(group_col)[col]
                  .transform(lambda s, _w=w: s.rolling(_w, min_periods=1).max())
            )

    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline to the merged raw dataset.

    Parameters
    ----------
    df  : Raw DataFrame from spornado_weather_spore_data.csv
    cfg : Config dict loaded by src.config_loader.load()

    Returns
    -------
    DataFrame with all feature columns, the binary TARGET_COL, and metadata
    columns (crop_type, start_date, location_ref) needed for splitting.
    """
    feat_cfg = cfg["features"]
    data_cfg = cfg.get("data", {})

    df = normalise_columns(df)
    df = parse_dates(df)

    # ── Date quality guard ────────────────────────────────────────────────────
    # Must run BEFORE temporal features so rolling windows are not mis-sorted
    # by phantom years like 0204 that slip through pd.to_datetime silently.
    min_year = data_cfg.get("valid_year_min", 2010)
    max_year = data_cfg.get("valid_year_max", 2030)
    df = filter_corrupted_dates(df, min_year, max_year)

    df = encode_target(df)
    df = coerce_weather_to_numeric(df)

    # ── Weather availability flag ─────────────────────────────────────────────
    # Must run BEFORE rolling stats so the flag reflects raw GPS-match status,
    # not the imputed values (which would make every row look "available").
    df = flag_weather_availability(df)

    df = add_temporal_features(df)
    df = add_deployment_duration(df)
    df = add_rolling_weather_features(
        df,
        base_cols=feat_cfg["rolling_base_cols"],
        windows=feat_cfg["rolling_windows"],
    )

    # Drop rows where the target is undefined
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with undefined target.", dropped)

    logger.info("Feature engineering complete: %d rows, %d columns.", len(df), len(df.columns))
    return df


def get_temporal_feature_columns(cfg: dict) -> List[str]:
    """
    Return only the temporal feature columns (month, day_of_year, week_of_year).

    These columns are derived from the trap date and are available for *every*
    row regardless of GPS or weather-station coverage.  They are the feature set
    used by ``_model.pkl`` — the temporal-only fallback trained on all rows.

    Parameters
    ----------
    cfg : dict
        Config loaded by src.config_loader.load().
    """
    return list(cfg["features"].get("temporal", ["month", "day_of_year", "week_of_year"]))


def get_feature_columns(cfg: dict, df: pd.DataFrame) -> List[str]:
    """
    Return the ordered list of *numeric* feature column names that exist in *df*.

    Column order
    ------------
    weather  →  identity/location  →  temporal  →  rolling derived

    Notes
    -----
    * Disease dummies (from the 'test' column) are NOT included here — they
      are computed per-crop inside build_crop_models.py after the crop filter
      and appended to the list returned by this function.
    * Location columns (gps_latitude, gps_longitude) are NaN for non-GPS rows;
      the downstream median imputer handles them without dropping rows.
    * deployment_duration_days is always included when present.
    * Missing columns are warned about but not raised as errors so that
      partial-weather datasets degrade gracefully.
    """
    feat_cfg = cfg["features"]

    weather  = feat_cfg.get("weather", [])
    identity = feat_cfg.get("identity", [])   # lat, lon, duration
    temporal = feat_cfg.get("temporal", [])

    rolling_derived: List[str] = []
    for col in feat_cfg["rolling_base_cols"]:
        for w in feat_cfg["rolling_windows"]:
            rolling_derived.append(f"{col}_roll{w}mean")
            rolling_derived.append(f"{col}_roll{w}max")

    all_features = weather + identity + temporal + rolling_derived
    available    = [f for f in all_features if f in df.columns]

    missing = set(all_features) - set(available)
    if missing:
        logger.warning("Feature columns absent from data: %s", sorted(missing))

    return available
