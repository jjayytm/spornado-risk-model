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

    df = normalise_columns(df)
    df = parse_dates(df)
    df = encode_target(df)
    df = coerce_weather_to_numeric(df)
    df = add_temporal_features(df)
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


def get_feature_columns(cfg: dict, df: pd.DataFrame) -> List[str]:
    """
    Return the ordered list of feature column names that exist in *df*.

    The order is: weather → temporal → rolling derived.
    Missing columns are warned about but not raised as errors so that
    partial-weather datasets (e.g., rows without a weather match) degrade
    gracefully.
    """
    feat_cfg = cfg["features"]
    base = feat_cfg["weather"] + feat_cfg["temporal"]

    rolling_derived: List[str] = []
    for col in feat_cfg["rolling_base_cols"]:
        for w in feat_cfg["rolling_windows"]:
            rolling_derived.append(f"{col}_roll{w}mean")
            rolling_derived.append(f"{col}_roll{w}max")

    all_features = base + rolling_derived
    available = [f for f in all_features if f in df.columns]

    missing = set(all_features) - set(available)
    if missing:
        logger.warning("Feature columns absent from data: %s", sorted(missing))

    return available
