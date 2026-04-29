"""
Tests for src/features.py — feature engineering correctness.

Run: pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    TARGET_COL,
    WEATHER_COLS,
    add_rolling_weather_features,
    add_temporal_features,
    coerce_weather_to_numeric,
    encode_target,
    get_feature_columns,
    normalise_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_df():
    """Minimal DataFrame mimicking the raw merged CSV."""
    return pd.DataFrame({
        "Customer name":   ["Acme", "Acme"],
        "Crop type":       ["Corn", "Soybean"],
        "Result":          ["positive", "negative"],
        "Start Date":      ["2024-07-01", "2024-08-15"],
        "End Date":        ["2024-07-07", "2024-08-22"],
        "Location Ref":    ["LOC001", "LOC002"],
        "GPS latitude":    ["43.5", "42.8"],
        "GPS longitude":   ["-80.1", "-81.3"],
        "temperature_max_c": ["28.0", "25.5"],
        "humidity_max_percent": ["85.0", "72.0"],
        "precipitation_mm": ["5.0", "0.0"],
        "temperature_min_c": ["14.0", "12.0"],
        "temperature_mean_c": ["21.0", "18.5"],
        "wind_speed_max_kmh": ["20.0", "15.0"],
        "dew_point_min_c":  ["10.0", "8.0"],
    })


@pytest.fixture()
def min_cfg():
    """Minimal config dict for feature tests."""
    return {
        "features": {
            "weather":         WEATHER_COLS,
            "temporal":        ["month", "day_of_year", "week_of_year"],
            "rolling_windows": [3],
            "rolling_base_cols": ["temperature_max_c", "humidity_max_percent"],
        }
    }


# ---------------------------------------------------------------------------
# normalise_columns
# ---------------------------------------------------------------------------

class TestNormaliseColumns:
    def test_renames_crop_type(self, raw_df):
        result = normalise_columns(raw_df)
        assert "crop_type" in result.columns
        assert "Crop type" not in result.columns

    def test_renames_start_date(self, raw_df):
        result = normalise_columns(raw_df)
        assert "start_date" in result.columns

    def test_unknown_columns_preserved(self, raw_df):
        raw_df = raw_df.copy()
        raw_df["unknown_col"] = 1
        result = normalise_columns(raw_df)
        assert "unknown_col" in result.columns

    def test_weather_cols_already_snake_case(self, raw_df):
        result = normalise_columns(raw_df)
        for col in WEATHER_COLS:
            if col in raw_df.columns:
                assert col in result.columns


# ---------------------------------------------------------------------------
# encode_target
# ---------------------------------------------------------------------------

class TestEncodeTarget:
    def test_positive_maps_to_one(self):
        df = pd.DataFrame({"result": ["positive"]})
        out = encode_target(df)
        assert out[TARGET_COL].iloc[0] == 1

    def test_negative_maps_to_zero(self):
        df = pd.DataFrame({"result": ["negative"]})
        out = encode_target(df)
        assert out[TARGET_COL].iloc[0] == 0

    def test_case_insensitive(self):
        df = pd.DataFrame({"result": ["Positive", "NEGATIVE", "positive"]})
        out = encode_target(df)
        assert list(out[TARGET_COL]) == [1, 0, 1]

    def test_unknown_value_becomes_nan(self):
        df = pd.DataFrame({"result": ["unknown", "", None]})
        out = encode_target(df)
        assert out[TARGET_COL].isna().all()

    def test_original_result_column_unchanged(self):
        df = pd.DataFrame({"result": ["positive"]})
        out = encode_target(df)
        assert "result" in out.columns


# ---------------------------------------------------------------------------
# coerce_weather_to_numeric
# ---------------------------------------------------------------------------

class TestCoerceWeatherToNumeric:
    def test_string_numbers_become_float(self):
        df = pd.DataFrame({"temperature_max_c": ["28.5", "25.0"]})
        out = coerce_weather_to_numeric(df)
        assert out["temperature_max_c"].dtype == float

    def test_invalid_strings_become_nan(self):
        df = pd.DataFrame({"temperature_max_c": ["n/a", "28.0"]})
        out = coerce_weather_to_numeric(df)
        assert np.isnan(out["temperature_max_c"].iloc[0])
        assert out["temperature_max_c"].iloc[1] == 28.0

    def test_missing_weather_col_ignored(self):
        df = pd.DataFrame({"some_other_col": [1, 2]})
        out = coerce_weather_to_numeric(df)   # should not raise
        assert "some_other_col" in out.columns


# ---------------------------------------------------------------------------
# add_temporal_features
# ---------------------------------------------------------------------------

class TestAddTemporalFeatures:
    def test_month_extracted(self):
        df = pd.DataFrame({"start_date": pd.to_datetime(["2024-07-15"])})
        out = add_temporal_features(df)
        assert out["month"].iloc[0] == 7

    def test_day_of_year_extracted(self):
        df = pd.DataFrame({"start_date": pd.to_datetime(["2024-01-01"])})
        out = add_temporal_features(df)
        assert out["day_of_year"].iloc[0] == 1

    def test_week_of_year_extracted(self):
        df = pd.DataFrame({"start_date": pd.to_datetime(["2024-07-01"])})
        out = add_temporal_features(df)
        assert out["week_of_year"].iloc[0] == 27   # ISO week 27 for 2024-07-01


# ---------------------------------------------------------------------------
# add_rolling_weather_features
# ---------------------------------------------------------------------------

class TestAddRollingWeatherFeatures:
    def _make_df(self):
        return pd.DataFrame({
            "location_ref":      ["A", "A", "A", "B", "B"],
            "start_date":        pd.to_datetime(
                ["2024-06-01", "2024-07-01", "2024-08-01",
                 "2024-06-01", "2024-07-01"]
            ),
            "temperature_max_c": [20.0, 25.0, 30.0, 15.0, 18.0],
        })

    def test_rolling_mean_computed(self):
        df = self._make_df()
        out = add_rolling_weather_features(df, ["temperature_max_c"], [2])
        assert "temperature_max_c_roll2mean" in out.columns

    def test_no_future_leakage(self):
        df = self._make_df()
        out = add_rolling_weather_features(df, ["temperature_max_c"], [2])
        # After sorting, first row of location A should equal its own value (min_periods=1)
        loc_a = out[out["location_ref"] == "A"].reset_index(drop=True)
        assert loc_a["temperature_max_c_roll2mean"].iloc[0] == pytest.approx(20.0)

    def test_rolling_is_per_location(self):
        df = self._make_df()
        out = add_rolling_weather_features(df, ["temperature_max_c"], [3])
        # Location B only has 2 rows; its rolling-3 mean should be its own mean, not influenced by A
        loc_b = out[out["location_ref"] == "B"].reset_index(drop=True)
        assert loc_b["temperature_max_c_roll3mean"].iloc[0] == pytest.approx(15.0)

    def test_missing_base_col_skipped_gracefully(self):
        df = pd.DataFrame({"location_ref": ["A"], "start_date": pd.to_datetime(["2024-01-01"])})
        out = add_rolling_weather_features(df, ["nonexistent_col"], [3])
        assert "nonexistent_col_roll3mean" not in out.columns


# ---------------------------------------------------------------------------
# get_feature_columns
# ---------------------------------------------------------------------------

class TestGetFeatureColumns:
    def test_returns_only_existing_columns(self, min_cfg):
        df = pd.DataFrame({
            "temperature_max_c": [1.0],
            "month": [7],
            "nonexistent": [99],
        })
        cols = get_feature_columns(min_cfg, df)
        assert all(c in df.columns for c in cols)
        assert "nonexistent" not in cols

    def test_all_weather_cols_included_when_present(self, min_cfg):
        df = pd.DataFrame({c: [1.0] for c in WEATHER_COLS + ["month", "day_of_year", "week_of_year"]})
        cols = get_feature_columns(min_cfg, df)
        for w in WEATHER_COLS:
            assert w in cols

    def test_spore_count_not_included(self, min_cfg):
        df = pd.DataFrame({c: [1.0] for c in WEATHER_COLS + ["spore_count", "month"]})
        cols = get_feature_columns(min_cfg, df)
        assert "spore_count" not in cols
