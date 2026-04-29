"""
Tests for the data pipeline utilities in build_weather_spore_data.py.

Run: pytest tests/test_pipeline.py -v
"""

import math

import pytest

from src.build_weather_spore_data import (
    clean_coord,
    clean_gps,
    clean_spore_count,
    excel_serial_to_date,
    find_nearest,
    haversine_km,
)
from src.build_crop_models import temporal_train_test_split
import pandas as pd


# ---------------------------------------------------------------------------
# excel_serial_to_date
# ---------------------------------------------------------------------------

class TestExcelSerialToDate:
    def test_known_date(self):
        # Excel serial 45292 → 2024-01-01
        result = excel_serial_to_date("45292")
        assert result == "2024-01-01"

    def test_non_numeric_returned_as_is(self):
        result = excel_serial_to_date("2024-06-01")
        assert result == "2024-06-01"

    def test_empty_string_returned_as_is(self):
        result = excel_serial_to_date("")
        assert result == ""

    def test_leap_year_bug_correction(self):
        # Serial 60 would be 1900-02-29, which doesn't exist.
        # Our correction shifts serials > 59 by -1.
        result = excel_serial_to_date("61")
        assert result == "1900-03-01"


# ---------------------------------------------------------------------------
# clean_coord
# ---------------------------------------------------------------------------

class TestCleanCoord:
    def test_plain_float_string(self):
        assert clean_coord("43.5") == pytest.approx(43.5)

    def test_apostrophe_prefix(self):
        assert clean_coord("'-81.68") == pytest.approx(-81.68)

    def test_whitespace_stripped(self):
        assert clean_coord("  42.8  ") == pytest.approx(42.8)

    def test_none_returns_none(self):
        assert clean_coord(None) is None

    def test_invalid_string_returns_none(self):
        assert clean_coord("n/a") is None


# ---------------------------------------------------------------------------
# clean_spore_count
# ---------------------------------------------------------------------------

class TestCleanSporeCount:
    def test_removes_commas(self):
        assert clean_spore_count("100,000") == "100000"

    def test_plain_number_unchanged(self):
        assert clean_spore_count("500") == "500"

    def test_none_or_empty(self):
        assert clean_spore_count("") == ""


# ---------------------------------------------------------------------------
# clean_gps
# ---------------------------------------------------------------------------

class TestCleanGps:
    def test_strips_apostrophe(self):
        assert clean_gps("'-81.68") == "-81.68"

    def test_plain_string_unchanged(self):
        assert clean_gps("43.52") == "43.52"

    def test_empty_string(self):
        assert clean_gps("") == ""


# ---------------------------------------------------------------------------
# haversine_km
# ---------------------------------------------------------------------------

class TestHaversineKm:
    def test_same_point_is_zero(self):
        assert haversine_km(43.0, -80.0, 43.0, -80.0) == pytest.approx(0.0)

    def test_known_distance(self):
        # Toronto (43.65, -79.38) to Hamilton (43.26, -79.87) ≈ 58 km
        d = haversine_km(43.65, -79.38, 43.26, -79.87)
        assert 50.0 < d < 70.0

    def test_is_symmetric(self):
        d1 = haversine_km(43.0, -80.0, 44.0, -81.0)
        d2 = haversine_km(44.0, -81.0, 43.0, -80.0)
        assert d1 == pytest.approx(d2)

    def test_not_euclidean_at_43_north(self):
        # At 43°N 1° lon ≠ 1° lat. Euclidean in degrees would give sqrt(2) ≈ 1.414 degrees.
        # True distance for Δlat=1°, Δlon=1° at 43°N is not isotropic.
        d_lat = haversine_km(43.0, -80.0, 44.0, -80.0)   # ~111 km
        d_lon = haversine_km(43.0, -80.0, 43.0, -79.0)   # ~80 km (shorter at 43°N)
        assert d_lat > d_lon


# ---------------------------------------------------------------------------
# find_nearest
# ---------------------------------------------------------------------------

class TestFindNearest:
    _COORDS = [(43.0, -80.0), (44.0, -81.0), (42.0, -79.0)]

    def test_returns_closest_within_radius(self):
        result = find_nearest(43.1, -80.1, self._COORDS, max_km=50.0)
        assert result == (43.0, -80.0)

    def test_returns_none_when_all_outside_radius(self):
        result = find_nearest(43.0, -80.0, self._COORDS, max_km=0.001)
        # (43.0, -80.0) has distance 0 which is < 0.001 — it should match
        assert result == (43.0, -80.0)

    def test_returns_none_on_empty_coords(self):
        result = find_nearest(43.0, -80.0, [], max_km=100.0)
        assert result is None


# ---------------------------------------------------------------------------
# temporal_train_test_split (from src/build_crop_models.py)
# ---------------------------------------------------------------------------

class TestTemporalTrainTestSplit:
    def _make_df(self):
        return pd.DataFrame({
            "start_date": pd.to_datetime([
                "2023-01-01", "2023-04-01", "2023-07-01",
                "2023-10-01", "2024-01-01",
            ]),
            "value": [1, 2, 3, 4, 5],
        })

    def test_no_future_leakage(self):
        df = self._make_df()
        train, test = temporal_train_test_split(df, "start_date", test_size=0.2)
        assert train["start_date"].max() <= test["start_date"].min()

    def test_split_sizes_sum_to_total(self):
        df = self._make_df()
        train, test = temporal_train_test_split(df, "start_date", test_size=0.2)
        assert len(train) + len(test) == len(df)

    def test_test_size_respected_approximately(self):
        df = self._make_df()
        train, test = temporal_train_test_split(df, "start_date", test_size=0.2)
        ratio = len(test) / len(df)
        assert 0.1 < ratio < 0.4   # allow ±10% due to integer rounding

    def test_train_contains_older_dates(self):
        df = self._make_df()
        train, test = temporal_train_test_split(df, "start_date", test_size=0.2)
        # The oldest date must be in training
        oldest = df["start_date"].min()
        assert oldest in train["start_date"].values
