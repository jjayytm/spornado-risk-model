#!/usr/bin/env python3
"""
Fetch per-location daily weather from the Open-Meteo Archive API.

For every unique GPS coordinate in More Data.csv this script downloads
historical daily weather and saves it as:

    data/weather/weather_<lat>_<lon>.csv

These files are the input to build_weather_spore_data.py.

Data source
-----------
Open-Meteo Historical Weather API  —  https://open-meteo.com/
  • Free, no API key required
  • Coverage: global, 1940-present (ERA5 reanalysis + near-real-time)
  • Resolution: 0.25° × 0.25° (≈ 25 km)

Variable mapping
----------------
  Open-Meteo daily variable     →  output column
  ─────────────────────────────────────────────────
  temperature_2m_min            →  temperature_min_c
  temperature_2m_max            →  temperature_max_c
  temperature_2m_mean           →  temperature_mean_c
  relative_humidity_2m_max      →  humidity_max_percent
  precipitation_sum             →  precipitation_mm
  wind_speed_10m_max            →  wind_speed_max_kmh
  dew_point_2m_min              →  dew_point_min_c

Usage
-----
  python -m src.fetch_weather
  python -m src.fetch_weather --help
  python -m src.fetch_weather --start 2025-05-01 --end 2025-10-31
  python -m src.fetch_weather --force    # re-download already-fetched files
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config_loader import load as load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Open-Meteo variable mapping
# ---------------------------------------------------------------------------

# Open-Meteo daily variable name → output CSV column name
_VAR_MAP: dict[str, str] = {
    "temperature_2m_min":       "temperature_min_c",
    "temperature_2m_max":       "temperature_max_c",
    "temperature_2m_mean":      "temperature_mean_c",
    "relative_humidity_2m_max": "humidity_max_percent",
    "precipitation_sum":        "precipitation_mm",
    "wind_speed_10m_max":       "wind_speed_max_kmh",
    "dew_point_2m_min":         "dew_point_min_c",
}

_OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

# Retry settings
_MAX_RETRIES  = 3
_RETRY_DELAY  = 5   # seconds between retries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2.0 * math.asin(math.sqrt(a))


def fetch_open_meteo(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Call Open-Meteo archive API and return a list of daily row dicts.

    Each dict has keys: date, temperature_min_c, temperature_max_c, ...

    Raises
    ------
    RuntimeError  if all retries are exhausted.
    """
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      start_date,
        "end_date":        end_date,
        "daily":           ",".join(_VAR_MAP.keys()),
        "timezone":        "auto",
        "wind_speed_unit": "kmh",
    }
    url = f"{_OPEN_METEO_BASE}?{urllib.parse.urlencode(params)}"

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            break
        except Exception as exc:
            logger.warning("  Attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"Open-Meteo request failed after {_MAX_RETRIES} attempts "
                    f"for ({lat}, {lon}): {exc}"
                ) from exc

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        logger.warning("  No daily data returned for (%.4f, %.4f)", lat, lon)
        return []

    rows: list[dict] = []
    for i, date_str in enumerate(dates):
        row = {"date": date_str}
        for api_col, out_col in _VAR_MAP.items():
            values = daily.get(api_col, [])
            row[out_col] = values[i] if i < len(values) else ""
        rows.append(row)

    return rows


def unique_coords(csv_path: Path) -> list[tuple[float, float]]:
    """
    Extract unique valid (lat, lon) pairs from More Data.csv.

    Rounds to 4 decimal places to avoid near-duplicate coordinate fetches.
    """
    seen:   set[tuple[float, float]] = set()
    result: list[tuple[float, float]] = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                lat = round(float(str(row.get("GPS latitude",  "")).strip().lstrip("'")), 4)
                lon = round(float(str(row.get("GPS longitude", "")).strip().lstrip("'")), 4)
            except ValueError:
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            key = (lat, lon)
            if key not in seen:
                seen.add(key)
                result.append(key)

    return result


def date_range_from_csv(csv_path: Path, valid_year_min: int, valid_year_max: int) -> tuple[str, str]:
    """
    Read Start Date min/max from the CSV (after year guard) and add a 7-day
    buffer on each side to guarantee weather coverage for every trap date.
    """
    import datetime

    def _parse_start_date(raw: str) -> datetime.date | None:
        """
        Parse supported Start Date formats.

        Supports:
        - YYYY-MM-DD (ISO)
        - M/D/YYYY or MM/DD/YYYY (common Excel/US export format)
        """
        value = raw.strip()
        if not value:
            return None

        # Try strict ISO first.
        try:
            return datetime.date.fromisoformat(value[:10])
        except ValueError:
            pass

        # Fallback for slash-delimited dates from CSV exports.
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        return None

    dates = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = str(row.get("Start Date", ""))
            d = _parse_start_date(raw)
            if d is None:
                continue
            if valid_year_min <= d.year <= valid_year_max:
                dates.append(d)

    if not dates:
        raise ValueError("No valid dates found in CSV after year guard.")

    buf        = datetime.timedelta(days=7)
    start_date = (min(dates) - buf).isoformat()
    end_date   = (max(dates) + buf).isoformat()
    return start_date, end_date


def deduplicate_coords(
    coords: list[tuple[float, float]],
    radius_km: float,
) -> list[tuple[float, float]]:
    """
    Cluster nearby coordinates and keep one representative per cluster.

    Avoids fetching multiple times for GPS points that are within radius_km
    of each other (they would all map to the same weather station anyway).
    """
    kept: list[tuple[float, float]] = []
    for lat, lon in coords:
        too_close = any(
            haversine_km(lat, lon, klat, klon) < radius_km
            for klat, klon in kept
        )
        if not too_close:
            kept.append((lat, lon))
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Config ────────────────────────────────────────────────────────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(Path(pre_args.config) if pre_args.config else None)

    csv_path    = Path(cfg["paths"]["data_raw_csv"])
    weather_dir = Path(cfg["paths"]["weather_dir"])
    max_km      = cfg["data"].get("weather_match_max_km", 15.0)
    yr_min      = cfg["data"].get("valid_year_min", 2010)
    yr_max      = cfg["data"].get("valid_year_max", 2030)

    parser = argparse.ArgumentParser(
        description=(
            "Fetch per-location daily weather from Open-Meteo.\n"
            "Saves weather_<lat>_<lon>.csv files to data/weather/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",    default=str(csv_path),    help="More Data.csv path")
    parser.add_argument("--out",    default=str(weather_dir), help="Output directory")
    parser.add_argument("--start",  default=None,             help="Override start date YYYY-MM-DD")
    parser.add_argument("--end",    default=None,             help="Override end date YYYY-MM-DD")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force",  action="store_true",      help="Re-download existing files")
    parser.add_argument(
        "--cluster-km", default=max_km / 2, type=float,
        help="Merge GPS points within this radius (km) to one fetch",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        handlers= [logging.StreamHandler(sys.stdout)],
    )

    csv_path    = Path(args.csv)
    weather_dir = Path(args.out)

    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    weather_dir.mkdir(parents=True, exist_ok=True)

    # ── Coordinates ───────────────────────────────────────────────────────────
    logger.info("Reading GPS coordinates from %s …", csv_path.name)
    coords = unique_coords(csv_path)
    logger.info("  Unique coordinates (raw)      : %d", len(coords))

    coords = deduplicate_coords(coords, args.cluster_km)
    logger.info("  After %.1f km deduplication   : %d fetch locations", args.cluster_km, len(coords))

    # ── Date range ────────────────────────────────────────────────────────────
    if args.start and args.end:
        start_date, end_date = args.start, args.end
    else:
        start_date, end_date = date_range_from_csv(csv_path, yr_min, yr_max)
        if args.start:
            start_date = args.start
        if args.end:
            end_date = args.end

    logger.info("  Fetch date range              : %s → %s", start_date, end_date)
    logger.info("  Output directory              : %s", weather_dir)
    logger.info("  Force re-download             : %s", args.force)
    logger.info("=" * 60)

    # ── Fetch loop ─────────────────────────────────────────────────────────────
    output_cols = ["date"] + list(_VAR_MAP.values())
    skipped     = 0
    fetched     = 0
    failed      = 0

    for idx, (lat, lon) in enumerate(coords, 1):
        fname = weather_dir / f"weather_{lat}_{lon}.csv"

        if fname.exists() and not args.force:
            logger.info("[%d/%d] SKIP  (%.4f, %.4f) → %s already exists",
                        idx, len(coords), lat, lon, fname.name)
            skipped += 1
            continue

        logger.info("[%d/%d] Fetching (%.4f, %.4f) …", idx, len(coords), lat, lon)

        try:
            rows = fetch_open_meteo(lat, lon, start_date, end_date)
        except RuntimeError as exc:
            logger.error("  FAILED: %s", exc)
            failed += 1
            continue

        if not rows:
            logger.warning("  No data returned — skipping.")
            failed += 1
            continue

        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=output_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        logger.info("  Saved %d days → %s", len(rows), fname.name)
        fetched += 1

        # Polite rate-limiting — Open-Meteo free tier allows ~10 req/s
        time.sleep(0.15)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Done.")
    logger.info("  Fetched  : %d", fetched)
    logger.info("  Skipped  : %d  (already existed)", skipped)
    logger.info("  Failed   : %d", failed)
    logger.info("  Files in %s : %d", weather_dir, len(list(weather_dir.glob("weather_*.csv"))))

    if failed:
        logger.warning("  %d location(s) failed — re-run with --force to retry.", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
