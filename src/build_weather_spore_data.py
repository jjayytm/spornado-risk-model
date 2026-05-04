#!/usr/bin/env python3
"""
Build the model-ready dataset: enrich GPS trap data with per-location weather.

Why only More Data.csv?
-----------------------
The disease triangle requires all three components at prediction time:
  1. Weather  (open-source, location-specific)
  2. Hybrid   (variety susceptibility)
  3. Spornado (spore count from trap)

All data.xlsx has no GPS coordinates, so weather cannot be matched to those
rows.  Training on structurally incomplete triangle data would teach the model
a corrupted pattern — more rows is not better when a third of the triangle is
systematically missing.

More Data.csv has GPS for every row, enabling full weather enrichment and
complete triangle representation.  This is the only source used for model
training.

All data.xlsx is preserved separately for spore count percentile analysis
(Spornado score normalisation), which does not require GPS or weather.

Steps
-----
1. Load More Data.csv  (GPS trap data — primary model training source)
2. Validate coordinates and dates
3. Load per-location weather files from weather_dir
4. Match each row to its nearest weather station (within max_km)
5. Merge weather columns into the dataset
6. Save enriched CSV → paths.data_merged

Configuration
-------------
All tunable values (paths, match radius, weather columns) are read from
config/config.yaml via src.config_loader.  CLI flags override config values
for one-off runs without editing the config file.

Usage
-----
  python -m src.build_weather_spore_data
  python -m src.build_weather_spore_data --help
  python -m src.build_weather_spore_data --csv "data/raw/More Data.csv"
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config_loader import load as load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed schema — dictated by More Data.csv column names.
# Change only if the upstream file format changes.
# ---------------------------------------------------------------------------

# Columns expected in More Data.csv (after strip)
_CSV_DATE_COLS = ("Start Date", "End Date", "Created At", "Updated At")

# Weather column names as they appear inside weather_<lat>_<lon>.csv files.
# These must match config.yaml features.weather exactly — validated at startup.
_WEATHER_FILE_COL_ALIASES: dict[str, str] = {
    # weather file col → canonical name  (identity if same)
}


# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two WGS-84 points."""
    R    = 6_371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2.0 * math.asin(math.sqrt(a))


def find_nearest(
    lat: float,
    lon: float,
    loc_coords: list[tuple[float, float]],
    max_km: float,
) -> tuple[float, float] | None:
    """Return (lat, lon) of the nearest weather station within max_km, or None."""
    best_key  = None
    best_dist = max_km
    for clat, clon in loc_coords:
        d = haversine_km(lat, lon, clat, clon)
        if d < best_dist:
            best_dist = d
            best_key  = (clat, clon)
    return best_key


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def clean_coord(val: str | None) -> float | None:
    """Strip apostrophe/quote artefacts and parse to float, or return None."""
    if val is None:
        return None
    try:
        return float(str(val).strip().strip("'\""))
    except ValueError:
        return None


def clean_spore_count(val: str) -> str:
    """Remove thousands-separator commas: '100,000' → '100000'."""
    return val.replace(",", "") if val else val


def normalize_date(value: str | None) -> str:
    """
    Normalize common date formats to ISO YYYY-MM-DD.

    Supports:
    - YYYY-MM-DD
    - M/D/YYYY, MM/DD/YYYY
    - M/D/YY, MM/DD/YY
    Returns empty string if parsing fails.
    """
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""

    # Common CSV export with time component: keep date token only.
    token = raw.split()[0]

    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            if fmt == "%Y-%m-%d":
                return date.fromisoformat(token).isoformat()
            return datetime.strptime(token, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def is_valid_coord(lat: float | None, lon: float | None) -> bool:
    """Basic sanity check — WGS-84 bounds."""
    if lat is None or lon is None:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


# ---------------------------------------------------------------------------
# Step 1 — load More Data.csv
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    """
    Load the GPS trap data CSV.

    Normalises column names (strip whitespace), cleans GPS artefacts,
    and logs a per-column missing-value summary for data quality awareness.
    """
    logger.info("[1/4] Loading %s …", path.name)

    records: list[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rec = {k.strip(): v.strip() for k, v in row.items()}
            # Clean known artefacts
            rec["GPS latitude"]  = rec.get("GPS latitude",  "").lstrip("'")
            rec["GPS longitude"] = rec.get("GPS longitude", "").lstrip("'")
            rec["Spore count"]   = clean_spore_count(rec.get("Spore count", ""))
            rec["Start Date"]    = normalize_date(rec.get("Start Date", ""))
            records.append(rec)

    if not records:
        raise ValueError(f"No records found in {path}. Check the file path and format.")

    # ── Data quality summary ─────────────────────────────────────────────────
    total = len(records)
    cols  = list(records[0].keys())
    logger.info("  Loaded %d rows × %d columns", total, len(cols))

    missing_summary = {
        col: sum(1 for r in records if not r.get(col, "").strip())
        for col in cols
    }
    problem_cols = {c: n for c, n in missing_summary.items() if n > 0}
    if problem_cols:
        logger.info("  Missing value counts:")
        for col, n in sorted(problem_cols.items(), key=lambda x: -x[1]):
            logger.info("    %-35s %d / %d  (%.1f%%)", col, n, total, 100 * n / total)

    # ── GPS coverage ─────────────────────────────────────────────────────────
    has_gps = sum(
        1 for r in records
        if is_valid_coord(clean_coord(r.get("GPS latitude")),
                          clean_coord(r.get("GPS longitude")))
    )
    logger.info(
        "  GPS-valid rows : %d / %d  (%.1f%%)",
        has_gps, total, 100 * has_gps / total,
    )
    if has_gps < total:
        logger.warning(
            "  %d rows have invalid/missing GPS — these will have no weather "
            "enrichment and will be excluded from model training.",
            total - has_gps,
        )

    return records


# ---------------------------------------------------------------------------
# Step 2 — load weather files
# ---------------------------------------------------------------------------

def load_weather(
    weather_dir: Path,
    weather_cols: list[str],
) -> tuple[dict[tuple[float, float], dict[str, dict]], list[tuple[float, float]]]:
    """
    Load all per-location weather CSVs from weather_dir.

    Validates that expected weather columns are present in at least one file
    so mismatches with config.yaml are caught before the merge step.

    Expected filename format: weather_<lat>_<lon>.csv

    Returns
    -------
    weather_by_loc : (lat, lon) → {date_str → row_dict}
    loc_coords     : list of (lat, lon) keys for nearest-neighbour search
    """
    logger.info("[2/4] Loading weather files from %s/ …", weather_dir.name)

    weather_files = sorted(weather_dir.glob("weather_*.csv"))
    if not weather_files:
        raise FileNotFoundError(
            f"No weather_*.csv files found in {weather_dir}.\n"
            "Fetch weather data first or point --weather at the correct directory."
        )
    logger.info("  Found %d weather files", len(weather_files))

    weather_by_loc: dict[tuple[float, float], dict[str, dict]] = {}
    loc_coords:     list[tuple[float, float]]                   = []
    cols_validated = False

    for wf in weather_files:
        loc_data: dict[str, dict] = {}
        with open(wf, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row.get("date", "").strip()[:10]
                if d:
                    loc_data[d] = row

            # Validate that expected weather columns exist (once)
            if not cols_validated and reader.fieldnames:
                missing_cols = [c for c in weather_cols if c not in reader.fieldnames]
                if missing_cols:
                    logger.warning(
                        "  Weather file '%s' is missing columns: %s\n"
                        "  Check that config.yaml features.weather matches "
                        "the weather file schema.",
                        wf.name, missing_cols,
                    )
                cols_validated = True

        if not loc_data:
            logger.debug("  Skipping empty weather file: %s", wf.name)
            continue

        # Parse lat/lon from filename: weather_<lat>_<lon>.csv
        stem  = wf.stem
        parts = stem.split("_", 1)[1]
        idx   = parts.rfind("_")
        try:
            lat = round(float(parts[:idx]),     4)
            lon = round(float(parts[idx + 1:]), 4)
        except ValueError:
            logger.warning("Cannot parse lat/lon from '%s' — skipping.", wf.name)
            continue

        key = (lat, lon)
        weather_by_loc[key] = loc_data
        loc_coords.append(key)

    logger.info("  Loaded %d unique weather locations", len(loc_coords))
    return weather_by_loc, loc_coords


# ---------------------------------------------------------------------------
# Step 3 — enrich with weather
# ---------------------------------------------------------------------------

def enrich_with_weather(
    records:        list[dict],
    weather_by_loc: dict[tuple[float, float], dict[str, dict]],
    loc_coords:     list[tuple[float, float]],
    weather_cols:   list[str],
    max_km:         float,
) -> list[dict]:
    """
    Join weather data onto each trap record using GPS + trap date.

    Match logic
    -----------
    1. Parse lat/lon from the record.
    2. Find the nearest weather station within max_km.
    3. Look up the row's Start Date in that station's daily records.
    4. Copy weather_cols into the record.  Missing = empty string.

    Rows that fail step 1 or 2 are kept in the output with empty weather
    columns so the full dataset is preserved; downstream feature engineering
    (build_crop_models.py) uses the weather_available flag to route them.

    Returns the enriched records list (new dicts — originals unchanged).
    """
    logger.info("[3/4] Enriching %d records with weather …", len(records))

    matched    = 0
    no_gps     = 0
    no_station = 0
    no_date    = 0

    enriched: list[dict] = []

    for rec in records:
        out = dict(rec)   # copy — do not mutate original

        lat      = clean_coord(rec.get("GPS latitude"))
        lon      = clean_coord(rec.get("GPS longitude"))
        date_str = normalize_date(rec.get("Start Date", ""))

        wx = {col: "" for col in weather_cols}

        if not is_valid_coord(lat, lon):
            no_gps += 1
        elif not date_str:
            no_date += 1
        else:
            nearest = find_nearest(lat, lon, loc_coords, max_km)
            if nearest is None:
                no_station += 1
            else:
                day_wx = weather_by_loc.get(nearest, {}).get(date_str)
                if day_wx:
                    wx = {col: day_wx.get(col, "") for col in weather_cols}
                    matched += 1
                else:
                    no_station += 1

        out.update(wx)
        enriched.append(out)

    total = len(enriched)
    logger.info(
        "  Weather matched        : %d / %d  (%.1f%%)",
        matched, total, 100 * matched / total if total else 0,
    )
    logger.info(
        "  No valid GPS           : %d / %d  (%.1f%%)",
        no_gps, total, 100 * no_gps / total if total else 0,
    )
    logger.info(
        "  No nearby station      : %d / %d  (%.1f%%)",
        no_station, total, 100 * no_station / total if total else 0,
    )
    if no_date:
        logger.warning(
            "  Missing Start Date     : %d rows — cannot look up daily weather.",
            no_date,
        )

    usable = matched
    logger.info(
        "  Rows usable for training (full triangle): %d / %d  (%.1f%%)",
        usable, total, 100 * usable / total if total else 0,
    )

    return enriched


# ---------------------------------------------------------------------------
# Step 4 — save
# ---------------------------------------------------------------------------

def save(
    records:      list[dict],
    weather_cols: list[str],
    output_path:  Path,
) -> None:
    """Write enriched records to CSV, preserving original columns + weather."""
    if not records:
        raise ValueError("No records to save.")

    all_cols = list(records[0].keys())
    # Ensure weather cols are present in the header even if all empty
    for col in weather_cols:
        if col not in all_cols:
            all_cols.append(col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    logger.info(
        "[4/4] Saved -> %s  (%d rows x %d columns)",
        output_path, len(records), len(all_cols),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Load config first so defaults come from one place ─────────────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(Path(pre_args.config) if pre_args.config else None)

    default_csv     = cfg["paths"]["data_raw_csv"]
    default_weather = cfg["paths"]["weather_dir"]
    default_output  = cfg["paths"]["data_merged"]
    default_max_km  = cfg["data"].get("weather_match_max_km", 15.0)
    weather_cols    = cfg["features"]["weather"]

    parser = argparse.ArgumentParser(
        description=(
            "Enrich Spornado GPS trap data with per-location weather.\n"
            "Primary input: More Data.csv (GPS rows only — full triangle possible)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",     default=default_csv,     help="Path to More Data.csv")
    parser.add_argument("--weather", default=default_weather, help="Directory of weather_<lat>_<lon>.csv files")
    parser.add_argument("--output",  default=default_output,  help="Output enriched CSV path")
    parser.add_argument("--max-km",  default=default_max_km,  type=float,
                        help="Maximum GPS-to-station distance in km")
    parser.add_argument("--config",  default=None,            help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        handlers= [logging.StreamHandler(sys.stdout)],
    )

    csv_path    = Path(args.csv)
    weather_dir = Path(args.weather)
    output_path = Path(args.output)

    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)
    if not weather_dir.exists():
        logger.error("Weather directory not found: %s", weather_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Spornado — Weather Enrichment Pipeline")
    logger.info("=" * 60)
    logger.info("  Input            : %s", csv_path)
    logger.info("  Weather dir      : %s", weather_dir)
    logger.info("  Output           : %s", output_path)
    logger.info("  Max match radius : %.1f km", args.max_km)
    logger.info("  Weather columns  : %s", weather_cols)

    records                 = load_csv(csv_path)
    weather_by_loc, coords  = load_weather(weather_dir, weather_cols)
    enriched                = enrich_with_weather(
                                  records, weather_by_loc, coords,
                                  weather_cols, args.max_km,
                              )
    save(enriched, weather_cols, output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
