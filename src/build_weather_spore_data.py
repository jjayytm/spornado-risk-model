#!/usr/bin/env python3
"""
End-to-end pipeline: merge raw trap data with per-location weather files.

Steps
-----
1. Read All data.xlsx   (no GPS, Excel serial dates)
2. Read More Data.csv   (includes GPS lat / lon)
3. Combine into a single dataset with canonical column names
4. For each row that has GPS: find the nearest weather file within
   MAX_MATCH_KM kilometres using the haversine formula
5. Merge daily weather columns into the combined dataset
6. Save → data/raw/spornado_weather_spore_data.csv

Usage
-----
  python build_weather_spore_data.py
  python build_weather_spore_data.py --help
  python build_weather_spore_data.py \\
      --xlsx   data/raw/All\\ data.xlsx \\
      --csv    data/raw/More\\ Data.csv \\
      --weather data/weather \\
      --output  data/raw/spornado_weather_spore_data.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent  # project root (one level up from src/)

# Default paths (all relative to project root; overridable via CLI)
_DEFAULT_XLSX    = _ROOT / "data" / "raw" / "All data.xlsx"
_DEFAULT_CSV     = _ROOT / "data" / "raw" / "More Data.csv"
_DEFAULT_WEATHER = _ROOT / "data" / "weather"
_DEFAULT_OUTPUT  = _ROOT / "data" / "raw" / "spornado_weather_spore_data.csv"

MAX_MATCH_KM = 15.0   # nearest-weather search radius in kilometres

FINAL_COLUMNS = [
    "Customer name", "Laboratory name", "Spornado serial", "Ref. number",
    "Crop type", "Result CQ", "Spore count", "Cassette", "Test", "Result",
    "Start Date", "End Date", "Location Ref", "GPS latitude", "GPS longitude",
    "Created by", "Created At", "Updated By", "Updated At", "_source",
]

WEATHER_COLS = [
    "temperature_min_c", "temperature_max_c", "temperature_mean_c",
    "humidity_max_percent", "precipitation_mm", "wind_speed_max_kmh",
    "dew_point_min_c",
]


# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two WGS-84 points."""
    R = 6_371.0
    phi1, phi2   = math.radians(lat1), math.radians(lat2)
    dphi         = math.radians(lat2 - lat1)
    dlambda      = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2.0 * math.asin(math.sqrt(a))


def find_nearest(
    lat: float,
    lon: float,
    loc_coords: list[tuple[float, float]],
    max_km: float,
) -> tuple[float, float] | None:
    """Return the (lat, lon) key of the nearest weather location within max_km."""
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

def excel_serial_to_date(serial: str) -> str:
    """Convert an Excel 1900-based date serial to an ISO-8601 string."""
    try:
        n = int(float(serial))
        if n > 59:          # Excel incorrectly treats 1900 as a leap year
            n -= 1
        return (date(1899, 12, 31) + timedelta(days=n)).isoformat()
    except Exception:
        return serial       # return as-is if not parseable


def clean_coord(val: str | None) -> float | None:
    """Strip leading apostrophe / quote artefacts and return float or None."""
    if val is None:
        return None
    cleaned = str(val).strip().strip("'").strip('"').strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def clean_spore_count(val: str) -> str:
    """Remove thousands-separator commas: '100,000' → '100000'."""
    return val.replace(",", "") if val else val


def clean_gps(val: str) -> str:
    """Strip leading apostrophe artefact: ''-81.68 → -81.68."""
    return val.strip().lstrip("'") if val else val


# ---------------------------------------------------------------------------
# Step 1 — parse All data.xlsx
# ---------------------------------------------------------------------------

def read_xlsx(path: Path) -> list[dict]:
    """Return a list of row-dicts with canonical FINAL_COLUMNS names."""
    logger.info("[1/5] Reading %s …", path.name)

    with zipfile.ZipFile(path) as z:
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        try:
            ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(t.text or "" for t in si.findall(".//x:t", ns))
                for si in ss_root.findall(".//x:si", ns)
            ]
        except KeyError:
            shared_strings = []

        sheet_root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    def col_letter_to_index(ref: str) -> int:
        letters = "".join(ch for ch in ref if ch.isalpha())
        idx = 0
        for ch in letters.upper():
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        return idx - 1

    def cell_val(cell: ET.Element) -> str:
        v = cell.find("x:v", ns)
        if v is None or v.text is None:
            return ""
        if cell.get("t") == "s":
            i = int(v.text)
            return shared_strings[i] if i < len(shared_strings) else ""
        return v.text

    rows_el = sheet_root.findall(".//x:row", ns)
    if not rows_el:
        logger.warning("No rows found in %s", path.name)
        return []

    header_by_col: dict[int, str] = {}
    for c in rows_el[0].findall("x:c", ns):
        ref = c.get("r", "")
        if ref:
            header_by_col[col_letter_to_index(ref)] = cell_val(c)

    records: list[dict] = []
    for row_el in rows_el[1:]:
        raw: dict[str, str] = {}
        for c in row_el.findall("x:c", ns):
            ref = c.get("r", "")
            if ref:
                col_idx  = col_letter_to_index(ref)
                col_name = header_by_col.get(col_idx, "")
                if col_name:
                    raw[col_name] = cell_val(c)

        records.append({
            "Customer name":   raw.get("Customer name", ""),
            "Laboratory name": raw.get("User name", ""),
            "Spornado serial": raw.get("Spornado serial", ""),
            "Ref. number":     raw.get("ref_number", ""),
            "Crop type":       raw.get("Crop type", ""),
            "Result CQ":       raw.get("Result CQ", ""),
            "Spore count":     raw.get("Spore count", ""),
            "Cassette":        raw.get("Sample/Cassette", ""),
            "Test":            raw.get("Test", ""),
            "Result":          raw.get("Result", ""),
            "Start Date":      excel_serial_to_date(raw.get("Start Date", "")),
            "End Date":        excel_serial_to_date(raw.get("End Date", "")),
            "Location Ref":    raw.get("gps", ""),
            "GPS latitude":    "",
            "GPS longitude":   "",
            "Created by":      raw.get("Created By", ""),
            "Created At":      excel_serial_to_date(raw.get("Created At", "")),
            "Updated By":      raw.get("Updated By", ""),
            "Updated At":      excel_serial_to_date(raw.get("Updated At", "")),
            "_source":         "All_data_xlsx",
        })

    logger.info("    → %d rows from xlsx", len(records))
    return records


# ---------------------------------------------------------------------------
# Step 2 — read More Data.csv
# ---------------------------------------------------------------------------

def read_more_data(path: Path) -> list[dict]:
    logger.info("[2/5] Reading %s …", path.name)
    records: list[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rec = {k.strip(): v.strip() for k, v in row.items()}
            rec["_source"] = "More_Data_csv"
            records.append(rec)
    logger.info("    → %d rows from csv", len(records))
    return records


# ---------------------------------------------------------------------------
# Step 3 — combine
# ---------------------------------------------------------------------------

def combine(xlsx_rows: list[dict], csv_rows: list[dict]) -> list[dict]:
    logger.info("[3/5] Combining datasets …")
    all_rows: list[dict] = []
    for r in xlsx_rows + csv_rows:
        rec = {col: r.get(col, "") for col in FINAL_COLUMNS}
        rec["Spore count"]   = clean_spore_count(rec["Spore count"])
        rec["GPS latitude"]  = clean_gps(rec["GPS latitude"])
        rec["GPS longitude"] = clean_gps(rec["GPS longitude"])
        all_rows.append(rec)
    logger.info("    → %d total rows", len(all_rows))
    return all_rows


# ---------------------------------------------------------------------------
# Step 4 — load weather files
# ---------------------------------------------------------------------------

def load_weather(
    weather_dir: Path,
) -> tuple[dict[tuple[float, float], dict[str, dict]], list[tuple[float, float]]]:
    logger.info("[4/5] Loading weather files from %s/ …", weather_dir.name)
    weather_files = sorted(weather_dir.glob("weather_*.csv"))
    logger.info("    Found %d files", len(weather_files))

    weather_by_loc: dict[tuple[float, float], dict[str, dict]] = {}
    loc_coords: list[tuple[float, float]] = []

    for wf in weather_files:
        loc_data: dict[str, dict] = {}
        with open(wf, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = row.get("date", "").strip()[:10]   # YYYY-MM-DD
                if d:
                    loc_data[d] = row

        if not loc_data:
            continue

        # Filename format: weather_<lat>_<lon>.csv
        stem  = wf.stem          # e.g. weather_36.4298_-89.9866
        parts = stem.split("_", 1)[1]   # 36.4298_-89.9866
        idx   = parts.rfind("_")
        try:
            lat = round(float(parts[:idx]), 4)
            lon = round(float(parts[idx + 1:]), 4)
        except ValueError:
            logger.warning("Cannot parse lat/lon from filename: %s — skipping.", wf.name)
            continue

        key = (lat, lon)
        weather_by_loc[key] = loc_data
        loc_coords.append(key)

    logger.info("    → %d unique weather locations loaded", len(loc_coords))
    return weather_by_loc, loc_coords


# ---------------------------------------------------------------------------
# Step 5 — merge and save
# ---------------------------------------------------------------------------

def merge_and_save(
    combined: list[dict],
    weather_by_loc: dict[tuple[float, float], dict[str, dict]],
    loc_coords: list[tuple[float, float]],
    output_path: Path,
    max_km: float,
) -> None:
    logger.info("[5/5] Merging weather data and saving …")

    matched  = 0
    no_gps   = 0
    no_match = 0

    out_cols = FINAL_COLUMNS + WEATHER_COLS
    out_rows: list[dict] = []

    for rec in combined:
        lat      = clean_coord(rec.get("GPS latitude"))
        lon      = clean_coord(rec.get("GPS longitude"))
        date_str = rec.get("Start Date", "")[:10]

        wx = {col: "" for col in WEATHER_COLS}

        if lat is None or lon is None:
            no_gps += 1
        else:
            nearest = find_nearest(lat, lon, loc_coords, max_km)
            if nearest and date_str:
                day_wx = weather_by_loc.get(nearest, {}).get(date_str)
                if day_wx:
                    wx = {col: day_wx.get(col, "") for col in WEATHER_COLS}
                    matched += 1
                else:
                    no_match += 1
            else:
                no_match += 1

        out_row = {col: rec.get(col, "") for col in FINAL_COLUMNS}
        out_row.update(wx)
        out_rows.append(out_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(out_rows)

    total = len(out_rows)
    logger.info("  Weather matched     : %d / %d  (%.1f%%)", matched,  total, 100 * matched  / total if total else 0)
    logger.info("  No GPS              : %d / %d  (%.1f%%)", no_gps,   total, 100 * no_gps   / total if total else 0)
    logger.info("  GPS but no match    : %d / %d  (%.1f%%)", no_match, total, 100 * no_match / total if total else 0)
    logger.info("  Saved → %s", output_path)
    logger.info("  Rows: %d  |  Columns: %d", len(out_rows), len(out_cols))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Spornado trap data with per-location weather files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xlsx",    default=str(_DEFAULT_XLSX),    help="Path to All data.xlsx")
    parser.add_argument("--csv",     default=str(_DEFAULT_CSV),     help="Path to More Data.csv")
    parser.add_argument("--weather", default=str(_DEFAULT_WEATHER), help="Directory of weather_<lat>_<lon>.csv files")
    parser.add_argument("--output",  default=str(_DEFAULT_OUTPUT),  help="Output CSV path")
    parser.add_argument("--max-km",  default=MAX_MATCH_KM, type=float,
                        help="Maximum distance (km) to match a GPS point to a weather file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    xlsx_path    = Path(args.xlsx)
    csv_path     = Path(args.csv)
    weather_dir  = Path(args.weather)
    output_path  = Path(args.output)

    for p, label in [(xlsx_path, "--xlsx"), (csv_path, "--csv")]:
        if not p.exists():
            logger.error("File not found (%s): %s", label, p)
            sys.exit(1)

    if not weather_dir.exists():
        logger.error("Weather directory not found (--weather): %s", weather_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Spornado — End-to-End Data Build")
    logger.info("=" * 60)

    xlsx_rows = read_xlsx(xlsx_path)
    csv_rows  = read_more_data(csv_path)
    combined  = combine(xlsx_rows, csv_rows)

    weather_by_loc, loc_coords = load_weather(weather_dir)

    if not loc_coords:
        logger.warning(
            "No weather files loaded. Output will have empty weather columns. "
            "Check that --weather points to a directory of weather_<lat>_<lon>.csv files."
        )

    merge_and_save(combined, weather_by_loc, loc_coords, output_path, args.max_km)
    logger.info("Done.")


if __name__ == "__main__":
    main()
