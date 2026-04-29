#!/usr/bin/env python3
"""
Integrate farmer variety records with the Spornado dataset.

When OCI supplies crop variety data, run this script to fuzzy-match
planted varieties against the disease tolerance database and merge the
resulting tolerance ratings into the weather+spore dataset.

Usage
-----
  python -m src.integrate_variety_data --help
  python -m src.integrate_variety_data \\
      --input  data/raw/farmer_varieties.csv \\
      --spornado data/raw/spornado_weather_spore_data.csv \\
      --output data/processed/spornado_with_variety_features.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from fuzzywuzzy import fuzz, process

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config_loader import load as load_config

logger = logging.getLogger(__name__)


def _setup_logging(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"]),
        format=cfg["logging"]["format"],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "integrate_variety.log", mode="w", encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Variety data integrator
# ---------------------------------------------------------------------------

class VarietyDataIntegrator:
    """Merge farmer variety records with the Spornado weather+spore dataset."""

    _REQUIRED_COLS = ["location_id", "crop_type", "planted_variety"]

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._threshold: int = cfg["variety_matching"]["fuzzy_threshold"]
        self._tolerance_dbs: dict[str, pd.DataFrame] = {}
        self._load_tolerance_dbs()

    # ------------------------------------------------------------------ I/O

    def _load_tolerance_dbs(self) -> None:
        tolerance_dir = Path(self._cfg["paths"]["tolerance_dir"])
        for crop in self._cfg.get("crops", ["Corn", "Soybean", "Potato"]):
            fp = tolerance_dir / f"{crop.lower()}_disease_tolerance.csv"
            if fp.exists():
                self._tolerance_dbs[crop] = pd.read_csv(fp)
                logger.info("Loaded %s tolerance DB: %d varieties", crop, len(self._tolerance_dbs[crop]))
            else:
                logger.warning("Tolerance file not found, skipping %s: %s", crop, fp)

    def read_variety_data(self, filepath: str | Path) -> pd.DataFrame:
        fp = Path(filepath)
        logger.info("Reading variety data: %s", fp)
        if fp.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(fp)
        else:
            try:
                df = pd.read_csv(fp, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(fp, encoding="latin-1")

        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        missing = [c for c in self._REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Variety file is missing required columns: {missing}")

        logger.info("Loaded %d farmer variety records", len(df))
        return df

    # ---------------------------------------------------------- Fuzzy match

    def _fuzzy_match_variety(self, planted_variety: str, crop_type: str) -> str | None:
        db = self._tolerance_dbs.get(crop_type)
        if db is None or db.empty:
            return None
        candidates = db["Variety"].tolist()
        matches = process.extract(
            planted_variety,
            candidates,
            scorer=fuzz.token_set_ratio,
            limit=1,
        )
        if matches and matches[0][1] >= self._threshold:
            return matches[0][0]
        return None

    def _get_tolerance_features(self, variety: str, crop_type: str) -> dict:
        db = self._tolerance_dbs.get(crop_type)
        if db is None:
            return {}
        row = db[db["Variety"] == variety]
        if row.empty:
            return {}
        disease_cols = [c for c in db.columns if c != "Variety"]
        features: dict = {}
        for col in disease_cols:
            try:
                features[f"{col}_Rating"] = float(row[col].iloc[0])
            except (ValueError, TypeError):
                features[f"{col}_Rating"] = row[col].iloc[0]
        return features

    # ----------------------------------------------------------- Integration

    def integrate(
        self,
        variety_filepath: str | Path,
        spornado_filepath: str | Path,
    ) -> pd.DataFrame | None:
        """
        Full integration workflow:
          1. Load and validate variety records
          2. Vectorised fuzzy match against tolerance DB
          3. Extract disease tolerance ratings
          4. Left-merge onto spornado dataset
        """
        variety_df   = self.read_variety_data(variety_filepath)
        spornado_df  = pd.read_csv(spornado_filepath, low_memory=False)
        logger.info("Spornado dataset: %d records", len(spornado_df))

        # --- fuzzy match (vectorised via apply; avoids iterrows overhead)
        logger.info("Fuzzy-matching varieties (threshold=%d) …", self._threshold)
        variety_df = variety_df.copy()
        variety_df["matched_variety"] = variety_df.apply(
            lambda r: self._fuzzy_match_variety(
                str(r["planted_variety"]), str(r["crop_type"])
            ),
            axis=1,
        )

        n_matched = variety_df["matched_variety"].notna().sum()
        logger.info(
            "Matched %d / %d varieties  (%.1f%%)",
            n_matched, len(variety_df), 100 * n_matched / len(variety_df) if len(variety_df) else 0,
        )

        # --- extract tolerance ratings for matched rows
        logger.info("Extracting disease tolerance ratings …")
        matched_rows = variety_df[variety_df["matched_variety"].notna()].copy()

        feature_records = matched_rows.apply(
            lambda r: {
                "location_id":      r["location_id"],
                "crop_type":        r["crop_type"],
                "original_variety": r["planted_variety"],
                "variety_matched":  r["matched_variety"],
                **self._get_tolerance_features(r["matched_variety"], r["crop_type"]),
            },
            axis=1,
        ).tolist()

        features_df = pd.DataFrame(feature_records)

        if features_df.empty:
            logger.error("No varieties could be matched — integration aborted.")
            return None

        logger.info("Created variety features for %d records", len(features_df))

        # --- merge onto spornado
        spornado_df["location_id"] = spornado_df["location_id"].astype(str)
        features_df["location_id"] = features_df["location_id"].astype(str)

        rating_cols = [c for c in features_df.columns if c.endswith("_Rating")]
        merge_cols  = ["location_id", "crop_type", "original_variety", "variety_matched"] + rating_cols

        integrated_df = spornado_df.merge(
            features_df[merge_cols],
            on=["location_id", "crop_type"],
            how="left",
        )
        logger.info(
            "Integrated dataset: %d rows, %d columns",
            len(integrated_df), len(integrated_df.columns),
        )
        return integrated_df

    def save(self, df: pd.DataFrame, output_path: str | Path) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved → %s  (%d rows, %d columns)", out, len(df), len(df.columns))

    def print_summary(self, df: pd.DataFrame) -> None:
        print()
        print("=" * 60)
        print("INTEGRATION SUMMARY")
        print("=" * 60)
        n_with_variety = df["variety_matched"].notna().sum() if "variety_matched" in df.columns else 0
        print(f"Total records           : {len(df)}")
        print(f"Records with variety    : {n_with_variety}")
        print(f"Total columns           : {len(df.columns)}")
        if n_with_variety:
            sample = df[df["variety_matched"].notna()][
                ["location_id", "crop_type", "original_variety", "variety_matched"]
            ].head()
            print("\nSample matches:")
            print(sample.to_string(index=False))
        print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate farmer variety records with the Spornado dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",    required=True,    help="Farmer variety records (CSV or Excel)")
    parser.add_argument("--spornado", required=True,    help="Merged weather+spore CSV")
    parser.add_argument("--output",   default=None,     help="Output path (default: from config)")
    parser.add_argument("--config",   default=None,     help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    _setup_logging(cfg)

    output_path = args.output or cfg["output"]["integrated_data"]

    for fp, label in [(args.input, "--input"), (args.spornado, "--spornado")]:
        if not Path(fp).exists():
            logger.error("File not found (%s): %s", label, fp)
            sys.exit(1)

    integrator   = VarietyDataIntegrator(cfg)
    integrated   = integrator.integrate(args.input, args.spornado)

    if integrated is None:
        logger.error("Integration failed.")
        sys.exit(1)

    integrator.save(integrated, output_path)
    integrator.print_summary(integrated)


if __name__ == "__main__":
    main()
