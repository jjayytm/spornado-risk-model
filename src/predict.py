"""
Inference module — load saved per-crop models and generate risk predictions.

The models are standard sklearn Pipelines (StandardScaler → LogisticRegression)
serialised with joblib.  This module handles loading, feature alignment, and
returning calibrated risk probabilities.

Usage
-----
Python API::

    from src.predict import Predictor

    p = Predictor()
    result = p.predict(
        crop="Corn",
        temperature_max_c=28.5,
        humidity_max_percent=83.0,
        precipitation_mm=2.1,
        temperature_min_c=14.0,
        temperature_mean_c=21.0,
        wind_speed_max_kmh=18.0,
        dew_point_min_c=12.5,
        month=8,
        day_of_year=220,
        week_of_year=32,
    )
    print(result)
    # {'crop': 'Corn', 'risk_probability': 0.73, 'risk_label': 'HIGH', ...}

CLI::

    python -m src.predict \\
        --crop Corn \\
        --temperature_max_c 28.5 \\
        --humidity_max_percent 83.0 \\
        --precipitation_mm 2.1 \\
        --month 8 --day_of_year 220 --week_of_year 32
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config_loader import load as load_config
from src.features import get_feature_columns
from src.build_crop_models import risk_label as _risk_label  # single source of truth

logger = logging.getLogger(__name__)

# Fallback thresholds — used ONLY when no threshold JSON exists on disk
# (e.g. first run before training, or unit tests).  Values are deliberately
# conservative (low thresholds = high sensitivity) to avoid silent false negatives.
# In production these are always overridden by the per-crop JSON written by
# build_crop_models.py.
_FALLBACK_THRESHOLDS = {"low_max": 0.25, "high_min": 0.40}


class Predictor:
    """
    Load saved crop models and produce risk predictions.

    Models are loaded lazily (on first predict call for a given crop) and
    cached in memory for subsequent calls.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg        = cfg or load_config()
        self._cache:      dict[str, Any]  = {}   # crop → fitted sklearn Pipeline
        self._thresholds: dict[str, dict] = {}   # crop → {low_max, high_min, optimal}

    def _load_model(self, crop: str) -> Any:
        if crop in self._cache:
            return self._cache[crop]

        models_dir = Path(self._cfg["paths"]["models_dir"])

        # Prefer variety-aware model if available
        for suffix in ("_model_variety.pkl", "_model.pkl"):
            path = models_dir / f"{crop.lower()}{suffix}"
            if path.exists():
                pipe = joblib.load(path)
                self._cache[crop] = pipe
                logger.info("Loaded model: %s", path)
                return pipe

        raise FileNotFoundError(
            f"No trained model found for crop '{crop}' in {models_dir}.\n"
            f"Run 'python -m src.build_crop_models' first."
        )

    def _load_thresholds(self, crop: str) -> dict:
        """
        Load the data-driven risk thresholds saved by build_crop_models.py.
        Falls back to conservative defaults only when no file exists (e.g. tests).
        """
        if crop in self._thresholds:
            return self._thresholds[crop]

        models_dir = Path(self._cfg["paths"]["models_dir"])
        for suffix in ("_thresholds.json",):
            path = models_dir / f"{crop.lower()}{suffix}"
            if path.exists():
                thresholds = json.loads(path.read_text(encoding="utf-8"))
                self._thresholds[crop] = thresholds
                logger.info(
                    "Loaded thresholds for %s: "
                    "LOW < %.3f | MEDIUM [%.3f, %.3f) | HIGH ≥ %.3f  "
                    "(recall target %.0f%%, achieved %.1f%%)",
                    crop,
                    thresholds["low_max"],
                    thresholds["low_max"],
                    thresholds["high_min"],
                    thresholds["high_min"],
                    thresholds.get("min_recall_target", 0) * 100,
                    thresholds.get("achieved_recall", 0)   * 100,
                )
                return thresholds

        logger.warning(
            "No threshold file found for '%s' — using fallback values %s. "
            "Run build_crop_models.py to generate data-driven thresholds.",
            crop, _FALLBACK_THRESHOLDS,
        )
        return _FALLBACK_THRESHOLDS

    def available_crops(self) -> list[str]:
        """Return crop names for which a saved model exists."""
        models_dir = Path(self._cfg["paths"]["models_dir"])
        found: list[str] = []
        for crop in self._cfg.get("crops", ["Corn", "Soybean", "Potato"]):
            for suffix in ("_model_variety.pkl", "_model.pkl"):
                if (models_dir / f"{crop.lower()}{suffix}").exists():
                    found.append(crop)
                    break
        return found

    def predict(self, crop: str, **weather_and_temporal: float) -> dict:
        """
        Predict disease risk for a single observation.

        Parameters
        ----------
        crop
            One of the trained crop names (e.g. 'Corn', 'Soybean', 'Potato').
        **weather_and_temporal
            Feature values keyed by column name.  Missing features default to NaN;
            the pipeline's StandardScaler will centre them using training statistics.

        Returns
        -------
        dict with keys:
            crop, risk_probability, risk_label, model_features_used, missing_features
        """
        pipe = self._load_model(crop)

        # Recover the ordered feature list from whichever fitted step exposes it.
        # "imputer" is checked first (current models); "scaler" is the fallback
        # for any legacy .pkl that lacks the imputer step.
        feature_cols: list[str] = []
        for step_name in ("imputer", "scaler"):
            step = pipe.named_steps.get(step_name)
            if step is not None and hasattr(step, "feature_names_in_"):
                feature_cols = list(step.feature_names_in_)
                break
        if not feature_cols:
            feature_cols = get_feature_columns(
                self._cfg, pd.DataFrame(columns=list(weather_and_temporal.keys()))
            )

        row = {col: weather_and_temporal.get(col, np.nan) for col in feature_cols}
        X   = pd.DataFrame([row], columns=feature_cols)

        missing = [col for col in feature_cols if pd.isna(X[col].iloc[0])]
        if missing:
            logger.warning(
                "Missing features for %s prediction — imputer will substitute "
                "the training-set mean for: %s", crop, missing
            )

        prob       = float(pipe.predict_proba(X)[0, 1])
        thresholds = self._load_thresholds(crop)
        label      = _risk_label(prob, thresholds["low_max"], thresholds["high_min"])

        return {
            "crop":                  crop,
            "risk_probability":      round(prob, 4),
            "risk_label":            label,
            "threshold_low_max":     thresholds["low_max"],
            "threshold_high_min":    thresholds["high_min"],
            "model_features_used":   len(feature_cols),
            "missing_features":      missing,
        }

    def predict_batch(self, df: pd.DataFrame, crop_col: str = "crop_type") -> pd.DataFrame:
        """
        Run predictions for a DataFrame that may contain multiple crops.

        The DataFrame must have a column named *crop_col* (default 'crop_type')
        and columns for all required weather / temporal features.

        Returns the original DataFrame with appended columns:
            risk_probability, risk_label
        """
        results = df.copy()
        results["risk_probability"] = np.nan
        results["risk_label"]       = ""

        for crop in results[crop_col].dropna().unique():
            try:
                pipe = self._load_model(crop)
            except FileNotFoundError:
                logger.warning("No model for crop '%s' — skipping.", crop)
                continue

            feature_cols = []
            for step_name in ("imputer", "scaler"):
                step = pipe.named_steps.get(step_name)
                if step is not None and hasattr(step, "feature_names_in_"):
                    feature_cols = list(step.feature_names_in_)
                    break
            if not feature_cols:
                feature_cols = get_feature_columns(self._cfg, results)

            mask = results[crop_col].str.strip().str.title() == crop
            X    = results.loc[mask, [c for c in feature_cols if c in results.columns]]
            # Align columns in case some are absent
            X = X.reindex(columns=feature_cols)

            thresholds = self._load_thresholds(crop)
            probs  = pipe.predict_proba(X)[:, 1]
            labels = [_risk_label(p, thresholds["low_max"], thresholds["high_min"]) for p in probs]

            results.loc[mask, "risk_probability"] = probs
            results.loc[mask, "risk_label"]        = labels

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict disease risk for a single Spornado observation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--crop",                   required=True,  help="Crop name (Corn, Soybean, Potato)")
    parser.add_argument("--temperature_max_c",      type=float, default=None)
    parser.add_argument("--temperature_min_c",      type=float, default=None)
    parser.add_argument("--temperature_mean_c",     type=float, default=None)
    parser.add_argument("--humidity_max_percent",   type=float, default=None)
    parser.add_argument("--precipitation_mm",       type=float, default=None)
    parser.add_argument("--wind_speed_max_kmh",     type=float, default=None)
    parser.add_argument("--dew_point_min_c",        type=float, default=None)
    parser.add_argument("--month",                  type=float, default=None)
    parser.add_argument("--day_of_year",            type=float, default=None)
    parser.add_argument("--week_of_year",           type=float, default=None)
    parser.add_argument("--config",                 default=None, help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg = load_config(Path(args.config) if args.config else None)
    p   = Predictor(cfg)

    features = {k: v for k, v in vars(args).items()
                if k not in ("crop", "config") and v is not None}

    result = p.predict(crop=args.crop, **features)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
