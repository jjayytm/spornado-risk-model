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
import csv
import datetime
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config_loader import load as load_config
from src.features import get_feature_columns, add_rolling_weather_features
from src.build_crop_models import _disease_key, risk_label as _risk_label  # single source of truth

logger = logging.getLogger(__name__)

# Fallback thresholds — intended for explicit test/dev mode only.
# Production should use per-crop threshold JSON artifacts written by
# build_crop_models.py and fail fast if they are missing.
_FALLBACK_THRESHOLDS = {"low_max": 0.25, "high_min": 0.40}


class Predictor:
    """
    Load saved crop models and produce risk predictions.

    Two-model lookup strategy
    -------------------------
    For each crop, up to three model files may exist (listed in priority order):

    1. ``<crop>_model_variety.pkl``  — variety-aware model (richer; if built)
       Threshold file: ``<crop>_thresholds.json``

    2. ``<crop>_model_weather.pkl``  — XGBoost trained on GPS rows with real
       location-specific weather data (full weather + temporal features).
       Threshold file: ``<crop>_weather_thresholds.json``

    3. ``<crop>_model.pkl``          — LogisticRegression trained on ALL rows
       using only temporal features (month, day_of_year, week_of_year).
       No weather imputation needed; safe universal fallback.
       Threshold file: ``<crop>_thresholds.json``

    Models are loaded lazily (on first predict call for a given crop) and
    cached in memory for subsequent calls.
    """

    # Priority-ordered (model_suffix, threshold_stem) pairs.
    # The first file that exists on disk wins.
    _MODEL_LOOKUP: list[tuple[str, str]] = [
        ("_model_variety.pkl",  "_thresholds"),
        ("_model_weather.pkl",  "_weather_thresholds"),
        ("_model.pkl",          "_thresholds"),
    ]

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg        = cfg or load_config()
        self._cache:      dict[str, Any]  = {}   # crop → fitted sklearn Pipeline
        self._thresholds: dict[str, dict] = {}   # crop → {low_max, high_min, optimal}
        self._model_thr:  dict[str, str]  = {}   # crop → threshold JSON stem used
        self._model_key:  dict[str, str]  = {}   # crop → loaded model suffix
        self._selection:  dict[str, dict] = self._load_model_selection()

    def _load_model_selection(self) -> dict[str, dict]:
        """
        Load optional per-crop model selection manifest.

        Manifest path is configured via ``prediction.model_selection_file``.
        Shape:
        {
          "Corn": {"model_suffix": "_model_weather.pkl", "threshold_stem": "_weather_thresholds"}
        }
        """
        pred_cfg = self._cfg.get("prediction", {})
        rel_path = pred_cfg.get("model_selection_file")
        if not rel_path:
            return {}

        path = Path(rel_path)
        if not path.is_absolute():
            path = Path(self._cfg["paths"]["models_dir"]).parent / path
        if not path.exists():
            logger.info("Model selection manifest not found at %s — using automatic routing.", path)
            return {}

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid model selection manifest format at {path}.")

        selection: dict[str, dict] = {}
        for crop, entry in data.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid entry for crop '{crop}' in {path}.")
            model_suffix = entry.get("model_suffix")
            threshold_stem = entry.get("threshold_stem")
            if not model_suffix or not threshold_stem:
                raise ValueError(
                    f"Manifest entry for '{crop}' must include model_suffix and threshold_stem."
                )
            selection[str(crop).strip().title()] = {
                "model_suffix": str(model_suffix),
                "threshold_stem": str(threshold_stem),
            }
        logger.info("Loaded model selection manifest from %s (%d crops).", path, len(selection))
        return selection

    @staticmethod
    def _is_missing_value(value: Any) -> bool:
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    def _weather_missing_columns(self, provided_features: dict[str, Any]) -> list[str]:
        weather_cols = list(self._cfg.get("features", {}).get("weather", []))
        return [
            col for col in weather_cols
            if col not in provided_features or self._is_missing_value(provided_features.get(col))
        ]

    def _load_model_for_suffix(self, crop: str, model_suffix: str, thr_stem: str) -> Any:
        cache_key = f"{crop}|{model_suffix}"
        if cache_key in self._cache:
            self._model_thr[crop] = thr_stem
            self._model_key[crop] = model_suffix
            return self._cache[cache_key]

        models_dir = Path(self._cfg["paths"]["models_dir"])
        path = models_dir / f"{crop.lower()}{model_suffix}"
        if not path.exists():
            raise FileNotFoundError(path)

        pipe = joblib.load(path)
        self._cache[cache_key] = pipe
        self._model_thr[crop] = thr_stem
        self._model_key[crop] = model_suffix
        logger.info("Loaded model: %s  (threshold stem: %s)", path, thr_stem)
        return pipe

    def _select_model(self, crop: str, provided_features: dict[str, Any] | None = None) -> Any:
        """
        Select the model variant for this request.

        Best-practice routing:
        - If a weather/variety model exists, weather features are mandatory.
        - Temporal fallback is used only when weather model artifacts are absent.
        """
        provided = provided_features or {}
        crop_t = str(crop).strip().title()

        # 1) Explicit per-crop champion selection (if provided).
        selected = self._selection.get(crop_t)
        if selected:
            model_suffix = selected["model_suffix"]
            threshold_stem = selected["threshold_stem"]
            if model_suffix in ("_model_weather.pkl", "_model_variety.pkl", "_model_baseline.pkl"):
                missing_weather = self._weather_missing_columns(provided)
                if missing_weather:
                    raise ValueError(
                        f"Missing required weather features for {crop_t}: {missing_weather}. "
                        "Selected champion model requires complete weather inputs."
                    )
            return self._load_model_for_suffix(crop_t, model_suffix, threshold_stem)

        # 2) Automatic routing fallback.
        models_dir = Path(self._cfg["paths"]["models_dir"])
        crop_l = crop_t.lower()
        has_variety = (models_dir / f"{crop_l}_model_variety.pkl").exists()
        has_weather = (models_dir / f"{crop_l}_model_weather.pkl").exists()

        if has_variety or has_weather:
            missing_weather = self._weather_missing_columns(provided)
            if missing_weather:
                raise ValueError(
                    f"Missing required weather features for {crop}: {missing_weather}. "
                    "Weather is a required disease-triangle pillar, so prediction "
                    "is blocked instead of silently imputing/falling back."
                )
            # Prefer richer model when weather is complete.
            if has_variety:
                return self._load_model_for_suffix(crop_t, "_model_variety.pkl", "_thresholds")
            return self._load_model_for_suffix(crop_t, "_model_weather.pkl", "_weather_thresholds")

        # Legacy temporal-only fallback if no weather model artifacts exist.
        return self._load_model_for_suffix(crop_t, "_model.pkl", "_thresholds")

    @staticmethod
    def _compute_rolling_from_history(
        current_weather: dict[str, float],
        history_df: pd.DataFrame,
        cfg: dict,
    ) -> dict[str, float]:
        """
        Compute rolling weather statistics for a single inference observation.

        At training time the rolling stats (e.g. ``temperature_max_c_roll3mean``)
        are derived from the N most-recent trap deployments at the same location.
        At inference time the caller can supply a small historical DataFrame
        containing recent rows at that location; this helper appends the current
        observation and re-runs the same rolling logic to produce identical
        feature values.

        Parameters
        ----------
        current_weather : dict
            Current-observation weather values keyed by column name
            (``temperature_max_c``, ``humidity_max_percent``, …).
        history_df : pd.DataFrame
            Recent historical rows at this location, sorted oldest → newest.
            Must contain the rolling base columns defined in config.
            If fewer rows than the largest window are available,
            ``min_periods=1`` ensures partial-window means are still computed.
        cfg : dict
            Loaded config (reads ``features.rolling_base_cols`` and
            ``features.rolling_windows``).

        Returns
        -------
        dict mapping rolling column names → computed float values,
        e.g. ``{"temperature_max_c_roll3mean": 25.1, ...}``.
        """
        feat_cfg  = cfg["features"]
        base_cols = feat_cfg["rolling_base_cols"]
        windows   = feat_cfg["rolling_windows"]
        _LOC      = "__inference__"

        # ── Build combined DataFrame: [history rows] + [current row] ──────────
        hist = history_df[[c for c in base_cols if c in history_df.columns]].copy()
        hist["location_ref"] = _LOC

        # Synthetic monotone dates — current row is always the last.
        n = len(hist)
        hist["start_date"] = pd.date_range(
            end=pd.Timestamp.now() - pd.Timedelta(days=1),
            periods=n,
            freq="D",
        )

        cur_row: dict[str, Any] = {col: current_weather.get(col, np.nan) for col in base_cols}
        cur_row["location_ref"] = _LOC
        cur_row["start_date"]   = pd.Timestamp.now()

        combined = pd.concat(
            [hist, pd.DataFrame([cur_row])], ignore_index=True
        ).sort_values("start_date").reset_index(drop=True)

        combined = add_rolling_weather_features(
            combined,
            base_cols  = base_cols,
            windows    = windows,
            group_col  = "location_ref",
            date_col   = "start_date",
        )

        # ── Extract rolling values for the current (last) row ─────────────────
        last   = combined.iloc[-1]
        result: dict[str, float] = {}
        for col in base_cols:
            for w in windows:
                for stat in ("mean", "max"):
                    key       = f"{col}_roll{w}{stat}"
                    val       = last.get(key, np.nan)
                    result[key] = float(val) if pd.notna(val) else np.nan
        return result

    def _load_model(self, crop: str) -> Any:
        # Backward-compatible default path for older call sites.
        return self._select_model(crop, provided_features=None)

    def _load_thresholds(self, crop: str, ensure_model_loaded: bool = True) -> dict:
        """
        Load the data-driven risk thresholds saved by build_crop_models.py.

        The correct threshold file is determined by which model was loaded:
        - weather model  → ``<crop>_weather_thresholds.json``
        - all others     → ``<crop>_thresholds.json``

        Falls back to conservative defaults only when no file exists (e.g. tests).
        """
        if crop in self._thresholds:
            return self._thresholds[crop]

        # Ensure the model is loaded so _model_thr is populated.
        # For strict weather-gated inference, callers should pass
        # ensure_model_loaded=False after _select_model(...) has already run.
        if ensure_model_loaded and crop not in self._model_thr:
            self._load_model(crop)

        models_dir = Path(self._cfg["paths"]["models_dir"])
        thr_stem   = self._model_thr.get(crop, "_thresholds")
        path       = models_dir / f"{crop.lower()}{thr_stem}.json"

        if path.exists():
            thresholds = json.loads(path.read_text(encoding="utf-8"))
            self._thresholds[crop] = thresholds
            logger.info(
                "Loaded thresholds for %s (%s): "
                "LOW < %.3f | MEDIUM [%.3f, %.3f) | HIGH ≥ %.3f  "
                "(recall target %.0f%%, achieved %.1f%%)",
                crop,
                thresholds.get("model", "unknown"),
                thresholds["low_max"],
                thresholds["low_max"],
                thresholds["high_min"],
                thresholds["high_min"],
                thresholds.get("min_recall_target", 0) * 100,
                thresholds.get("achieved_recall", 0)   * 100,
            )
            return thresholds

        logger.warning(
            "No threshold file found for '%s' at '%s'.",
            crop, path,
        )
        allow_fallback = bool(
            self._cfg.get("prediction", {}).get("allow_threshold_fallback", False)
        )
        if not allow_fallback:
            raise FileNotFoundError(
                f"Threshold file is required for crop '{crop}' but was not found: {path}. "
                "Retrain models to regenerate threshold JSON artifacts, or set "
                "prediction.allow_threshold_fallback=true for explicit dev/test fallback."
            )
        logger.warning(
            "Using fallback thresholds for '%s' because "
            "prediction.allow_threshold_fallback=true.",
            crop,
        )
        return _FALLBACK_THRESHOLDS

    def _log_prediction(self, result: dict, features: dict) -> None:
        """
        Append a prediction record to the prediction log CSV.

        Each row contains: timestamp, all result fields, all input feature values.
        The log is used for monitoring, drift detection, and audit trail.
        Enabled/disabled via ``prediction_logging.enabled`` in config.yaml.
        """
        log_cfg = self._cfg.get("prediction_logging", {})
        if not log_cfg.get("enabled", True):
            return

        log_file = log_cfg.get(
            "log_file",
            str(Path(self._cfg["paths"]["logs_dir"]) / "predictions.csv"),
        )
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            **result,
            **{k: v for k, v in features.items() if not isinstance(v, list)},
        }

        write_header = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()),
                                    extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def available_crops(self) -> list[str]:
        """Return crop names for which at least one saved model exists."""
        models_dir = Path(self._cfg["paths"]["models_dir"])
        found: list[str] = []
        for crop in self._cfg.get("crops", ["Corn", "Soybean", "Potato"]):
            for model_suffix, _ in self._MODEL_LOOKUP:
                if (models_dir / f"{crop.lower()}{model_suffix}").exists():
                    found.append(crop)
                    break
        return found

    def predict(
        self,
        crop:       str,
        disease:    str | None           = None,
        history_df: pd.DataFrame | None  = None,
        **weather_and_temporal: float,
    ) -> dict:
        """
        Predict disease risk for a single observation.

        Parameters
        ----------
        crop
            One of the trained crop names (e.g. 'Corn', 'Soybean', 'Wheat').
        disease
            Disease name exactly as it appears in the 'Test' column
            (e.g. 'Gray Leaf Spot', 'Tar Spot', 'Frogeye Leaf Spot').
            When provided, the corresponding disease dummy is set to 1 and all
            others to 0 — this activates the disease-specific part of the model
            and is STRONGLY recommended for accurate predictions.
            When omitted, all disease dummies default to NaN and are imputed to
            their training-set medians — effectively predicting average-disease
            risk, which is less accurate.
        history_df : pd.DataFrame, optional
            Recent trap readings at this location (sorted oldest → newest),
            containing the rolling base columns
            (``temperature_max_c``, ``humidity_max_percent``,
            ``precipitation_mm``, ``dew_point_min_c``).
            When supplied, rolling weather statistics
            (``temperature_max_c_roll3mean``, etc.) are computed from this
            history instead of being imputed to training-set medians.
            Providing at least ``max(rolling_windows)`` = 7 rows gives the most
            accurate rolling values; fewer rows degrade gracefully via
            ``min_periods=1``.
            If omitted, rolling features are imputed — acceptable for quick
            one-off predictions, but less accurate for production inference.
        **weather_and_temporal
            Feature values keyed by column name.  Missing features default to NaN
            and are filled by the pipeline's median imputer.

        Returns
        -------
        dict with keys:
            crop, disease, risk_probability, risk_label,
            model_features_used, missing_features
        """
        pipe = self._select_model(crop, provided_features=weather_and_temporal)

        # Recover the ordered feature list from the fitted imputer/scaler step.
        # _CalibratedWrapper wraps the base Pipeline — unwrap it first so
        # named_steps is accessible regardless of whether calibration was applied.
        inner = pipe
        if hasattr(pipe, "calibrated_classifiers_"):
            try:
                inner = pipe.calibrated_classifiers_[0].estimator
            except (IndexError, AttributeError):
                pass

        feature_cols: list[str] = []
        for step_name in ("imputer", "scaler"):
            step = inner.named_steps.get(step_name)
            if step is not None and hasattr(step, "feature_names_in_"):
                feature_cols = list(step.feature_names_in_)
                break
        if not feature_cols:
            feature_cols = get_feature_columns(
                self._cfg, pd.DataFrame(columns=list(weather_and_temporal.keys()))
            )

        # ── Build feature row ─────────────────────────────────────────────────
        row: dict[str, float] = {col: np.nan for col in feature_cols}

        # Fill provided numeric / temporal features
        for k, v in weather_and_temporal.items():
            if k in row:
                row[k] = float(v)

        # ── Rolling features from historical trap data (optional) ─────────────
        # When the caller supplies recent readings at this location we can
        # compute the same rolling statistics the training pipeline produced,
        # giving the model its full feature vector instead of falling back to
        # imputed training-set medians.
        if history_df is not None and not history_df.empty:
            rolling_vals = self._compute_rolling_from_history(
                dict(weather_and_temporal), history_df, self._cfg
            )
            for k, v in rolling_vals.items():
                if k in row:            # only overwrite features the model knows
                    row[k] = v
            logger.info(
                "Rolling features computed from %d historical rows for %s.",
                len(history_df), crop,
            )

        # ── Disease dummies ───────────────────────────────────────────────────
        disease_dummy_cols = [c for c in feature_cols if c.startswith("disease_")]
        if disease_dummy_cols:
            if disease is not None:
                dk = _disease_key(disease)
                for col in disease_dummy_cols:
                    row[col] = 1.0 if col == dk else 0.0
                if dk not in disease_dummy_cols:
                    logger.warning(
                        "Disease '%s' (key='%s') not seen during training for %s. "
                        "Known diseases: %s. All dummies set to 0.",
                        disease, dk, crop,
                        [c.replace("disease_", "") for c in disease_dummy_cols],
                    )
                    for col in disease_dummy_cols:
                        row[col] = 0.0
            else:
                logger.warning(
                    "No disease provided for %s prediction. Disease dummies will "
                    "be imputed to training medians (less accurate). "
                    "Pass disease='<name>' for disease-specific risk.", crop,
                )

        X       = pd.DataFrame([row], columns=feature_cols)
        missing = [col for col in feature_cols
                   if not col.startswith("disease_") and pd.isna(X[col].iloc[0])]
        if missing:
            logger.warning(
                "Missing features for %s / %s prediction -- "
                "imputer will substitute training medians for: %s",
                crop, disease or "unknown disease", missing,
            )

        prob       = float(pipe.predict_proba(X)[0, 1])
        thresholds = self._load_thresholds(crop, ensure_model_loaded=False)
        label      = _risk_label(prob, thresholds["low_max"], thresholds["high_min"])

        result = {
            "crop":                  crop,
            "disease":               disease or "unknown",
            "risk_probability":      round(prob, 4),
            "risk_label":            label,
            "threshold_low_max":     thresholds["low_max"],
            "threshold_high_min":    thresholds["high_min"],
            "model_features_used":   len(feature_cols),
            "missing_features":      missing,
        }
        self._log_prediction(result, {"disease": disease, **weather_and_temporal})
        return result

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
                weather_cols = list(self._cfg.get("features", {}).get("weather", []))
                crop_mask = results[crop_col].str.strip().str.title() == crop
                models_dir = Path(self._cfg["paths"]["models_dir"])
                crop_l = str(crop).lower()
                has_variety = (models_dir / f"{crop_l}_model_variety.pkl").exists()
                has_weather = (models_dir / f"{crop_l}_model_weather.pkl").exists()
                if has_variety or has_weather:
                    missing_cols = [c for c in weather_cols if c not in results.columns]
                    if missing_cols:
                        raise ValueError(
                            f"Batch prediction for {crop} is missing required weather columns: {missing_cols}."
                        )
                    missing_rows = int(results.loc[crop_mask, weather_cols].isna().any(axis=1).sum())
                    if missing_rows:
                        raise ValueError(
                            f"Batch prediction for {crop} has {missing_rows} row(s) with missing weather features. "
                            "Weather is required for triangle-consistent inference."
                        )
                # Build sample_features for routing only — pick the FIRST fully-
                # populated row so _select_model sees complete weather even when
                # row-0 happens to be sparse.  The actual prediction values come
                # from the full aligned DataFrame X, not from sample_features.
                sample_features: dict[str, Any] = {}
                if weather_cols:
                    w_in_df = [c for c in weather_cols if c in results.columns]
                    if w_in_df:
                        complete_rows = results.loc[crop_mask, w_in_df].dropna(how="any")
                        if not complete_rows.empty:
                            sample_features = complete_rows.iloc[0].to_dict()
                pipe = self._select_model(crop, provided_features=sample_features)
            except FileNotFoundError:
                logger.warning("No model for crop '%s' — skipping.", crop)
                continue

            feature_cols = []
            inner_b = pipe
            if hasattr(pipe, "calibrated_classifiers_"):
                try:
                    inner_b = pipe.calibrated_classifiers_[0].estimator
                except (IndexError, AttributeError):
                    pass
            for step_name in ("imputer", "scaler"):
                step = inner_b.named_steps.get(step_name)
                if step is not None and hasattr(step, "feature_names_in_"):
                    feature_cols = list(step.feature_names_in_)
                    break
            if not feature_cols:
                feature_cols = get_feature_columns(self._cfg, results)

            mask = results[crop_col].str.strip().str.title() == crop

            # ── Disease dummies for batch rows ────────────────────────────────
            disease_dummy_cols = [c for c in feature_cols if c.startswith("disease_")]
            if disease_dummy_cols:
                # Initialise all disease dummies to 0 for this crop's rows
                for col in disease_dummy_cols:
                    if col not in results.columns:
                        results[col] = 0.0
                    results.loc[mask, col] = 0.0
                # Set the correct dummy to 1 for each row based on 'test' column
                if "test" in results.columns:
                    for idx in results.index[mask]:
                        test_val = results.at[idx, "test"]
                        if pd.notna(test_val):
                            dk = _disease_key(str(test_val))
                            if dk in disease_dummy_cols:
                                results.at[idx, dk] = 1.0

            X    = results.loc[mask, [c for c in feature_cols if c in results.columns]]
            # Align columns in case some are absent
            X = X.reindex(columns=feature_cols)

            thresholds = self._load_thresholds(crop, ensure_model_loaded=False)
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
    parser.add_argument("--crop",                   required=True,  help="Crop name (Corn, Soybean, Wheat)")
    parser.add_argument("--disease",                default=None,   help="Disease name, e.g. 'Gray Leaf Spot', 'Tar Spot'")
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
    parser.add_argument("--config",      default=None, help="Path to config.yaml")
    parser.add_argument(
        "--history_csv",
        default=None,
        metavar="PATH",
        help=(
            "CSV with recent trap readings at this location "
            "(columns: temperature_max_c, humidity_max_percent, "
            "precipitation_mm, dew_point_min_c — oldest row first). "
            "Used to compute rolling weather statistics so the model "
            "receives its full feature vector rather than imputed medians. "
            "Providing at least 7 rows is recommended (matches the largest "
            "rolling window). If omitted, rolling features are imputed."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg = load_config(Path(args.config) if args.config else None)
    p   = Predictor(cfg)

    # Load optional location history for rolling-feature computation
    history_df: pd.DataFrame | None = None
    if args.history_csv:
        history_path = Path(args.history_csv)
        if not history_path.exists():
            logger.error("--history_csv file not found: %s", history_path)
            sys.exit(1)
        history_df = pd.read_csv(history_path)
        logger.info(
            "Loaded %d historical rows from %s for rolling-feature computation.",
            len(history_df), history_path,
        )

    _skip = {"crop", "disease", "config", "history_csv"}
    features = {k: v for k, v in vars(args).items() if k not in _skip and v is not None}

    result = p.predict(crop=args.crop, disease=args.disease, history_df=history_df, **features)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
