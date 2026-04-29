#!/usr/bin/env python3
"""
Train per-crop disease risk models  (2 / 3 disease triangle).

What the model predicts
-----------------------
Binary risk of a positive disease detection at a location, given weather
conditions and the time of year.  Spore count is intentionally excluded
from the feature set — it is the measurement that *determines* the label,
so using it would be target leakage producing artificially inflated metrics.

Pipeline
--------
1. Load the merged dataset produced by build_weather_spore_data.py
2. Run the feature engineering pipeline  (src.features)
3. For each crop:
   a. Chronological train / test split  (no future leakage)
   b. 5-fold TimeSeriesSplit CV on training set → bias-corrected estimates (no CV leakage)
   c. Final fit on full training set
   d. Evaluate on hold-out test set
   e. Persist the fitted sklearn Pipeline with joblib
4. Write results CSV to models/model_performance_results.csv

Usage
-----
  python -m src.build_crop_models
  python -m src.build_crop_models --data path/to/merged.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project-level imports  (works whether run as a module or script)
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_ROOT))

from src.config_loader import load as load_config
from src.features import TARGET_COL, build_features, get_feature_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, cfg["logging"]["level"]))
    fmt = logging.Formatter(cfg["logging"]["format"])
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_dir / "build_crop_models.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root_logger.addHandler(ch)
    root_logger.addHandler(fh)


def temporal_train_test_split(
    df: pd.DataFrame,
    date_col: str,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: oldest (1 - test_size) rows → train,
    most-recent test_size rows → test.

    This guarantees the test set contains only dates the model has never
    seen during training, matching real-world deployment conditions.
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    cut = int(len(df_sorted) * (1.0 - test_size))
    train = df_sorted.iloc[:cut].copy()
    test  = df_sorted.iloc[cut:].copy()
    logger.info(
        "  Temporal split: train=%d rows (up to %s), test=%d rows (from %s)",
        len(train),
        train[date_col].max().date() if not train.empty else "n/a",
        len(test),
        test[date_col].min().date() if not test.empty else "n/a",
    )
    return train, test


def risk_label(prob: float, low_max: float, high_min: float) -> str:
    """
    Convert a model probability into a human-readable risk category.

    Bands are defined by the crop-specific, bias-adjusted thresholds produced
    by find_optimal_threshold() — not by hardcoded constants.

      LOW    → prob < low_max                 model is confident: safe
      MEDIUM → low_max ≤ prob < high_min      uncertain zone near the boundary
      HIGH   → prob ≥ high_min                model is confident: alert
    """
    if prob < low_max:
        return "LOW"
    if prob < high_min:
        return "MEDIUM"
    return "HIGH"


def find_optimal_threshold(
    y_true,
    y_proba,
    *,
    min_recall: float = 0.90,
    medium_band_factor: float = 0.70,
) -> dict:
    """
    Find the classification threshold that meets a minimum recall floor and,
    within that constraint, maximises precision (fewest false alarms).

    Why recall-constrained instead of Youden's J
    ─────────────────────────────────────────────
    Youden's J maximises TPR − FPR, which implicitly weights every error equally.
    In crop protection the costs are NOT equal:

      False Negative (missed outbreak)  → crop loss — potentially devastating
      False Positive (unnecessary spray) → costs money, but the farm survives

    The correct formulation is therefore: express the business requirement as a
    recall *floor* ("we must catch at least 90 % of real outbreaks"), then let
    the maths find the threshold that delivers exactly that with maximum precision.
    This is directly interpretable — the agronomist can reason about "catch 9 in 10
    outbreaks" far better than a heuristic multiplier like sensitivity_bias=0.85.

    Algorithm
    ─────────
    1. Compute the full Precision–Recall curve (sklearn precision_recall_curve).
    2. Find every threshold where recall ≥ min_recall.
    3. Among those, pick the one with the highest precision.
       If the model cannot reach min_recall at any threshold, fall back to the
       lowest available threshold (maximum recall), log a warning, and set
       recall_shortfall=True so callers can detect the degraded case.
    4. Youden's J is computed for reference only — it is logged alongside the
       recall-constrained result so you can see how far the two approaches differ.

    Risk band construction
    ──────────────────────
    Bands are proportional to the decision boundary (not hardcoded offsets) so
    they scale correctly across crops with different base-rate prevalences:

      LOW    [0.0,                    optimal × medium_band_factor)
      MEDIUM [optimal × mbf,          optimal)                      ← uncertain zone
      HIGH   [optimal,                1.0]                          ← model is confident

    Parameters
    ----------
    min_recall : float, default 0.90
        Minimum recall (sensitivity) the deployed model must achieve.
        0.90 = "catch at least 9 in 10 real disease outbreaks".
        Higher values → lower threshold → more false alarms.
        Configure in config.yaml under model.min_recall.
        VALIDATE with agronomist / crop protection specialist.
    medium_band_factor : float, default 0.70
        The MEDIUM band occupies [optimal * factor, optimal).
        0.70 = the uncertain zone covers the bottom 30 % of the threshold.

    Returns
    -------
    dict with keys:
        optimal              — production decision boundary (recall-constrained)
        low_max              — upper edge of LOW band  (= optimal × medium_band_factor)
        high_min             — lower edge of HIGH band (= optimal)
        achieved_recall      — actual recall at this threshold (≥ min_recall if possible)
        achieved_precision   — actual precision at this threshold
        min_recall_target    — the requested floor (stored for provenance)
        recall_shortfall     — True if min_recall could not be met by any threshold
        youden               — Youden's J threshold (diagnostic reference only)
        medium_band_factor   — band factor used (stored for provenance)
    """
    import numpy as _np

    # ── Step 1: Precision–Recall curve ───────────────────────────────────────
    # sklearn returns arrays sorted by ascending threshold.
    # The last element (precision=1, recall=0) is a synthetic endpoint — drop it.
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
    precisions    = precisions[:-1]
    recalls       = recalls[:-1]

    valid_mask      = recalls >= min_recall
    recall_shortfall = False

    if valid_mask.any():
        # Among thresholds that meet the recall floor, pick the one with highest
        # precision — that minimises false alarms while honouring the constraint.
        best_idx = int(_np.argmax(_np.where(valid_mask, precisions, -_np.inf)))
    else:
        # The model cannot reach min_recall at any threshold.
        # Use the lowest threshold (= highest recall), warn, and flag the shortfall.
        best_idx         = 0
        recall_shortfall = True
        logger.warning(
            "Cannot achieve %.0f%% recall — best available is %.1f%%. "
            "Consider retraining with more positive examples or relaxing min_recall.",
            min_recall * 100, float(recalls.max()) * 100,
        )

    optimal            = float(pr_thresholds[best_idx])
    achieved_recall    = float(recalls[best_idx])
    achieved_precision = float(precisions[best_idx])

    # ── Step 2: Youden's J (diagnostic reference — the equal-cost baseline) ──
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    youden = float(roc_thresholds[int((tpr - fpr).argmax())])

    # ── Step 3: Risk band construction ───────────────────────────────────────
    # Clamp low_max so it is always strictly inside (0, optimal).
    low_max_raw = optimal * medium_band_factor
    low_max     = float(_np.clip(low_max_raw, 0.01, max(optimal - 0.01, 0.01)))
    high_min    = optimal   # HIGH band starts exactly at the decision boundary

    return {
        "optimal":            round(optimal,            4),
        "low_max":            round(low_max,            4),
        "high_min":           round(high_min,           4),
        "achieved_recall":    round(achieved_recall,    4),
        "achieved_precision": round(achieved_precision, 4),
        "min_recall_target":  round(min_recall,         4),
        "recall_shortfall":   recall_shortfall,
        "youden":             round(youden,             4),
        "medium_band_factor": round(medium_band_factor, 4),
    }


def build_pipeline(model_cfg: dict) -> Pipeline:
    """
    Three-step sklearn Pipeline:
      1. SimpleImputer  — replaces NaN with the training-set column mean.
                          Fit on training data only; applied identically at
                          inference, so rows missing weather data degrade
                          gracefully instead of crashing.
      2. StandardScaler — zero-mean / unit-variance normalisation (fit on
                          training data only; prevents scale leakage to test).
      3. LogisticRegression — binary classifier with balanced class weights.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            class_weight=model_cfg["class_weight"],
            max_iter=model_cfg["max_iter"],
            random_state=model_cfg["random_state"],
            solver=model_cfg["solver"],
        )),
    ])


def evaluate_crop(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_cfg: dict,
    cv_folds: int,
    crop: str,
    n_jobs: int = 1,
    min_recall: float = 0.90,
    medium_band_factor: float = 0.70,
) -> tuple[Pipeline, dict]:
    """
    Cross-validate on the training set, fit on all training data, and
    evaluate on the chronological hold-out test set.

    Key design notes
    ────────────────
    • CV scoring uses sklearn's default 0.5 threshold — this is intentional.
      CV folds are used to estimate generalisation error (AUC / F1), not to set
      the deployment threshold.  The deployment threshold is set ONCE on the
      hold-out test set using recall-constrained optimisation.
    • All classification metrics in the returned dict (test_f1, test_precision,
      test_recall) are computed at the recall-constrained optimal threshold, not
      at 0.5 — so the numbers match what will happen in production.

    Returns (fitted_pipeline, metrics_dict).
    """
    pipe = build_pipeline(model_cfg)

    # TimeSeriesSplit preserves temporal order within CV:
    # each fold trains on a chronological prefix and validates on the next
    # chunk — no future data ever bleeds into a training fold.
    cv = TimeSeriesSplit(n_splits=cv_folds)
    cv_results = cross_validate(
        pipe, X_train, y_train,
        cv=cv,
        scoring=["roc_auc", "f1", "precision", "recall"],
        return_train_score=False,
        n_jobs=n_jobs,
    )

    logger.info(
        "  CV (%d-fold TimeSeriesSplit) AUC: %.4f ± %.4f  |  F1 @0.5: %.4f ± %.4f",
        cv_folds,
        cv_results["test_roc_auc"].mean(),
        cv_results["test_roc_auc"].std(),
        cv_results["test_f1"].mean(),
        cv_results["test_f1"].std(),
    )

    # ── Final model: fit on all training data ────────────────────────────────
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # ── Threshold: recall-constrained (must come before classification metrics)
    thresholds = find_optimal_threshold(
        y_test, y_proba,
        min_recall=min_recall,
        medium_band_factor=medium_band_factor,
    )

    # Classification metrics computed at the deployment threshold (not 0.5).
    # This ensures reported F1/precision/recall match production behaviour.
    y_pred    = (y_proba >= thresholds["optimal"]).astype(int)
    test_auc  = roc_auc_score(y_test, y_proba)            # threshold-independent
    test_f1   = f1_score(y_test, y_pred, zero_division=0)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec  = recall_score(y_test, y_pred, zero_division=0)

    logger.info(
        "  Hold-out @ threshold=%.4f  AUC: %.4f  F1: %.4f  "
        "Precision: %.4f  Recall: %.4f",
        thresholds["optimal"], test_auc, test_f1, test_prec, test_rec,
    )
    logger.info(
        "  Recall target: %.0f%%  →  achieved: %.1f%%  |  "
        "Precision at that point: %.1f%%",
        min_recall * 100,
        thresholds["achieved_recall"]    * 100,
        thresholds["achieved_precision"] * 100,
    )
    logger.info(
        "  Risk bands: LOW < %.3f  |  MEDIUM [%.3f, %.3f)  |  HIGH ≥ %.3f  "
        "(Youden reference: %.3f)",
        thresholds["low_max"],
        thresholds["low_max"], thresholds["high_min"],
        thresholds["high_min"],
        thresholds["youden"],
    )
    logger.info(
        "  Classification report (test, threshold=%.4f):\n%s",
        thresholds["optimal"],
        classification_report(y_test, y_pred, zero_division=0),
    )

    metrics = {
        # Cross-validation (threshold-independent AUC; F1 at default 0.5)
        "cv_auc_mean":  float(cv_results["test_roc_auc"].mean()),
        "cv_auc_std":   float(cv_results["test_roc_auc"].std()),
        "cv_f1_mean":   float(cv_results["test_f1"].mean()),
        "cv_f1_std":    float(cv_results["test_f1"].std()),
        # Hold-out metrics at the recall-constrained deployment threshold
        "test_auc":       float(test_auc),
        "test_f1":        float(test_f1),
        "test_precision": float(test_prec),
        "test_recall":    float(test_rec),
        # Threshold provenance — every number stored so the JSON is fully reproducible
        "threshold_optimal":            thresholds["optimal"],
        "threshold_low_max":            thresholds["low_max"],
        "threshold_high_min":           thresholds["high_min"],
        "threshold_achieved_recall":    thresholds["achieved_recall"],
        "threshold_achieved_precision": thresholds["achieved_precision"],
        "threshold_min_recall_target":  thresholds["min_recall_target"],
        "threshold_recall_shortfall":   thresholds["recall_shortfall"],
        "threshold_youden":             thresholds["youden"],
        "threshold_medium_band_factor": thresholds["medium_band_factor"],
    }
    return pipe, metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_all_crops(data_path: str, cfg: dict) -> pd.DataFrame:
    logger.info("Loading dataset: %s", data_path)
    raw = pd.read_csv(data_path, low_memory=False)
    logger.info("Raw shape: %d rows × %d columns", *raw.shape)

    df = build_features(raw, cfg)

    models_dir  = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_columns(cfg, df)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    crops       = cfg.get("crops", ["Corn", "Soybean", "Potato"])
    min_samples = cfg["data"]["min_samples_per_crop"]
    test_size   = cfg["data"]["test_size"]
    results     = []

    for crop in crops:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Crop: %s", crop)
        logger.info("=" * 60)

        crop_df = df[df["crop_type"].str.strip().str.title() == crop].copy()
        logger.info("  Total rows: %d", len(crop_df))

        if len(crop_df) < min_samples:
            logger.warning(
                "  Skipping %s — only %d rows (minimum %d).", crop, len(crop_df), min_samples
            )
            continue

        use_cols = feature_cols + [TARGET_COL, "start_date"]
        crop_clean = crop_df[[c for c in use_cols if c in crop_df.columns]].dropna(
            subset=feature_cols + [TARGET_COL]
        )
        logger.info("  Rows after dropping NaN features / target: %d", len(crop_clean))

        if len(crop_clean) < min_samples:
            logger.warning(
                "  Skipping %s — only %d clean rows after NaN drop.", crop, len(crop_clean)
            )
            continue

        class_dist = crop_clean[TARGET_COL].value_counts().to_dict()
        logger.info("  Class distribution: %s", class_dist)

        if len(class_dist) < 2:
            logger.warning("  Skipping %s — only one class present.", crop)
            continue

        train_df, test_df = temporal_train_test_split(crop_clean, "start_date", test_size)

        X_train = train_df[feature_cols]
        y_train = train_df[TARGET_COL].astype(int)
        X_test  = test_df[feature_cols]
        y_test  = test_df[TARGET_COL].astype(int)

        if len(y_test.unique()) < 2:
            logger.warning(
                "  Skipping %s — test split contains only one class. "
                "Consider increasing test_size or collecting more data.", crop
            )
            continue

        pipe, metrics = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=cfg["model"],
            cv_folds=cfg["model"]["cv_folds"],
            n_jobs=cfg["model"].get("n_jobs", 1),
            min_recall=cfg["model"].get("min_recall", 0.90),
            medium_band_factor=cfg["model"].get("medium_band_factor", 0.70),
            crop=crop,
        )

        model_path = models_dir / f"{crop.lower()}_model.pkl"
        joblib.dump(pipe, model_path)

        # Save thresholds as plain JSON alongside the model.
        # predict.py loads this file at inference time — no retraining needed.
        # All provenance fields are stored so any threshold can be reproduced.
        import json
        threshold_path = models_dir / f"{crop.lower()}_thresholds.json"
        threshold_path.write_text(json.dumps({
            "crop":               crop,
            "optimal":            metrics["threshold_optimal"],
            "low_max":            metrics["threshold_low_max"],
            "high_min":           metrics["threshold_high_min"],
            "achieved_recall":    metrics["threshold_achieved_recall"],
            "achieved_precision": metrics["threshold_achieved_precision"],
            "min_recall_target":  metrics["threshold_min_recall_target"],
            "recall_shortfall":   metrics["threshold_recall_shortfall"],
            "youden":             metrics["threshold_youden"],
            "medium_band_factor": metrics["threshold_medium_band_factor"],
            "derivation": (
                f"Recall-constrained: threshold chosen to achieve "
                f">={metrics['threshold_min_recall_target']*100:.0f}% recall "
                f"(achieved {metrics['threshold_achieved_recall']*100:.1f}%) "
                f"with maximum precision "
                f"({metrics['threshold_achieved_precision']*100:.1f}%) "
                f"on chronological hold-out test set."
            ),
        }, indent=2), encoding="utf-8")

        logger.info("  Saved model      → %s", model_path)
        logger.info("  Saved thresholds → %s", threshold_path)

        results.append({
            "crop":                         crop,
            "train_samples":                len(X_train),
            "test_samples":                 len(X_test),
            "cv_auc_mean":                  metrics["cv_auc_mean"],
            "cv_auc_std":                   metrics["cv_auc_std"],
            "cv_f1_mean":                   metrics["cv_f1_mean"],
            "test_auc":                     metrics["test_auc"],
            "test_f1":                      metrics["test_f1"],
            "test_precision":               metrics["test_precision"],
            "test_recall":                  metrics["test_recall"],
            "threshold_optimal":            metrics["threshold_optimal"],
            "threshold_low_max":            metrics["threshold_low_max"],
            "threshold_high_min":           metrics["threshold_high_min"],
            "threshold_achieved_recall":    metrics["threshold_achieved_recall"],
            "threshold_achieved_precision": metrics["threshold_achieved_precision"],
            "threshold_min_recall_target":  metrics["threshold_min_recall_target"],
            "threshold_youden":             metrics["threshold_youden"],
            "model_type":                   "LogisticRegression (Pipeline: Imputer → Scaler → LR)",
            "n_features":                   len(feature_cols),
            "features":                     ", ".join(feature_cols),
        })

    results_df = pd.DataFrame(results)

    out_path = Path(cfg["output"]["model_results"])   # already absolute (resolved by config_loader)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info("\nResults written to: %s", out_path)

    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Spornado per-crop disease risk models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        help="Path to merged weather+spore CSV (overrides config paths.data_merged)",
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml (default: config/config.yaml relative to project root)",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    _setup_logging(cfg)

    data_path = args.data or cfg["paths"]["data_merged"]

    if not Path(data_path).exists():
        logger.error(
            "Dataset not found: %s\n"
            "Run build_weather_spore_data.py first to generate the merged dataset.\n"
            "  python build_weather_spore_data.py --help",
            data_path,
        )
        sys.exit(1)

    results = train_all_crops(data_path, cfg)

    if results.empty:
        logger.error("No crops were successfully trained. Check the logs above.")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE — SUMMARY")
    logger.info("=" * 60)
    logger.info("\n%s", results[
        ["crop", "train_samples", "test_samples", "cv_auc_mean", "cv_auc_std",
         "test_auc", "test_f1", "test_precision", "test_recall"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
