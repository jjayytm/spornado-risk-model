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
import re
import sys
from pathlib import Path

import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve as sk_calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as _LRCalibrator
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
from src.features import (
    TARGET_COL,
    build_features,
    get_feature_columns,
    get_temporal_feature_columns,
)

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
    Find the recall-constrained decision threshold and risk bands.

    HIGH boundary — recall-constrained
    ───────────────────────────────────
    In crop protection costs are asymmetric:
      False Negative (missed outbreak)   → crop loss — potentially catastrophic
      False Positive (unnecessary spray) → costs money, but farm survives

    The threshold is set as a recall floor: "catch at least min_recall × 100 %
    of real outbreaks."  Among all thresholds that meet that floor, pick the one
    with the highest precision — fewest false alarms subject to the constraint.

    MEDIUM band — proportional factor
    ──────────────────────────────────
    tau_low = tau* × medium_band_factor  (default 0.70)

    This gives a MEDIUM band of consistent width relative to the decision
    boundary across all crops and dataset sizes.  The factor is configurable
    via config.yaml (model.medium_band_factor).

      LOW    → prob < tau_low            safe, no action
      MEDIUM → tau_low ≤ prob < tau*     uncertain — scout and monitor
      HIGH   → prob ≥ tau*               act now

    Algorithm
    ─────────
    1. Precision–Recall curve from calibrated probabilities.
    2. Recall-constrained optimal threshold (HIGH boundary = tau*).
    3. tau_low = tau* × medium_band_factor, clamped to (0.01, tau* − 0.01).
    4. Youden's J computed for diagnostic reference only.

    Returns
    -------
    dict with keys:
        optimal            — production decision boundary (tau*)
        low_max            — upper edge of LOW band (tau_low)
        high_min           — lower edge of HIGH band (= optimal)
        achieved_recall    — actual recall at tau*
        achieved_precision — actual precision at tau*
        min_recall_target  — requested floor (provenance)
        recall_shortfall   — True if min_recall could not be met
        youden             — Youden's J (diagnostic reference only)
        base_rate          — positive class fraction in y_true
        medium_band_factor — factor used to derive tau_low (provenance)
    """
    import numpy as _np

    # ── Step 1: Precision–Recall curve ───────────────────────────────────────
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
    precisions = precisions[:-1]   # drop trailing synthetic (1.0, 0.0) point
    recalls    = recalls[:-1]

    # ── Step 2: HIGH boundary — recall-constrained ───────────────────────────
    valid_mask       = recalls >= min_recall
    recall_shortfall = False

    if valid_mask.any():
        best_idx = int(_np.argmax(_np.where(valid_mask, precisions, -_np.inf)))
    else:
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

    # ── Step 3: Youden's J (diagnostic reference — equal-cost baseline) ──────
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    youden = float(roc_thresholds[int((tpr - fpr).argmax())])

    # ── Step 4: MEDIUM band — proportional factor ─────────────────────────────
    base_rate = float(_np.mean(y_true))
    low_max   = float(_np.clip(
        optimal * medium_band_factor,
        0.01, max(optimal - 0.01, 0.01),
    ))
    high_min  = optimal

    logger.info(
        "  MEDIUM band: low_max=%.4f  (tau*=%.4f × factor=%.2f)  base_rate=%.1f%%",
        low_max, optimal, medium_band_factor, base_rate * 100,
    )

    return {
        "optimal":            round(optimal,            4),
        "low_max":            round(low_max,            4),
        "high_min":           round(high_min,           4),
        "achieved_recall":    round(achieved_recall,    4),
        "achieved_precision": round(achieved_precision, 4),
        "min_recall_target":  round(min_recall,         4),
        "recall_shortfall":   recall_shortfall,
        "youden":             round(youden,             4),
        "base_rate":          round(base_rate,          4),
        "medium_band_factor": round(medium_band_factor, 4),
    }


def build_pipeline(model_cfg: dict, pos_weight: float = 1.0) -> Pipeline:
    """
    Build a sklearn Pipeline based on ``model_cfg["type"]``.

    Supported types
    ---------------
    XGBoost (default, recommended)
        Pipeline: SimpleImputer → XGBClassifier
        XGBoost handles missing values natively and captures nonlinear
        weather interactions that logistic regression cannot.
        ``pos_weight`` is passed as ``scale_pos_weight`` to correct for
        class imbalance (equivalent to class_weight='balanced' in LR).

    LogisticRegression
        Pipeline: SimpleImputer → StandardScaler → LogisticRegression
        Fully interpretable coefficients; good baseline.

    Both pipelines expose an "imputer" step first, so ``predict.py`` can
    recover feature names from ``pipe.named_steps['imputer'].feature_names_in_``
    regardless of which model type was trained.
    """
    model_type = model_cfg.get("type", "LogisticRegression")

    if model_type == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for model.type=XGBoost. "
                "Run: pip install xgboost"
            ) from exc

        xgb = model_cfg.get("xgboost", {})
        clf = XGBClassifier(
            scale_pos_weight  = pos_weight,
            n_estimators      = xgb.get("n_estimators",     300),
            max_depth         = xgb.get("max_depth",          5),
            learning_rate     = xgb.get("learning_rate",   0.05),
            subsample         = xgb.get("subsample",         0.8),
            colsample_bytree  = xgb.get("colsample_bytree",  0.8),
            min_child_weight  = xgb.get("min_child_weight",    5),
            eval_metric       = xgb.get("eval_metric",  "logloss"),
            random_state      = xgb.get("random_state",
                                        model_cfg.get("random_state", 42)),
            n_jobs            = model_cfg.get("n_jobs", 1),
            verbosity         = 0,
        )
        # XGBoost does not require StandardScaler — tree splits are scale-invariant.
        # strategy="median" is more robust than "mean" for skewed weather distributions
        # (extreme precipitation events inflate the mean, distorting imputed values).
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     clf),
        ])

    # ── LogisticRegression (fallback / baseline) ─────────────────────────────
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            class_weight = model_cfg.get("class_weight", "balanced"),
            max_iter     = model_cfg.get("max_iter",     1000),
            random_state = model_cfg.get("random_state",   42),
            solver       = model_cfg.get("solver",      "lbfgs"),
        )),
    ])


def evaluate_crop(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    model_cfg:         dict,
    cv_folds:          int,
    crop:              str,
    n_jobs:            int   = 1,
    min_recall:        float = 0.90,
    medium_band_factor: float = 0.70,
    calibration_cfg:   dict | None = None,
) -> tuple:
    """
    Cross-validate, fit, optionally calibrate, and evaluate a crop risk model.

    Training data split (leakage-safe)
    ──────────────────────────────────
    X_train is split chronologically into:
      X_model (oldest 80 %)  — used to fit the model/calibrator
      X_thr   (newest 20 %)  — used ONLY to tune thresholds

    This prevents threshold leakage from the final hold-out test set.
    Reported test metrics are measured at a threshold learned strictly from
    training-time data.

    When calibration is enabled, X_model is further split into:
      X_fit (oldest) — base model fit
      X_cal (newest) — probability calibrator fit

    Key design notes
    ────────────────
    • CV uses the default 0.5 threshold — CV estimates generalisation, not
      the deployment threshold.
    • Deployment thresholds are tuned on X_thr, never on X_test.
    • All hold-out metrics (test_f1, test_precision, test_recall) are computed
      on X_test at the pre-tuned deployment threshold — matching production
      behaviour with no test-time tuning.
    • Thresholds are computed on CALIBRATED probabilities so that low_max and
      high_min have honest probabilistic meaning after calibration.

    Returns
    ───────
    (fitted_pipeline, metrics_dict, y_proba_uncal)

    fitted_pipeline : the final model (CalibratedClassifierCV if calibrated,
                      plain Pipeline otherwise)
    metrics_dict    : performance and threshold provenance
    y_proba_uncal   : raw (pre-calibration) probabilities on X_test, or None
                      if calibration was skipped.  Used by save_calibration_curve.
    """
    # ── Class imbalance ratio (XGBoost scale_pos_weight) ─────────────────────
    neg        = int((y_train == 0).sum())
    pos        = int((y_train == 1).sum())
    pos_weight = neg / pos if pos > 0 else 1.0

    base_pipe = build_pipeline(model_cfg, pos_weight=pos_weight)

    # ── TimeSeriesSplit CV on full training set ───────────────────────────────
    cv         = TimeSeriesSplit(n_splits=cv_folds)
    cv_results = cross_validate(
        base_pipe, X_train, y_train,
        cv=cv,
        scoring=["roc_auc", "f1", "precision", "recall"],
        return_train_score=False,
        n_jobs=n_jobs,
    )
    logger.info(
        "  CV (%d-fold TimeSeriesSplit) AUC: %.4f ± %.4f  |  F1 @0.5: %.4f ± %.4f",
        cv_folds,
        cv_results["test_roc_auc"].mean(), cv_results["test_roc_auc"].std(),
        cv_results["test_f1"].mean(),      cv_results["test_f1"].std(),
    )

    # ── Threshold tuning split (chronological, train-only) ───────────────────
    # Keep the newest slice of training data for threshold tuning so the test
    # set remains a true untouched hold-out.
    thr_size = 0.20
    min_thr_n = 30
    cut_thr = int(len(X_train) * (1.0 - thr_size))
    if cut_thr <= 0 or (len(X_train) - cut_thr) < min_thr_n:
        cut_thr = max(1, len(X_train) - min_thr_n)
    X_model, X_thr = X_train.iloc[:cut_thr], X_train.iloc[cut_thr:]
    y_model, y_thr = y_train.iloc[:cut_thr], y_train.iloc[cut_thr:]

    if len(X_thr) < min_thr_n:
        logger.warning(
            "  Threshold tuning window is small (%d rows < %d). "
            "Thresholds may be noisy; consider more data.",
            len(X_thr), min_thr_n,
        )
    logger.info(
        "  Threshold tuning split: model=%d rows, threshold_tune=%d rows (newest %.0f%%)",
        len(X_model), len(X_thr), thr_size * 100,
    )

    # ── Calibration split (chronological, inside model-fit window) ───────────
    cal_cfg     = calibration_cfg or {}
    do_cal      = cal_cfg.get("enabled", False)
    y_proba_uncal: np.ndarray | None = None

    if do_cal:
        cal_size    = cal_cfg.get("cal_size", 0.20)
        min_cal_n   = cal_cfg.get("min_cal_samples", 30)
        cut         = int(len(X_model) * (1.0 - cal_size))
        X_fit, X_cal = X_model.iloc[:cut], X_model.iloc[cut:]
        y_fit, y_cal = y_model.iloc[:cut], y_model.iloc[cut:]

        if len(X_cal) < min_cal_n:
            logger.warning(
                "  Calibration skipped — only %d calibration samples (min %d).",
                len(X_cal), min_cal_n,
            )
            do_cal = False

    if not do_cal:
        X_fit, y_fit = X_model, y_model

    # ── Fit base pipeline on X_fit ────────────────────────────────────────────
    base_pipe.fit(X_fit, y_fit)

    # ── Calibrate (isotonic or sigmoid) ──────────────────────────────────────
    if do_cal:
        # Raw (uncalibrated) probabilities on the test set — for reliability diagram
        y_proba_uncal = base_pipe.predict_proba(X_test)[:, 1]

        n_cal  = len(X_cal)
        method = cal_cfg.get("method", "auto")
        if method == "auto":
            method = "isotonic" if n_cal >= 50 else "sigmoid"

        pipe = _calibrate_pipeline(base_pipe, X_cal, y_cal, method=method)
        logger.info(
            "  Calibrated with %s on %d samples.",
            method, n_cal,
        )
    else:
        pipe = base_pipe

    # ── Tune deployment threshold on train-only threshold window ──────────────
    y_thr_proba = pipe.predict_proba(X_thr)[:, 1]
    thresholds = find_optimal_threshold(
        y_thr, y_thr_proba,
        min_recall=min_recall,
        medium_band_factor=medium_band_factor,
    )

    # ── Evaluate on untouched hold-out test set ───────────────────────────────
    y_proba = pipe.predict_proba(X_test)[:, 1]   # calibrated if calibration ran
    y_pred    = (y_proba >= thresholds["optimal"]).astype(int)
    test_auc  = roc_auc_score(y_test, y_proba)
    test_f1   = f1_score(y_test,  y_pred, zero_division=0)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec  = recall_score(y_test,  y_pred, zero_division=0)

    logger.info(
        "  Hold-out (threshold pre-tuned on train-only window) @ %.4f  "
        "AUC: %.4f  F1: %.4f  "
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
        "  Risk bands: LOW < %.4f  |  MEDIUM [%.4f, %.4f)  |  HIGH ≥ %.4f  "
        "(Youden ref: %.4f)",
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

    calibration_info = {
        "calibrated":          do_cal,
        "calibration_method":  method if do_cal else None,
        "calibration_samples": int(len(X_cal)) if do_cal else 0,
        "threshold_tuning_samples": int(len(X_thr)),
    }

    metrics = {
        "cv_auc_mean":  float(cv_results["test_roc_auc"].mean()),
        "cv_auc_std":   float(cv_results["test_roc_auc"].std()),
        "cv_f1_mean":   float(cv_results["test_f1"].mean()),
        "cv_f1_std":    float(cv_results["test_f1"].std()),
        "test_auc":       float(test_auc),
        "test_f1":        float(test_f1),
        "test_precision": float(test_prec),
        "test_recall":    float(test_rec),
        # Threshold provenance
        "threshold_optimal":            thresholds["optimal"],
        "threshold_low_max":            thresholds["low_max"],
        "threshold_high_min":           thresholds["high_min"],
        "threshold_achieved_recall":    thresholds["achieved_recall"],
        "threshold_achieved_precision": thresholds["achieved_precision"],
        "threshold_min_recall_target":  thresholds["min_recall_target"],
        "threshold_recall_shortfall":   thresholds["recall_shortfall"],
        "threshold_youden":             thresholds["youden"],
        "threshold_base_rate":          thresholds["base_rate"],
        "threshold_medium_band_factor": thresholds["medium_band_factor"],
        **calibration_info,
    }
    return pipe, metrics, y_proba_uncal


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

class _CalibratedWrapper:
    """
    A thin, sklearn-version-agnostic wrapper that applies a fitted probability
    calibrator on top of a pre-fitted sklearn Pipeline.

    Motivation
    ──────────
    sklearn 1.8 removed ``CalibratedClassifierCV(cv='prefit')``.  Rather than
    depending on a specific sklearn version we own the calibration step directly:

      1. Fit the base Pipeline on X_fit.
      2. Generate raw probability scores on X_cal.
      3. Fit an IsotonicRegression or LogisticRegression (Platt scaling) on
         (scores, y_cal) — this is the calibrator.
      4. At inference: base_pipe.predict_proba → calibrator → calibrated prob.

    The wrapper intentionally exposes ``named_steps`` and a
    ``calibrated_classifiers_`` shim so that ``_get_inner_pipeline`` and
    ``predict.py`` can reach the inner Pipeline without any version-specific logic.
    """

    def __init__(self, base_pipeline: Pipeline, calibrator, method: str) -> None:
        self._base   = base_pipeline
        self._cal    = calibrator
        self._method = method

    # ── sklearn-compatible interface ─────────────────────────────────────────

    def predict_proba(self, X) -> np.ndarray:
        raw = self._base.predict_proba(X)[:, 1]
        if self._method == "sigmoid":
            # LogisticRegression calibrator
            cal_pos = self._cal.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            # IsotonicRegression calibrator
            cal_pos = self._cal.predict(raw)
        cal_pos = np.clip(cal_pos, 0.0, 1.0)
        return np.column_stack([1.0 - cal_pos, cal_pos])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ── Introspection shims ──────────────────────────────────────────────────

    @property
    def named_steps(self):
        """Passthrough so code that calls ``pipe.named_steps`` still works."""
        return self._base.named_steps

    @property
    def calibrated_classifiers_(self):
        """Compatibility shim so _get_inner_pipeline works uniformly."""
        class _Compat:
            def __init__(self, estimator):
                self.estimator = estimator
        return [_Compat(self._base)]


def _calibrate_pipeline(
    base_pipe: Pipeline,
    X_cal:     pd.DataFrame,
    y_cal:     pd.Series,
    method:    str = "isotonic",
) -> "_CalibratedWrapper":
    """
    Fit a probability calibrator on a held-out calibration set and return a
    ``_CalibratedWrapper`` that applies it at inference time.

    Parameters
    ----------
    base_pipe : already-fitted sklearn Pipeline
    X_cal     : calibration features (chronologically after the training set)
    y_cal     : calibration labels
    method    : "isotonic" or "sigmoid"
    """
    raw_scores = base_pipe.predict_proba(X_cal)[:, 1]

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_scores, y_cal)
    else:
        # Platt scaling (sigmoid): logistic regression on the raw scores
        calibrator = _LRCalibrator(max_iter=1000)
        calibrator.fit(raw_scores.reshape(-1, 1), y_cal)

    return _CalibratedWrapper(base_pipe, calibrator, method)


def _get_inner_pipeline(pipe) -> Pipeline:
    """
    Retrieve the base sklearn Pipeline from a ``_CalibratedWrapper``
    (or return the pipe unchanged if it is not wrapped).

    Used by SHAP (needs the raw XGBoost/LR model) and ``predict.py``
    (needs ``feature_names_in_`` from the imputer step).
    """
    if hasattr(pipe, "calibrated_classifiers_"):
        try:
            return pipe.calibrated_classifiers_[0].estimator
        except (IndexError, AttributeError):
            pass
    return pipe


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def save_shap_analysis(
    pipe:         Pipeline,
    X_test:       pd.DataFrame,
    feature_cols: list[str],
    crop:         str,
    outputs_dir:  Path,
    max_display:  int = 20,
) -> None:
    """
    Generate a SHAP beeswarm summary plot for a trained crop model and save
    it to *outputs_dir*.

    Works with any sklearn-compatible estimator inside the Pipeline:
    - XGBoost → shap.TreeExplainer  (fast, exact)
    - LogisticRegression → shap.LinearExplainer  (fast, exact)
    - Any other model → shap.Explainer  (model-agnostic, slower)

    The SHAP values are computed on the same feature space the classifier sees
    (i.e. after the imputer, and after the scaler if present).  Feature names
    in the plot always correspond to the original column names.

    Silently skips if ``shap`` is not installed.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")           # non-interactive — safe in scripts
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "shap is not installed — skipping SHAP analysis for %s. "
            "Run: pip install shap", crop
        )
        return

    # Unwrap CalibratedClassifierCV — SHAP needs the raw XGBoost/LR model
    inner_pipe = _get_inner_pipeline(pipe)
    clf = inner_pipe.named_steps["clf"]

    # Transform through every preprocessing step except the final classifier
    X_preprocessed = inner_pipe[:-1].transform(X_test)
    X_df = pd.DataFrame(X_preprocessed, columns=feature_cols)

    # Choose the most accurate / fastest explainer for the model type
    if hasattr(clf, "get_booster"):                     # XGBoost
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_df)
    elif hasattr(clf, "coef_"):                         # LogisticRegression / LinearSVC
        explainer   = shap.LinearExplainer(clf, X_df)
        shap_values = explainer.shap_values(X_df)
    else:                                               # model-agnostic fallback
        explainer   = shap.Explainer(clf.predict_proba, X_df)
        shap_values = explainer(X_df).values[:, :, 1]  # class=1 (positive)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_df,
        feature_names = feature_cols,
        show          = False,
        max_display   = max_display,
    )
    plt.title(f"{crop} — SHAP Feature Importance (positive risk class)", fontsize=13)
    plt.tight_layout()

    out_path = outputs_dir / f"shap_{crop.lower()}_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved SHAP plot → %s", out_path)


def save_confusion_matrix(
    y_test:      pd.Series,
    y_proba:     "np.ndarray",
    threshold:   float,
    crop:        str,
    model_label: str,
    outputs_dir: Path,
) -> None:
    """
    Plot and save the confusion matrix at the deployment threshold.

    Why at the deployment threshold (not 0.5)?
    Because that is the threshold actually used in production.  A confusion
    matrix computed at 0.5 would describe a model you never actually deploy.

    Layout
    ------
    Rows  = actual class  (Negative / Positive)
    Cols  = predicted class
    Cells = raw counts + row-normalised percentage (recall / specificity)

    Saved to outputs/confusion_<crop>_<model_label>.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_pred = (y_proba >= threshold).astype(int)
    cm     = confusion_matrix(y_test, y_pred)

    # Row-normalised (recall for each class)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    labels   = ["Negative", "Positive"]
    n_neg, n_pos = cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Draw coloured cells manually for full control
    colours = [["#d5e8d4", "#f8cecc"], ["#f8cecc", "#d5e8d4"]]   # TN/FP/FN/TP
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                       color=colours[i][j], zorder=0))
            ax.text(
                j + 0.5, 1.5 - i,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center", fontsize=14, fontweight="bold",
            )

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(labels[::-1], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)

    tn, fp, fn, tp = cm.ravel()
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    ax.set_title(
        f"{crop} — {model_label} Confusion Matrix\n"
        f"Threshold = {threshold:.4f}  |  "
        f"Recall = {recall:.1%}  |  Precision = {precision:.1%}  |  Specificity = {specificity:.1%}\n"
        f"(Actual negatives = {n_neg}   Actual positives = {n_pos})",
        fontsize=10,
    )

    # Corner labels
    corner = {"TN": (0.02, 1.97), "FP": (1.02, 1.97),
              "FN": (0.02, 0.97), "TP": (1.02, 0.97)}
    for lbl, (x, y) in corner.items():
        ax.text(x, y, lbl, fontsize=9, color="#555555",
                ha="left", va="top", style="italic")

    plt.tight_layout()
    safe_label = model_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_path   = outputs_dir / f"confusion_{crop.lower()}_{safe_label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved confusion matrix → %s", out_path)


def save_calibration_curve(
    y_test:        pd.Series,
    y_proba_uncal: "np.ndarray | None",
    y_proba_cal:   "np.ndarray",
    crop:          str,
    model_label:   str,
    outputs_dir:   Path,
    n_bins:        int = 10,
) -> None:
    """
    Plot and save a reliability diagram (calibration curve) comparing raw model
    probabilities against calibrated probabilities.

    A well-calibrated model's curve lies close to the diagonal: when it says
    60 %, roughly 60 % of those cases actually had disease.  XGBoost raw scores
    typically bow away from the diagonal — the calibration curve makes this
    visible and confirms the isotonic/sigmoid correction worked.

    Two-panel output
    ────────────────
    Left  : Reliability diagram — fraction of positives vs mean predicted
            probability, in n_bins equal-width buckets.  Shows both before
            and after calibration so the improvement is obvious.
    Right : Probability histogram — distribution of calibrated probabilities
            split by true class.  Well-separated = model discriminates well;
            peaked near 0 and 1 = model is confident where it should be.

    Saved to outputs/calibration_<crop>_<model_label>.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{crop} — {model_label}  Probability Calibration",
        fontsize=13, fontweight="bold",
    )

    # ── Left: reliability diagram ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    if y_proba_uncal is not None:
        try:
            frac_u, mean_u = sk_calibration_curve(
                y_test, y_proba_uncal, n_bins=n_bins, strategy="uniform"
            )
            ax.plot(mean_u, frac_u, "s-", color="#e74c3c", linewidth=2,
                    markersize=6, label="Before calibration (raw XGBoost)")
        except Exception:
            pass

    frac_c, mean_c = sk_calibration_curve(
        y_test, y_proba_cal, n_bins=n_bins, strategy="uniform"
    )
    ax.plot(mean_c, frac_c, "o-", color="#27ae60", linewidth=2,
            markersize=6, label="After calibration")

    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title("Reliability Diagram\n(closer to diagonal = better calibrated)", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Right: probability histogram by true class ────────────────────────────
    ax2 = axes[1]
    neg_mask = (np.asarray(y_test) == 0)
    pos_mask = ~neg_mask
    ax2.hist(y_proba_cal[neg_mask], bins=20, alpha=0.65,
             color="#2980b9", label="Actual negative", density=True)
    ax2.hist(y_proba_cal[pos_mask], bins=20, alpha=0.65,
             color="#e74c3c", label="Actual positive", density=True)
    ax2.set_xlabel("Calibrated probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title(
        "Calibrated Probability Distribution\nby True Class", fontsize=10
    )
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    safe_label = model_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_path   = outputs_dir / f"calibration_{crop.lower()}_{safe_label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved calibration curve → %s", out_path)


# ---------------------------------------------------------------------------
# Disease encoding helper
# ---------------------------------------------------------------------------

def _encode_disease_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    One-hot encode the ``test`` (disease name) column within a crop subset.

    Why this matters
    ----------------
    Different diseases respond to the same weather very differently:
      - Tar Spot thrives in cool, humid nights (positive rate ~24 % in data)
      - Southern Rust explodes in hot, dry wind (positive rate ~95 % in data)

    Without the disease identity, the model sees contradictory labels for
    identical weather observations and learns an average that is accurate for
    no individual disease.  Adding disease as a one-hot feature lets XGBoost
    learn disease-specific weather thresholds — the single largest missing
    signal in the current model.

    Implementation notes
    --------------------
    * Category names are sanitised (lower-case, non-alphanumeric -> underscore)
      so they are safe as DataFrame column names and CLI arguments.
    * Dummies are computed on the whole crop subset BEFORE the train/test split
      so that both halves see identical column sets.
    * Columns are sorted for deterministic ordering across runs.
    * Returns the augmented DataFrame and the sorted list of dummy column names.
    * If the 'test' column is absent, returns the input unchanged with an
      empty dummy list — training degrades gracefully to the original features.
    """
    test_col = "test"
    if test_col not in df.columns:
        logger.warning(
            "  'test' column not found — disease feature will be omitted. "
            "Check that build_weather_spore_data.py produced the merged CSV correctly."
        )
        return df, []

    test_clean = (
        df[test_col]
        .fillna("unknown")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    dummies  = pd.get_dummies(test_clean, prefix="disease", dtype=float)
    dcols    = sorted(dummies.columns.tolist())
    df_out   = pd.concat(
        [df.reset_index(drop=True), dummies.reset_index(drop=True)],
        axis=1,
    )
    logger.info(
        "  Disease dummies: %d categories -> %s", len(dcols), dcols
    )
    return df_out, dcols


def _disease_key(disease_name: str) -> str:
    """Convert a human-readable disease name to its dummy column name."""
    return "disease_" + re.sub(r"[^a-z0-9]+", "_", disease_name.strip().lower()).strip("_")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_all_crops(data_path: str, cfg: dict) -> pd.DataFrame:
    import datetime

    logger.info("Loading dataset: %s", data_path)
    raw = pd.read_csv(data_path, low_memory=False)
    logger.info("Raw shape: %d rows × %d columns", *raw.shape)

    # ── Data fingerprint — detect silent data changes between runs ────────────
    data_hash = hashlib.md5(Path(data_path).read_bytes()).hexdigest()
    logger.info("Dataset MD5: %s", data_hash)

    df = build_features(raw, cfg)

    models_dir  = Path(cfg["paths"]["models_dir"])
    outputs_dir = Path(cfg["paths"].get("outputs_dir",
                        str(Path(cfg["_root"]) / "outputs")))
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Full feature set (weather + temporal + rolling) — used by weather model
    feature_cols = get_feature_columns(cfg, df)
    logger.info("Full feature columns (%d): %s", len(feature_cols), feature_cols)

    # Temporal-only feature set — safe for rows without GPS/weather data
    temporal_cols = get_temporal_feature_columns(cfg)
    logger.info("Temporal feature columns (%d): %s", len(temporal_cols), temporal_cols)

    crops               = cfg.get("crops", ["Corn", "Soybean", "Potato"])
    min_samples         = cfg["data"]["min_samples_per_crop"]
    min_weather_samples = cfg["data"].get("min_weather_samples", 50)
    test_size           = cfg["data"]["test_size"]
    results             = []

    # ── LR config — used for baseline and the temporal-only fallback model ──
    lr_cfg = {
        "type":         "LogisticRegression",
        "class_weight": cfg["model"].get("class_weight", "balanced"),
        "max_iter":     cfg["model"].get("max_iter", 1000),
        "random_state": cfg["model"].get("random_state", 42),
        "solver":       cfg["model"].get("solver", "lbfgs"),
    }

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

        cv_folds           = cfg["model"]["cv_folds"]
        n_jobs             = cfg["model"].get("n_jobs", 1)
        min_recall         = cfg["model"].get("min_recall", 0.90)
        medium_band_factor = cfg["model"].get("medium_band_factor", 0.70)
        calibration_cfg    = cfg["model"].get("calibration", {})

        # ══════════════════════════════════════════════════════════════════════
        # Step A — Logistic Regression baseline  (optional, on GPS rows only)
        # ══════════════════════════════════════════════════════════════════════
        # Trains LR on the full feature set so the numbers are directly
        # comparable to Step B (XGBoost on the same rows and features).
        # Saved as <crop>_model_baseline.pkl — NOT used by predict.py.
        # Provides a sanity-check: "how much does XGBoost add over LR?"
        do_baseline = (
            cfg.get("baseline", {}).get("enabled", False)
            and cfg["model"].get("type", "LogisticRegression") != "LogisticRegression"
        )

        # ── Disease one-hot encoding (per-crop) ──────────────────────────────
        # Computed on the full crop_df BEFORE any split so train and test
        # always see the same column set.  disease_cols are appended to both
        # the full (weather) and temporal feature lists below.
        crop_df, disease_cols = _encode_disease_dummies(crop_df)

        # Extended feature sets
        full_feature_cols     = feature_cols + disease_cols   # weather model
        # Temporal fallback: temporal + deployment_duration + disease
        extra_temporal = (
            ["deployment_duration_days"]
            if "deployment_duration_days" in crop_df.columns
            else []
        )
        temporal_full_cols = temporal_cols + extra_temporal + disease_cols

        # ── Prepare GPS (weather-available) rows ─────────────────────────────
        weather_available_col = "weather_available"
        if weather_available_col in crop_df.columns:
            weather_df = crop_df[crop_df[weather_available_col]].copy()
        else:
            weather_df = crop_df.copy()   # graceful fallback if flag is missing
        logger.info("  GPS-matched rows (weather available): %d", len(weather_df))

        # Full-feature clean subset for weather-capable rows
        # Drop rows where ANY core numeric feature (weather + rolling) is NaN.
        # Disease dummies are 0/1 — never NaN — so they are excluded from dropna.
        core_numeric_cols = [c for c in full_feature_cols if not c.startswith("disease_")]
        weather_use_cols  = full_feature_cols + [TARGET_COL, "start_date"]
        weather_clean     = weather_df[
            [c for c in weather_use_cols if c in weather_df.columns]
        ].dropna(subset=core_numeric_cols + [TARGET_COL])
        logger.info(
            "  GPS rows after NaN drop (full features): %d", len(weather_clean)
        )

        if do_baseline and len(weather_clean) >= min_weather_samples:
            class_dist_w = weather_clean[TARGET_COL].value_counts().to_dict()
            if len(class_dist_w) >= 2:
                train_w, test_w = temporal_train_test_split(
                    weather_clean, "start_date", test_size
                )
                X_tr_w = train_w[full_feature_cols]
                y_tr_w = train_w[TARGET_COL].astype(int)
                X_te_w = test_w[full_feature_cols]
                y_te_w = test_w[TARGET_COL].astype(int)

                if len(y_te_w.unique()) >= 2:
                    logger.info(
                        "  [A] Training LR baseline (GPS rows, %d features = "
                        "%d numeric + %d disease dummies) ...",
                        len(full_feature_cols), len(feature_cols), len(disease_cols),
                    )
                    baseline_pipe, baseline_metrics, _ = evaluate_crop(
                        X_tr_w, y_tr_w, X_te_w, y_te_w,
                        model_cfg=lr_cfg, cv_folds=cv_folds, n_jobs=n_jobs,
                        min_recall=min_recall,
                        medium_band_factor=medium_band_factor,
                        calibration_cfg=calibration_cfg,
                        crop=crop,
                    )
                    baseline_path = models_dir / f"{crop.lower()}_model_baseline.pkl"
                    joblib.dump(baseline_pipe, baseline_path)
                    logger.info("  [A] Saved baseline -> %s", baseline_path)

                    bl_proba = baseline_pipe.predict_proba(X_te_w)[:, 1]
                    save_confusion_matrix(
                        y_test      = y_te_w,
                        y_proba     = bl_proba,
                        threshold   = baseline_metrics["threshold_optimal"],
                        crop        = crop,
                        model_label = "LR_baseline",
                        outputs_dir = outputs_dir,
                    )

                    results.append({
                        "crop":                         crop,
                        "model_type":                   "LogisticRegression (baseline)",
                        "data_subset":                  "GPS rows",
                        "train_samples":                len(X_tr_w),
                        "test_samples":                 len(X_te_w),
                        "cv_auc_mean":                  baseline_metrics["cv_auc_mean"],
                        "cv_auc_std":                   baseline_metrics["cv_auc_std"],
                        "cv_f1_mean":                   baseline_metrics["cv_f1_mean"],
                        "test_auc":                     baseline_metrics["test_auc"],
                        "test_f1":                      baseline_metrics["test_f1"],
                        "test_precision":               baseline_metrics["test_precision"],
                        "test_recall":                  baseline_metrics["test_recall"],
                        "threshold_optimal":            baseline_metrics["threshold_optimal"],
                        "threshold_low_max":            baseline_metrics["threshold_low_max"],
                        "threshold_high_min":           baseline_metrics["threshold_high_min"],
                        "threshold_achieved_recall":    baseline_metrics["threshold_achieved_recall"],
                        "threshold_achieved_precision": baseline_metrics["threshold_achieved_precision"],
                        "threshold_min_recall_target":  baseline_metrics["threshold_min_recall_target"],
                        "threshold_youden":             baseline_metrics["threshold_youden"],
                        "n_features":                   len(full_feature_cols),
                        "features":                     ", ".join(full_feature_cols),
                    })

        # ══════════════════════════════════════════════════════════════════════
        # Step B — Weather model  (XGBoost, GPS rows only, full feature set)
        # ══════════════════════════════════════════════════════════════════════
        # Trained exclusively on rows that have real, location-specific weather
        # data.  Only these rows have agronomically meaningful features; imputing
        # cross-location weather means into non-GPS rows would add noise, not
        # signal.
        #
        # Saved as: <crop>_model_weather.pkl + <crop>_weather_thresholds.json
        # predict.py uses this model when weather data is available at inference.
        if len(weather_clean) >= min_weather_samples:
            class_dist_w = weather_clean[TARGET_COL].value_counts().to_dict()
            logger.info("  [B] Weather subset class distribution: %s", class_dist_w)

            if len(class_dist_w) < 2:
                logger.warning("  [B] Skipping weather model -- only one class in GPS rows.")
            else:
                train_w, test_w = temporal_train_test_split(
                    weather_clean, "start_date", test_size
                )
                X_tr_w = train_w[full_feature_cols]
                y_tr_w = train_w[TARGET_COL].astype(int)
                X_te_w = test_w[full_feature_cols]
                y_te_w = test_w[TARGET_COL].astype(int)

                if len(y_te_w.unique()) < 2:
                    logger.warning(
                        "  [B] Skipping weather model -- test split has only one class."
                    )
                else:
                    logger.info(
                        "  [B] Training %s (GPS rows, %d features = %d numeric + %d disease dummies) ...",
                        cfg["model"].get("type", "XGBoost"),
                        len(full_feature_cols), len(feature_cols), len(disease_cols),
                    )
                    weather_pipe, weather_metrics, w_proba_uncal = evaluate_crop(
                        X_tr_w, y_tr_w, X_te_w, y_te_w,
                        model_cfg=cfg["model"],
                        cv_folds=cv_folds, n_jobs=n_jobs,
                        min_recall=min_recall,
                        medium_band_factor=medium_band_factor,
                        calibration_cfg=calibration_cfg,
                        crop=crop,
                    )

                    weather_model_path = models_dir / f"{crop.lower()}_model_weather.pkl"
                    joblib.dump(weather_pipe, weather_model_path)

                    weather_thr_path = models_dir / f"{crop.lower()}_weather_thresholds.json"
                    weather_thr_path.write_text(json.dumps({
                        "crop":                 crop,
                        "model":                "weather",
                        "feature_set":          "full (weather + location + temporal + rolling + disease)",
                        "data_subset":          "GPS-matched rows only",
                        "n_features":           len(full_feature_cols),
                        "numeric_features":     feature_cols,
                        "disease_features":     disease_cols,
                        "calibrated":           weather_metrics.get("calibrated", False),
                        "calibration_method":   weather_metrics.get("calibration_method"),
                        "calibration_samples":  weather_metrics.get("calibration_samples", 0),
                        "optimal":            weather_metrics["threshold_optimal"],
                        "low_max":            weather_metrics["threshold_low_max"],
                        "base_rate":          weather_metrics["threshold_base_rate"],
                        "high_min":           weather_metrics["threshold_high_min"],
                        "achieved_recall":    weather_metrics["threshold_achieved_recall"],
                        "achieved_precision": weather_metrics["threshold_achieved_precision"],
                        "min_recall_target":  weather_metrics["threshold_min_recall_target"],
                        "recall_shortfall":   weather_metrics["threshold_recall_shortfall"],
                        "youden":             weather_metrics["threshold_youden"],
                        "medium_band_factor": weather_metrics["threshold_medium_band_factor"],
                        "derivation": (
                            f"Recall-constrained: >={weather_metrics['threshold_min_recall_target']*100:.0f}% "
                            f"recall (achieved {weather_metrics['threshold_achieved_recall']*100:.1f}%) "
                            f"with max precision ({weather_metrics['threshold_achieved_precision']*100:.1f}%) "
                            f"on GPS-rows chronological hold-out. "
                            f"MEDIUM band: tau_low = tau* x {weather_metrics['threshold_medium_band_factor']:.2f}."
                        ),
                    }, indent=2), encoding="utf-8")

                    logger.info("  [B] Saved weather model      -> %s", weather_model_path)
                    logger.info("  [B] Saved weather thresholds -> %s", weather_thr_path)

                    # Confusion matrix + calibration curve
                    w_proba = weather_pipe.predict_proba(X_te_w)[:, 1]
                    w_label = f"{cfg['model'].get('type', 'XGBoost')}_weather"
                    save_confusion_matrix(
                        y_test      = y_te_w,
                        y_proba     = w_proba,
                        threshold   = weather_metrics["threshold_optimal"],
                        crop        = crop,
                        model_label = w_label,
                        outputs_dir = outputs_dir,
                    )
                    if weather_metrics.get("calibrated"):
                        save_calibration_curve(
                            y_test        = y_te_w,
                            y_proba_uncal = w_proba_uncal,
                            y_proba_cal   = w_proba,
                            crop          = crop,
                            model_label   = w_label,
                            outputs_dir   = outputs_dir,
                        )

                    # SHAP — pass full feature list so disease dummies are labelled
                    shap_cfg = cfg.get("explainability", {})
                    if shap_cfg.get("enabled", True):
                        save_shap_analysis(
                            pipe         = weather_pipe,
                            X_test       = X_te_w,
                            feature_cols = full_feature_cols,
                            crop         = crop,
                            outputs_dir  = outputs_dir,
                            max_display  = shap_cfg.get("max_display", 20),
                        )

                    results.append({
                        "crop":                         crop,
                        "model_type":                   cfg["model"].get("type", "XGBoost") + " (weather)",
                        "data_subset":                  "GPS rows",
                        "train_samples":                len(X_tr_w),
                        "test_samples":                 len(X_te_w),
                        "cv_auc_mean":                  weather_metrics["cv_auc_mean"],
                        "cv_auc_std":                   weather_metrics["cv_auc_std"],
                        "cv_f1_mean":                   weather_metrics["cv_f1_mean"],
                        "test_auc":                     weather_metrics["test_auc"],
                        "test_f1":                      weather_metrics["test_f1"],
                        "test_precision":               weather_metrics["test_precision"],
                        "test_recall":                  weather_metrics["test_recall"],
                        "threshold_optimal":            weather_metrics["threshold_optimal"],
                        "threshold_low_max":            weather_metrics["threshold_low_max"],
                        "threshold_high_min":           weather_metrics["threshold_high_min"],
                        "threshold_achieved_recall":    weather_metrics["threshold_achieved_recall"],
                        "threshold_achieved_precision": weather_metrics["threshold_achieved_precision"],
                        "threshold_min_recall_target":  weather_metrics["threshold_min_recall_target"],
                        "threshold_youden":             weather_metrics["threshold_youden"],
                        "n_features":                   len(full_feature_cols),
                        "features":                     ", ".join(full_feature_cols),
                    })
        else:
            logger.warning(
                "  [B] Skipping weather model -- only %d GPS rows (minimum %d).",
                len(weather_clean), min_weather_samples,
            )

        # ══════════════════════════════════════════════════════════════════════
        # Step C — Temporal fallback model  (LogisticRegression, ALL rows)
        # ══════════════════════════════════════════════════════════════════════
        # Trained on the full crop dataset (GPS + non-GPS) using temporal
        # features + deployment duration + disease one-hot dummies.
        #
        # Disease identity is ALWAYS known at inference time (the user selected
        # the cassette type), so it is safe to include here even for non-GPS
        # rows that lack weather.  Going from 3 to 7-9 features makes this
        # fallback significantly more accurate without adding imputation risk.
        #
        # Why LogisticRegression (not XGBoost) for this model?
        # With 7-9 features and ~3k rows, LR is the right choice: it converges
        # faster, is not prone to overfitting a thin feature space, and its
        # coefficients are directly interpretable by agronomists.
        #
        # Saved as: <crop>_model.pkl + <crop>_thresholds.json
        # predict.py falls back to this model when no weather data is available.

        # Require non-NaN in temporal features only (disease dummies are 0/1,
        # deployment_duration is median-imputed downstream).
        all_use_cols = temporal_full_cols + [TARGET_COL, "start_date"]
        all_clean    = crop_df[
            [c for c in all_use_cols if c in crop_df.columns]
        ].dropna(subset=temporal_cols + [TARGET_COL])
        logger.info(
            "  [C] All rows after NaN drop (temporal + disease features): %d", len(all_clean)
        )

        if len(all_clean) < min_samples:
            logger.warning(
                "  [C] Skipping temporal model — only %d clean rows.", len(all_clean)
            )
            continue

        class_dist_t = all_clean[TARGET_COL].value_counts().to_dict()
        logger.info("  [C] Temporal subset class distribution: %s", class_dist_t)

        if len(class_dist_t) < 2:
            logger.warning("  [C] Skipping temporal model — only one class present.")
            continue

        train_t, test_t = temporal_train_test_split(all_clean, "start_date", test_size)
        # Use the extended temporal feature set (temporal + duration + disease)
        avail_temporal = [c for c in temporal_full_cols if c in all_clean.columns]
        X_tr_t = train_t[avail_temporal]
        y_tr_t = train_t[TARGET_COL].astype(int)
        X_te_t = test_t[avail_temporal]
        y_te_t = test_t[TARGET_COL].astype(int)

        if len(y_te_t.unique()) < 2:
            logger.warning(
                "  [C] Skipping temporal model -- test split has only one class."
            )
            continue

        logger.info(
            "  [C] Training LR fallback (all rows, %d features = "
            "%d temporal + %d duration + %d disease dummies) ...",
            len(avail_temporal),
            len(temporal_cols),
            len(extra_temporal),
            len(disease_cols),
        )
        temporal_pipe, temporal_metrics, t_proba_uncal = evaluate_crop(
            X_tr_t, y_tr_t, X_te_t, y_te_t,
            model_cfg=lr_cfg,
            cv_folds=cv_folds, n_jobs=n_jobs,
            min_recall=min_recall,
            medium_band_factor=medium_band_factor,
            calibration_cfg=calibration_cfg,
            crop=crop,
        )

        temporal_model_path = models_dir / f"{crop.lower()}_model.pkl"
        joblib.dump(temporal_pipe, temporal_model_path)

        temporal_thr_path = models_dir / f"{crop.lower()}_thresholds.json"
        temporal_thr_path.write_text(json.dumps({
            "crop":                 crop,
            "model":                "temporal",
            "feature_set":          "temporal + deployment_duration + disease (all rows)",
            "data_subset":          "all rows",
            "n_features":           len(avail_temporal),
            "temporal_features":    temporal_cols,
            "duration_features":    extra_temporal,
            "disease_features":     disease_cols,
            "calibrated":           temporal_metrics.get("calibrated", False),
            "calibration_method":   temporal_metrics.get("calibration_method"),
            "calibration_samples":  temporal_metrics.get("calibration_samples", 0),
            "optimal":            temporal_metrics["threshold_optimal"],
            "low_max":            temporal_metrics["threshold_low_max"],
            "base_rate":          temporal_metrics["threshold_base_rate"],
            "high_min":           temporal_metrics["threshold_high_min"],
            "achieved_recall":    temporal_metrics["threshold_achieved_recall"],
            "achieved_precision": temporal_metrics["threshold_achieved_precision"],
            "min_recall_target":  temporal_metrics["threshold_min_recall_target"],
            "recall_shortfall":   temporal_metrics["threshold_recall_shortfall"],
            "youden":             temporal_metrics["threshold_youden"],
            "medium_band_factor": temporal_metrics["threshold_medium_band_factor"],
            "derivation": (
                f"Recall-constrained: >={temporal_metrics['threshold_min_recall_target']*100:.0f}% "
                f"recall (achieved {temporal_metrics['threshold_achieved_recall']*100:.1f}%) "
                f"with max precision ({temporal_metrics['threshold_achieved_precision']*100:.1f}%) "
                f"on all-rows chronological hold-out. "
                f"MEDIUM band: tau_low = tau* x {temporal_metrics['threshold_medium_band_factor']:.2f}."
            ),
        }, indent=2), encoding="utf-8")

        logger.info("  [C] Saved temporal model      -> %s", temporal_model_path)
        logger.info("  [C] Saved temporal thresholds -> %s", temporal_thr_path)

        # Confusion matrix + calibration curve
        t_proba = temporal_pipe.predict_proba(X_te_t)[:, 1]
        save_confusion_matrix(
            y_test      = y_te_t,
            y_proba     = t_proba,
            threshold   = temporal_metrics["threshold_optimal"],
            crop        = crop,
            model_label = "LR_temporal",
            outputs_dir = outputs_dir,
        )
        if temporal_metrics.get("calibrated"):
            save_calibration_curve(
                y_test        = y_te_t,
                y_proba_uncal = t_proba_uncal,
                y_proba_cal   = t_proba,
                crop          = crop,
                model_label   = "LR_temporal",
                outputs_dir   = outputs_dir,
            )

        results.append({
            "crop":                         crop,
            "model_type":                   "LogisticRegression (temporal fallback)",
            "data_subset":                  "all rows",
            "train_samples":                len(X_tr_t),
            "test_samples":                 len(X_te_t),
            "cv_auc_mean":                  temporal_metrics["cv_auc_mean"],
            "cv_auc_std":                   temporal_metrics["cv_auc_std"],
            "cv_f1_mean":                   temporal_metrics["cv_f1_mean"],
            "test_auc":                     temporal_metrics["test_auc"],
            "test_f1":                      temporal_metrics["test_f1"],
            "test_precision":               temporal_metrics["test_precision"],
            "test_recall":                  temporal_metrics["test_recall"],
            "threshold_optimal":            temporal_metrics["threshold_optimal"],
            "threshold_low_max":            temporal_metrics["threshold_low_max"],
            "threshold_high_min":           temporal_metrics["threshold_high_min"],
            "threshold_achieved_recall":    temporal_metrics["threshold_achieved_recall"],
            "threshold_achieved_precision": temporal_metrics["threshold_achieved_precision"],
            "threshold_min_recall_target":  temporal_metrics["threshold_min_recall_target"],
            "threshold_youden":             temporal_metrics["threshold_youden"],
            "n_features":                   len(avail_temporal),
            "features":                     ", ".join(avail_temporal),
        })

    results_df = pd.DataFrame(results)

    out_path = Path(cfg["output"]["model_results"])   # already absolute (resolved by config_loader)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info("\nResults written to: %s", out_path)

    # ── Data fingerprint — saved alongside models for reproducibility ─────────
    # If the source CSV changes, the MD5 won't match and you'll know to retrain.
    fingerprint = {
        "training_file":  data_path,
        "md5_hash":       data_hash,
        "row_count":      int(raw.shape[0]),
        "col_count":      int(raw.shape[1]),
        "trained_at":     datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "model_type":     cfg["model"].get("type", "LogisticRegression"),
        "min_recall":     cfg["model"].get("min_recall", 0.90),
        "crops_trained":  results_df["crop"].unique().tolist() if not results_df.empty else [],
    }
    fp_path = models_dir / "training_metadata.json"
    fp_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    logger.info("Fingerprint saved  → %s", fp_path)

    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Force UTF-8 on Windows consoles (cp1252 chokes on → ≥ etc.)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
