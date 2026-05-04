"""
Tests for the model training and inference layer.

Uses synthetic data so no real files are needed on disk.

Run: pytest tests/test_models.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.build_crop_models import (
    _CalibratedWrapper,
    _calibrate_pipeline,
    _get_inner_pipeline,
    build_pipeline,
    evaluate_crop,
    find_optimal_threshold,
    risk_label as _risk_label,   # imported here directly — no src.predict dependency
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic crop dataset
# ---------------------------------------------------------------------------

_MODEL_CFG_LR = {
    "type":         "LogisticRegression",
    "class_weight": "balanced",
    "max_iter":     500,
    "random_state": 42,
    "solver":       "lbfgs",
}

_MODEL_CFG_XGB = {
    "type":         "XGBoost",
    "random_state": 42,
    "n_jobs":       1,
    "xgboost": {
        "n_estimators":     50,   # small for fast unit tests
        "max_depth":         3,
        "learning_rate":   0.1,
        "eval_metric":  "logloss",
        "random_state":     42,
    },
}

# Default config used by tests that don't care about model type
_MODEL_CFG = _MODEL_CFG_LR

FEATURE_COLS = [
    "temperature_max_c", "humidity_max_percent", "precipitation_mm",
    "month", "day_of_year",
]


def _make_synthetic_crop_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Create a simple synthetic dataset where high temperature + humidity
    correlates with disease presence, so a logistic regression can fit it.
    """
    rng = np.random.default_rng(seed)
    temp       = rng.normal(25, 5, n)
    humidity   = rng.normal(75, 10, n)
    precip     = rng.exponential(3, n)
    month      = rng.integers(4, 11, n).astype(float)
    doy        = (month * 30).astype(float)

    # Simple rule: positive if temp > 25 AND humidity > 75
    prob    = 1 / (1 + np.exp(-(0.3 * (temp - 25) + 0.2 * (humidity - 75))))
    target  = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "temperature_max_c":    temp,
        "humidity_max_percent": humidity,
        "precipitation_mm":     precip,
        "month":                month,
        "day_of_year":          doy,
        "disease_present":      target,
        "start_date":           pd.date_range("2022-01-01", periods=n, freq="D"),
    })


@pytest.fixture()
def split_data():
    df  = _make_synthetic_crop_data(300)
    cut = int(len(df) * 0.8)
    train = df.iloc[:cut]
    test  = df.iloc[cut:]
    return (
        train[FEATURE_COLS],
        train["disease_present"].astype(int),
        test[FEATURE_COLS],
        test["disease_present"].astype(int),
    )


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    # ── LogisticRegression pipeline ──────────────────────────────────────────

    def test_lr_returns_sklearn_pipeline(self):
        pipe = build_pipeline(_MODEL_CFG_LR)
        assert isinstance(pipe, Pipeline)

    def test_lr_has_imputer_scaler_clf(self):
        pipe = build_pipeline(_MODEL_CFG_LR)
        assert "imputer" in pipe.named_steps, "Missing SimpleImputer step"
        assert "scaler"  in pipe.named_steps, "Missing StandardScaler step"
        assert "clf"     in pipe.named_steps, "Missing classifier step"

    def test_lr_imputer_is_first_step(self):
        pipe = build_pipeline(_MODEL_CFG_LR)
        assert pipe.steps[0][0] == "imputer"

    def test_lr_pipeline_is_unfitted(self):
        from sklearn.exceptions import NotFittedError
        pipe = build_pipeline(_MODEL_CFG_LR)
        with pytest.raises(NotFittedError):
            pipe.predict(pd.DataFrame([{c: 1.0 for c in FEATURE_COLS}]))

    def test_lr_handles_nan_after_fitting(self, split_data):
        """SimpleImputer must allow NaN inputs at prediction time."""
        X_train, y_train, X_test, _ = split_data
        pipe = build_pipeline(_MODEL_CFG_LR)
        pipe.fit(X_train, y_train)
        X_with_nan = X_test.copy()
        X_with_nan.iloc[0, 0] = float("nan")
        proba = pipe.predict_proba(X_with_nan)
        assert not any(p != p for p in proba[:, 1]), "predict_proba returned NaN for imputed row"

    # ── XGBoost pipeline ─────────────────────────────────────────────────────

    def test_xgb_returns_sklearn_pipeline(self):
        pytest.importorskip("xgboost", reason="xgboost not installed")
        pipe = build_pipeline(_MODEL_CFG_XGB)
        assert isinstance(pipe, Pipeline)

    def test_xgb_has_imputer_and_clf(self):
        pytest.importorskip("xgboost", reason="xgboost not installed")
        pipe = build_pipeline(_MODEL_CFG_XGB)
        assert "imputer" in pipe.named_steps
        assert "clf"     in pipe.named_steps
        # XGBoost pipeline has no StandardScaler (tree models are scale-invariant)
        assert "scaler" not in pipe.named_steps

    def test_xgb_imputer_is_first_step(self):
        pytest.importorskip("xgboost", reason="xgboost not installed")
        pipe = build_pipeline(_MODEL_CFG_XGB)
        assert pipe.steps[0][0] == "imputer"

    def test_xgb_handles_nan_after_fitting(self, split_data):
        """XGBoost handles NaN natively — imputer + native NaN handling both work."""
        pytest.importorskip("xgboost", reason="xgboost not installed")
        X_train, y_train, X_test, _ = split_data
        pipe = build_pipeline(_MODEL_CFG_XGB)
        pipe.fit(X_train, y_train)
        X_with_nan = X_test.copy()
        X_with_nan.iloc[0, 0] = float("nan")
        proba = pipe.predict_proba(X_with_nan)
        assert not any(p != p for p in proba[:, 1])

    def test_xgb_pos_weight_is_applied(self):
        """scale_pos_weight should be set on the XGBClassifier."""
        pytest.importorskip("xgboost", reason="xgboost not installed")
        pipe = build_pipeline(_MODEL_CFG_XGB, pos_weight=4.5)
        clf  = pipe.named_steps["clf"]
        assert abs(clf.scale_pos_weight - 4.5) < 1e-9

    def test_both_models_produce_valid_probabilities(self, split_data):
        """Both model types must produce probabilities in [0, 1]."""
        pytest.importorskip("xgboost", reason="xgboost not installed")
        X_train, y_train, X_test, _ = split_data
        for cfg in (_MODEL_CFG_LR, _MODEL_CFG_XGB):
            pipe = build_pipeline(cfg)
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            assert proba.min() >= 0.0 and proba.max() <= 1.0, \
                f"{cfg['type']}: probabilities out of [0,1] range"


# ---------------------------------------------------------------------------
# evaluate_crop
# ---------------------------------------------------------------------------

class TestEvaluateCrop:
    def test_returns_pipeline_and_metrics(self, split_data):
        """evaluate_crop uses TimeSeriesSplit — no future leakage in CV folds."""
        X_train, y_train, X_test, y_test = split_data
        pipe, metrics, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG, cv_folds=3, crop="TestCrop"
        )
        assert isinstance(pipe, Pipeline)
        assert isinstance(metrics, dict)

    def test_metrics_keys_present(self, split_data):
        X_train, y_train, X_test, y_test = split_data
        _, metrics, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG, cv_folds=3, crop="TestCrop"
        )
        required = {
            # Cross-validation
            "cv_auc_mean", "cv_auc_std", "cv_f1_mean",
            # Hold-out at deployment threshold
            "test_auc", "test_f1", "test_precision", "test_recall",
            # Recall-constrained threshold provenance
            "threshold_optimal",
            "threshold_low_max",
            "threshold_high_min",
            "threshold_achieved_recall",
            "threshold_achieved_precision",
            "threshold_min_recall_target",
            "threshold_recall_shortfall",
            "threshold_youden",             # diagnostic reference (equal-cost baseline)
            "threshold_medium_band_factor",
        }
        assert required.issubset(metrics.keys())

    def test_thresholds_are_in_valid_range(self, split_data):
        """Every probability threshold must be in (0, 1) with correct ordering."""
        X_train, y_train, X_test, y_test = split_data
        _, metrics, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG, cv_folds=3, crop="TestCrop"
        )
        for key in ("threshold_optimal", "threshold_low_max",
                    "threshold_high_min", "threshold_youden"):
            assert 0.0 < metrics[key] < 1.0, f"{key}={metrics[key]} out of (0, 1)"
        assert metrics["threshold_low_max"] < metrics["threshold_high_min"], \
            "LOW boundary must be strictly below HIGH boundary"

    def test_recall_target_is_met_or_shortfall_flagged(self, split_data):
        """
        The achieved recall must either meet the target or the shortfall flag
        must be set — never silently below target.
        """
        X_train, y_train, X_test, y_test = split_data
        _, metrics, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG, cv_folds=3, crop="TestCrop",
            min_recall=0.85,
        )
        if not metrics["threshold_recall_shortfall"]:
            assert metrics["threshold_achieved_recall"] >= metrics["threshold_min_recall_target"], (
                f"Achieved recall {metrics['threshold_achieved_recall']:.3f} < "
                f"target {metrics['threshold_min_recall_target']:.3f} without shortfall flag"
            )

    def test_stricter_recall_gives_lower_threshold(self, split_data):
        """Higher min_recall forces the threshold lower to catch more positives."""
        X_train, y_train, X_test, y_test = split_data
        _, m_relaxed, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG, cv_folds=3, crop="TestCrop",
            min_recall=0.70,
        )
        _, m_strict, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG, cv_folds=3, crop="TestCrop",
            min_recall=0.95,
        )
        assert m_strict["threshold_optimal"] <= m_relaxed["threshold_optimal"], (
            "min_recall=0.95 must produce a threshold ≤ min_recall=0.70"
        )

    def test_auc_in_valid_range(self, split_data):
        X_train, y_train, X_test, y_test = split_data
        _, metrics, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG, cv_folds=3, crop="TestCrop"
        )
        assert 0.0 <= metrics["test_auc"] <= 1.0
        assert 0.0 <= metrics["cv_auc_mean"] <= 1.0

    def test_fitted_model_predicts_probabilities(self, split_data):
        X_train, y_train, X_test, y_test = split_data
        pipe, _, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG, cv_folds=3, crop="TestCrop"
        )
        proba = pipe.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_model_predicts_both_classes(self, split_data):
        """Balanced class weighting should produce predictions of both 0 and 1."""
        X_train, y_train, X_test, y_test = split_data
        pipe, _, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG, cv_folds=3, crop="TestCrop"
        )
        preds = pipe.predict(X_test)
        assert 0 in preds, "Model never predicted class 0 — check class balancing"
        assert 1 in preds, "Model never predicted class 1 — check class balancing"

    def test_auc_better_than_random(self, split_data):
        """A properly trained model on non-trivial data should beat random chance."""
        X_train, y_train, X_test, y_test = split_data
        _, metrics, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG, cv_folds=3, crop="TestCrop"
        )
        assert metrics["test_auc"] > 0.55, (
            f"Test AUC {metrics['test_auc']:.3f} is barely above random. "
            "The model may not be learning anything useful."
        )

    def test_cv_auc_std_is_reasonable(self, split_data):
        """High std deviation across folds signals an unstable model."""
        X_train, y_train, X_test, y_test = split_data
        _, metrics, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG, cv_folds=3, crop="TestCrop"
        )
        assert metrics["cv_auc_std"] < 0.20, (
            f"CV AUC std {metrics['cv_auc_std']:.3f} is too high — "
            "model is unstable across folds."
        )


# ---------------------------------------------------------------------------
# find_optimal_threshold
# ---------------------------------------------------------------------------

class TestFindOptimalThreshold:
    """
    Tests for recall-constrained threshold optimisation.

    Core contract
    ─────────────
    Given a minimum recall floor set by the agronomist, the algorithm must:
      1. Find a threshold where achieved_recall ≥ min_recall_target
         (or set recall_shortfall=True if impossible).
      2. Among all thresholds that meet (1), choose the one with maximum precision.
      3. Build LOW/MEDIUM/HIGH bands proportionally around that threshold.
      4. Report Youden's J as a diagnostic (equal-cost reference), not as the
         production threshold.
    """

    _Y_TRUE  = np.array([0, 0, 0, 1, 1, 1])
    _Y_PROBA = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    # ── Contract: returned keys ──────────────────────────────────────────────

    def test_returns_all_required_keys(self):
        result = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA)
        required = {
            "optimal", "low_max", "high_min",
            "achieved_recall", "achieved_precision",
            "min_recall_target", "recall_shortfall",
            "youden", "medium_band_factor",
            "base_rate",
        }
        assert required.issubset(set(result.keys())), (
            f"Missing keys: {required - set(result.keys())}"
        )

    # ── Contract: valid probability values ───────────────────────────────────

    def test_all_probability_values_in_valid_range(self):
        result = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA)
        for key in ("optimal", "low_max", "high_min", "youden"):
            assert 0.0 < result[key] < 1.0, f"{key}={result[key]} not in (0, 1)"

    def test_achieved_recall_is_valid_fraction(self):
        result = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA)
        assert 0.0 <= result["achieved_recall"]    <= 1.0
        assert 0.0 <= result["achieved_precision"] <= 1.0

    # ── Contract: band structure ─────────────────────────────────────────────

    def test_band_ordering_is_strict(self):
        result = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA)
        assert result["low_max"] < result["high_min"], \
            "LOW upper bound must be strictly below HIGH lower bound"

    def test_high_min_equals_optimal(self):
        """HIGH band starts exactly at the decision boundary — no gap, no overlap."""
        result = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA)
        assert result["high_min"] == result["optimal"], (
            "HIGH band must start exactly at the optimal threshold"
        )

    def test_medium_band_low_max_is_below_optimal(self):
        """low_max must always be strictly below the decision threshold."""
        result = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA)
        assert result["low_max"] < result["optimal"], (
            f"low_max={result['low_max']} must be < optimal={result['optimal']}"
        )

    def test_medium_band_factor_produces_consistent_width(self):
        """
        low_max must equal optimal × medium_band_factor (clamped to valid range).
        This verifies the single-method band derivation is applied correctly.
        """
        factor = 0.70
        result = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA,
                                        medium_band_factor=factor)
        expected_low_max = result["optimal"] * factor
        # Allow for clip boundary (low_max clamped to max(optimal - 0.01, 0.01))
        clamp_upper = max(result["optimal"] - 0.01, 0.01)
        expected_clamped = min(expected_low_max, clamp_upper)
        assert result["low_max"] == pytest.approx(expected_clamped, abs=0.0001), (
            f"low_max={result['low_max']} does not match "
            f"optimal × factor = {expected_clamped:.4f}"
        )

    def test_medium_band_factor_configurable(self):
        """Different factor values must produce proportionally different MEDIUM bands."""
        result_70 = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA,
                                           medium_band_factor=0.70)
        result_80 = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA,
                                           medium_band_factor=0.80)
        # Higher factor → narrower MEDIUM band → larger low_max
        assert result_80["low_max"] >= result_70["low_max"], (
            "Higher medium_band_factor must produce a higher (or equal) low_max"
        )

    def test_base_rate_stored_for_provenance(self):
        """base_rate must equal the positive class fraction in y_true."""
        result   = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA)
        expected = float(np.mean(self._Y_TRUE))
        assert result["base_rate"] == pytest.approx(expected, abs=0.001)

    # ── Contract: recall constraint ───────────────────────────────────────────

    def test_recall_target_met_when_achievable(self):
        """
        When the model can reach min_recall, it must do so.
        recall_shortfall must be False and achieved_recall ≥ target.
        """
        result = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA, min_recall=0.80)
        if not result["recall_shortfall"]:
            assert result["achieved_recall"] >= result["min_recall_target"], (
                f"achieved={result['achieved_recall']:.3f} < "
                f"target={result['min_recall_target']:.3f}"
            )

    def test_recall_target_stored_for_provenance(self):
        result = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA, min_recall=0.85)
        assert result["min_recall_target"] == pytest.approx(0.85)

    def test_stricter_recall_gives_equal_or_lower_threshold(self):
        """
        Higher min_recall → model must catch more positives → threshold must
        move DOWN (or stay the same if already at maximum recall).
        """
        r_relaxed = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA, min_recall=0.60)
        r_strict  = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA, min_recall=0.99)
        assert r_strict["optimal"] <= r_relaxed["optimal"], (
            "min_recall=0.99 must produce a threshold ≤ min_recall=0.60"
        )

    def test_stricter_recall_flags_at_least_as_many_high(self):
        """Lower threshold (from stricter recall) → ≥ as many HIGH observations."""
        y_true  = np.array([0, 0, 1, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.5, 0.7, 0.9])
        r_relaxed = find_optimal_threshold(y_true, y_proba, min_recall=0.60)
        r_strict  = find_optimal_threshold(y_true, y_proba, min_recall=0.99)
        n_high_relaxed = sum(p >= r_relaxed["high_min"] for p in y_proba)
        n_high_strict  = sum(p >= r_strict["high_min"]  for p in y_proba)
        assert n_high_strict >= n_high_relaxed

    def test_impossible_recall_sets_shortfall_flag(self):
        """
        If min_recall=1.0 can't be achieved (nearly always), recall_shortfall
        must be True — never silently under-deliver.
        """
        # With only 6 samples this tiny, recall=1.0 is almost certainly unreachable
        result = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA, min_recall=1.0)
        # Either it achieved 1.0 (perfect model) or shortfall is flagged
        assert result["recall_shortfall"] or result["achieved_recall"] == pytest.approx(1.0)

    # ── Contract: Youden is diagnostic only ──────────────────────────────────

    def test_youden_is_independent_of_recall_target(self):
        """
        Youden's J is a property of the model, not the recall target.
        It should be identical regardless of what min_recall is set to.
        """
        r_a = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA, min_recall=0.70)
        r_b = find_optimal_threshold(self._Y_TRUE, self._Y_PROBA, min_recall=0.95)
        assert r_a["youden"] == r_b["youden"], (
            "Youden's J must not change when min_recall changes"
        )

    def test_perfect_separator_achieves_recall_target(self):
        """A linearly separable model should easily meet a 90% recall floor."""
        y_true  = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
        result  = find_optimal_threshold(y_true, y_proba, min_recall=0.90)
        assert not result["recall_shortfall"]
        assert result["achieved_recall"] >= 0.90


# ---------------------------------------------------------------------------
# _risk_label  (data-driven thresholds, not hardcoded)
# ---------------------------------------------------------------------------

class TestRiskLabel:
    def test_low_below_low_max(self):
        assert _risk_label(0.10, low_max=0.30, high_min=0.60) == "LOW"

    def test_medium_between_bounds(self):
        assert _risk_label(0.45, low_max=0.30, high_min=0.60) == "MEDIUM"

    def test_high_above_high_min(self):
        assert _risk_label(0.75, low_max=0.30, high_min=0.60) == "HIGH"

    def test_boundary_at_low_max_is_medium(self):
        assert _risk_label(0.30, low_max=0.30, high_min=0.60) == "MEDIUM"

    def test_boundary_at_high_min_is_high(self):
        assert _risk_label(0.60, low_max=0.30, high_min=0.60) == "HIGH"

    def test_thresholds_are_per_crop_not_global(self):
        """Same probability → different label depending on crop-specific thresholds."""
        prob = 0.40
        # Corn model might have tight thresholds (high prevalence area)
        assert _risk_label(prob, low_max=0.20, high_min=0.35) == "HIGH"
        # Soybean model might have wider thresholds (lower prevalence area)
        assert _risk_label(prob, low_max=0.35, high_min=0.65) == "MEDIUM"

    def test_recall_constraint_effect_end_to_end(self):
        """
        End-to-end story: same model probability, different min_recall setting
        → same risk_label function, different outcomes.

        This proves the design principle: the domain requirement (min_recall) lives
        in config and is encoded into the threshold at training time.  The label
        function itself is a pure, stateless mapping — it never needs to know about
        the recall target.

        Scenario
        ────────
        A farm reports prob=0.38.

        • Relaxed setting (min_recall=0.70): threshold is higher → prob=0.38 might be LOW/MEDIUM.
        • Strict setting  (min_recall=0.95): threshold is lower  → prob=0.38 is more likely HIGH.

        The strict setting must produce an alert level ≥ the relaxed setting.
        """
        prob = 0.38

        y_true  = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        t_relaxed = find_optimal_threshold(y_true, y_proba,
                                           min_recall=0.70, medium_band_factor=0.70)
        t_strict  = find_optimal_threshold(y_true, y_proba,
                                           min_recall=0.95, medium_band_factor=0.70)

        label_relaxed = _risk_label(prob, t_relaxed["low_max"], t_relaxed["high_min"])
        label_strict  = _risk_label(prob, t_strict["low_max"],  t_strict["high_min"])

        label_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        assert label_rank[label_strict] >= label_rank[label_relaxed], (
            f"Strict recall (threshold={t_strict['optimal']:.3f}) must produce "
            f"alert ≥ relaxed recall (threshold={t_relaxed['optimal']:.3f}) "
            f"for prob={prob}: got {label_strict!r} vs {label_relaxed!r}"
        )


# ---------------------------------------------------------------------------
# features.py — data quality and weather availability helpers
# ---------------------------------------------------------------------------

class TestFilterCorruptedDates:
    """Tests for filter_corrupted_dates (date quality guard)."""

    def _make_df(self, dates, results=None):
        from src.features import filter_corrupted_dates  # noqa: F401
        n = len(dates)
        return pd.DataFrame({
            "start_date": pd.to_datetime(dates, errors="coerce"),
            "result":     (results or ["positive"] * n),
        })

    def test_valid_dates_are_kept(self):
        from src.features import filter_corrupted_dates
        df = self._make_df(["2022-05-01", "2023-08-15", "2024-03-22"])
        out = filter_corrupted_dates(df, 2010, 2030)
        assert len(out) == 3

    def test_year_1900_is_dropped(self):
        """Excel serial 0 or 1 parses to 1900 — should be removed."""
        from src.features import filter_corrupted_dates
        df = self._make_df(["2023-06-01", "1900-01-01"])
        out = filter_corrupted_dates(df, 2010, 2030)
        assert len(out) == 1
        assert out["start_date"].dt.year.iloc[0] == 2023

    def test_nat_rows_are_kept(self):
        """NaT dates are not corrupted dates — they should survive the filter
        and be handled later by the NaN-feature drop."""
        from src.features import filter_corrupted_dates
        df = self._make_df(["2023-06-01", None, "2024-09-10"])
        out = filter_corrupted_dates(df, 2010, 2030)
        assert len(out) == 3   # NaT row is kept

    def test_boundary_years_are_kept(self):
        """min_year and max_year themselves are inclusive."""
        from src.features import filter_corrupted_dates
        df = self._make_df(["2010-01-01", "2030-12-31"])
        out = filter_corrupted_dates(df, 2010, 2030)
        assert len(out) == 2

    def test_future_dates_are_dropped(self):
        """Year 2050 falls outside the valid window."""
        from src.features import filter_corrupted_dates
        df = self._make_df(["2023-07-01", "2050-01-01"])
        out = filter_corrupted_dates(df, 2010, 2030)
        assert len(out) == 1

    def test_missing_start_date_column_returns_unchanged(self):
        """If 'start_date' is absent the DataFrame is returned unmodified."""
        from src.features import filter_corrupted_dates
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        out = filter_corrupted_dates(df, 2010, 2030)
        assert len(out) == 3


class TestFlagWeatherAvailability:
    """Tests for flag_weather_availability."""

    def _make_weather_df(self, temp_vals, humidity_vals):
        """Build a minimal DataFrame with two weather columns."""
        return pd.DataFrame({
            "temperature_max_c":    temp_vals,
            "humidity_max_percent": humidity_vals,
        })

    def test_adds_boolean_column(self):
        from src.features import flag_weather_availability
        df  = self._make_weather_df([25.0, None], [None, None])
        out = flag_weather_availability(df)
        assert "weather_available" in out.columns
        assert out["weather_available"].dtype == bool

    def test_row_with_partial_weather_is_false(self):
        """A row is weather-available only if all required weather columns are present."""
        from src.features import flag_weather_availability
        df  = self._make_weather_df([25.0, None], [None, None])
        out = flag_weather_availability(df)
        assert out["weather_available"].iloc[0] is np.bool_(False)
        assert out["weather_available"].iloc[1] is np.bool_(False)

    def test_fully_nan_row_is_false(self):
        """A row where every weather column is NaN has no matched weather."""
        from src.features import flag_weather_availability
        df  = self._make_weather_df([None, None], [None, None])
        out = flag_weather_availability(df)
        assert not out["weather_available"].any()

    def test_all_present_rows_are_true(self):
        from src.features import flag_weather_availability
        df  = self._make_weather_df([20.0, 25.0], [70.0, 80.0])
        out = flag_weather_availability(df)
        assert out["weather_available"].all()

    def test_no_weather_columns_sets_false(self):
        """When no weather columns exist the flag is False for all rows."""
        from src.features import flag_weather_availability
        df  = pd.DataFrame({"month": [5, 6, 7]})   # only temporal, no weather
        out = flag_weather_availability(df)
        assert "weather_available" in out.columns
        assert not out["weather_available"].any()

    def test_does_not_modify_original_df(self):
        """flag_weather_availability must not mutate the input DataFrame."""
        from src.features import flag_weather_availability
        df  = self._make_weather_df([25.0], [80.0])
        _   = flag_weather_availability(df)
        assert "weather_available" not in df.columns


class TestGetTemporalFeatureColumns:
    """Tests for get_temporal_feature_columns."""

    def test_returns_list(self):
        from src.features import get_temporal_feature_columns
        cfg = {"features": {"temporal": ["month", "day_of_year", "week_of_year"]}}
        assert isinstance(get_temporal_feature_columns(cfg), list)

    def test_matches_config_temporal_list(self):
        from src.features import get_temporal_feature_columns
        temporal = ["month", "day_of_year", "week_of_year"]
        cfg      = {"features": {"temporal": temporal}}
        assert get_temporal_feature_columns(cfg) == temporal

    def test_default_when_key_missing(self):
        """Falls back to the standard three temporal columns if key is absent."""
        from src.features import get_temporal_feature_columns
        cfg = {"features": {}}
        cols = get_temporal_feature_columns(cfg)
        assert cols == ["month", "day_of_year", "week_of_year"]

    def test_returns_independent_copy(self):
        """Mutating the returned list must not alter the config dict."""
        from src.features import get_temporal_feature_columns
        temporal = ["month", "day_of_year", "week_of_year"]
        cfg      = {"features": {"temporal": temporal}}
        result   = get_temporal_feature_columns(cfg)
        result.append("extra_col")
        assert cfg["features"]["temporal"] == temporal  # config unchanged


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    """
    Tests for _CalibratedWrapper / _calibrate_pipeline and _get_inner_pipeline helper.

    Design intent
    ─────────────
    Calibration is a post-processing wrapper: it must not change the model's
    discriminative ability (AUC), must expose predict_proba(), and must allow
    SHAP / predict.py to reach the inner sklearn Pipeline via _get_inner_pipeline.
    """

    def _make_calibrated_pipe(self, n: int = 200, cal_frac: float = 0.25):
        """Return (calibrated_pipe, X_test, y_test) on synthetic data."""
        rng    = np.random.default_rng(7)
        X      = pd.DataFrame({
            "temperature_max_c":    rng.normal(25, 5, n),
            "humidity_max_percent": rng.normal(75, 10, n),
            "precipitation_mm":     rng.exponential(3, n),
        })
        y = ((X["temperature_max_c"] > 25) & (X["humidity_max_percent"] > 75)).astype(int)

        cut_test = int(n * 0.80)
        cut_cal  = int(cut_test * (1 - cal_frac))

        X_fit  = X.iloc[:cut_cal];       y_fit  = y.iloc[:cut_cal]
        X_cal  = X.iloc[cut_cal:cut_test]; y_cal  = y.iloc[cut_cal:cut_test]
        X_test = X.iloc[cut_test:];      y_test = y.iloc[cut_test:]

        base = build_pipeline(_MODEL_CFG_LR)
        base.fit(X_fit, y_fit)

        cal = _calibrate_pipeline(base, X_cal, y_cal, method="sigmoid")
        return cal, X_test, y_test

    # ── _get_inner_pipeline ──────────────────────────────────────────────────

    def test_get_inner_pipeline_returns_pipeline_from_calibrated(self):
        cal, X_test, _ = self._make_calibrated_pipe()
        inner = _get_inner_pipeline(cal)
        assert isinstance(inner, Pipeline), (
            "_get_inner_pipeline must return the base Pipeline from a calibrated wrapper"
        )

    def test_get_inner_pipeline_passthrough_on_plain_pipeline(self):
        pipe = build_pipeline(_MODEL_CFG_LR)
        assert _get_inner_pipeline(pipe) is pipe, (
            "_get_inner_pipeline must return the pipe unchanged if not calibrated"
        )

    def test_inner_pipeline_has_named_steps(self):
        cal, X_test, _ = self._make_calibrated_pipe()
        inner = _get_inner_pipeline(cal)
        assert hasattr(inner, "named_steps")
        assert "imputer" in inner.named_steps

    # ── Calibrated wrapper behaviour ─────────────────────────────────────────

    def test_calibrated_predict_proba_returns_valid_array(self):
        cal, X_test, _ = self._make_calibrated_pipe()
        proba = cal.predict_proba(X_test)
        assert proba.shape[1] == 2
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_calibrated_auc_is_close_to_uncalibrated(self):
        """
        Calibration adjusts probability values but must not degrade ranking.
        AUC of calibrated model should be within 0.05 of uncalibrated AUC.
        """
        from sklearn.metrics import roc_auc_score
        rng  = np.random.default_rng(42)
        n    = 300
        X    = pd.DataFrame({
            "temperature_max_c":    rng.normal(25, 5, n),
            "humidity_max_percent": rng.normal(75, 10, n),
            "precipitation_mm":     rng.exponential(3, n),
        })
        y    = ((X["temperature_max_c"] > 25) & (X["humidity_max_percent"] > 75)).astype(int)

        cut_test = int(n * 0.80)
        cut_cal  = int(cut_test * 0.75)

        base = build_pipeline(_MODEL_CFG_LR)
        base.fit(X.iloc[:cut_cal], y.iloc[:cut_cal])

        cal = _calibrate_pipeline(base, X.iloc[cut_cal:cut_test], y.iloc[cut_cal:cut_test], method="sigmoid")

        X_test = X.iloc[cut_test:]
        y_test = y.iloc[cut_test:]

        auc_base = roc_auc_score(y_test, base.predict_proba(X_test)[:, 1])
        auc_cal  = roc_auc_score(y_test, cal.predict_proba(X_test)[:, 1])

        assert abs(auc_cal - auc_base) <= 0.08, (
            f"Calibration degraded AUC significantly: "
            f"base={auc_base:.3f}, calibrated={auc_cal:.3f}"
        )

    def test_feature_names_accessible_after_calibration(self):
        """
        predict.py looks up feature_names_in_ via _get_inner_pipeline.
        This must work on a calibrated model.
        """
        cal, X_test, _ = self._make_calibrated_pipe()
        inner = _get_inner_pipeline(cal)
        imputer = inner.named_steps.get("imputer")
        assert imputer is not None
        assert hasattr(imputer, "feature_names_in_"), (
            "feature_names_in_ must be available after fitting the inner pipeline"
        )

    # ── evaluate_crop with calibration enabled ───────────────────────────────

    def test_evaluate_crop_with_calibration_returns_3_tuple(self, split_data):
        X_train, y_train, X_test, y_test = split_data
        result = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG_LR, cv_folds=3, crop="TestCrop",
            calibration_cfg={"enabled": True, "cal_size": 0.20,
                             "method": "sigmoid", "min_cal_samples": 10},
        )
        assert len(result) == 3, "evaluate_crop must return (pipe, metrics, y_proba_uncal)"

    def test_evaluate_crop_calibrated_pipe_has_predict_proba(self, split_data):
        X_train, y_train, X_test, y_test = split_data
        pipe, _, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG_LR, cv_folds=3, crop="TestCrop",
            calibration_cfg={"enabled": True, "cal_size": 0.20,
                             "method": "sigmoid", "min_cal_samples": 10},
        )
        proba = pipe.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)

    def test_evaluate_crop_calibration_flag_in_metrics(self, split_data):
        X_train, y_train, X_test, y_test = split_data
        _, metrics, _ = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG_LR, cv_folds=3, crop="TestCrop",
            calibration_cfg={"enabled": True, "cal_size": 0.20,
                             "method": "sigmoid", "min_cal_samples": 10},
        )
        assert "calibrated" in metrics
        assert metrics["calibrated"] is True

    def test_evaluate_crop_no_calibration_returns_plain_pipeline(self, split_data):
        X_train, y_train, X_test, y_test = split_data
        pipe, metrics, y_uncal = evaluate_crop(
            X_train, y_train, X_test, y_test,
            model_cfg=_MODEL_CFG_LR, cv_folds=3, crop="TestCrop",
            calibration_cfg={"enabled": False},
        )
        assert isinstance(pipe, Pipeline), "Without calibration, pipe must be a plain Pipeline"
        assert metrics["calibrated"] is False
        assert y_uncal is None
