#!/usr/bin/env python3
"""
Retrain per-crop models using variety disease tolerance features  (3 / 3 disease triangle).

Run this after integrate_variety_data.py has produced the enriched dataset.
The pipeline is identical to build_crop_models.py but extends the feature set
with per-variety disease tolerance ratings.

Usage
-----
  python -m src.build_crop_models_with_variety
  python -m src.build_crop_models_with_variety \\
      --data data/processed/spornado_with_variety_features.csv
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
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.build_crop_models import evaluate_crop, temporal_train_test_split
from src.config_loader import load as load_config
from src.features import TARGET_COL, build_features, get_feature_columns

logger = logging.getLogger(__name__)


def _setup_logging(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"]),
        format=cfg["logging"]["format"],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "build_models_variety.log", mode="w", encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Variety-aware model builder
# ---------------------------------------------------------------------------

class VarietyAwareModelBuilder:
    """
    Extends the baseline model by including variety disease tolerance ratings
    as additional features alongside the weather and temporal features.
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg       = cfg
        self._models:   dict[str, Pipeline]      = {}
        self._features: dict[str, list[str]]     = {}

    def _get_feature_cols(self, cfg: dict, df: pd.DataFrame) -> list[str]:
        """
        Base weather + temporal + rolling features, plus any variety rating columns.
        Variety rating columns are identified by the '_Rating' suffix added by
        integrate_variety_data.py.
        """
        base     = get_feature_columns(cfg, df)
        variety  = [c for c in df.columns if c.endswith("_Rating") or c.endswith("_Tolerance")]
        combined = base + variety
        if variety:
            logger.info("  Variety features (%d): %s", len(variety), variety)
        else:
            logger.warning(
                "  No variety rating columns found — running without variety features. "
                "Run integrate_variety_data.py first if variety data is available."
            )
        return combined

    def _build_pipeline(self) -> Pipeline:
        m = self._cfg["model"]
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),   # handles NaN weather / variety features
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(
                class_weight=m["class_weight"],
                max_iter=m["max_iter"],
                random_state=m["random_state"],
                solver=m["solver"],
            )),
        ])

    def _evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_folds: int,
        crop: str,
    ) -> tuple[Pipeline, dict]:
        """
        Leakage-safe evaluation:
        - CV on X_train only
        - threshold tuning on train-only chronological window
        - untouched hold-out test metrics at pre-tuned threshold
        """
        cv = TimeSeriesSplit(n_splits=cv_folds)   # preserves temporal order; no CV leakage
        cv_pipe = self._build_pipeline()
        cv_res = cross_validate(
            cv_pipe, X_train, y_train,
            cv=cv,
            scoring=["roc_auc", "f1", "precision", "recall"],
            return_train_score=False,
            n_jobs=self._cfg["model"].get("n_jobs", 1),
        )
        logger.info(
            "  CV (%d-fold) AUC: %.4f ± %.4f  |  F1: %.4f ± %.4f",
            cv_folds,
            cv_res["test_roc_auc"].mean(), cv_res["test_roc_auc"].std(),
            cv_res["test_f1"].mean(),       cv_res["test_f1"].std(),
        )

        # Use the shared leakage-safe evaluator so variety path stays aligned
        # with the main training protocol.
        pipe, metrics, _ = evaluate_crop(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_cfg={
                "type": "LogisticRegression",
                "class_weight": self._cfg["model"]["class_weight"],
                "max_iter": self._cfg["model"]["max_iter"],
                "random_state": self._cfg["model"]["random_state"],
                "solver": self._cfg["model"]["solver"],
                "n_jobs": self._cfg["model"].get("n_jobs", 1),
            },
            cv_folds=cv_folds,
            crop=crop,
            n_jobs=self._cfg["model"].get("n_jobs", 1),
            min_recall=self._cfg["model"].get("min_recall", 0.90),
            medium_band_factor=self._cfg["model"].get("medium_band_factor", 0.70),
            calibration_cfg=self._cfg["model"].get("calibration", {}),
        )
        # Keep externally reported CV fields from this function's explicit CV run.
        metrics["cv_auc_mean"] = float(cv_res["test_roc_auc"].mean())
        metrics["cv_auc_std"] = float(cv_res["test_roc_auc"].std())
        metrics["cv_f1_mean"] = float(cv_res["test_f1"].mean())
        metrics["cv_f1_std"] = float(cv_res["test_f1"].std())
        return pipe, metrics

    def build_all_models(self, data_path: str) -> pd.DataFrame:
        cfg         = self._cfg
        min_samples = cfg["data"]["min_samples_per_crop"]
        test_size   = cfg["data"]["test_size"]
        crops       = cfg.get("crops", ["Corn", "Soybean", "Potato"])
        models_dir  = Path(cfg["paths"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading dataset: %s", data_path)
        raw = pd.read_csv(data_path, low_memory=False)
        logger.info("Raw shape: %d rows × %d columns", *raw.shape)

        df = build_features(raw, cfg)

        results: list[dict] = []

        for crop in crops:
            logger.info("")
            logger.info("=" * 60)
            logger.info("Crop: %s  (with variety features)", crop)
            logger.info("=" * 60)

            crop_df = df[df["crop_type"].str.strip().str.title() == crop].copy()
            logger.info("  Total rows: %d", len(crop_df))

            if len(crop_df) < min_samples:
                logger.warning("  Skipping — insufficient data (%d < %d).", len(crop_df), min_samples)
                continue

            feature_cols = self._get_feature_cols(cfg, crop_df)
            self._features[crop] = feature_cols

            use_cols   = feature_cols + [TARGET_COL, "start_date"]
            crop_clean = crop_df[[c for c in use_cols if c in crop_df.columns]].dropna(
                subset=feature_cols + [TARGET_COL]
            )
            logger.info("  Clean rows: %d", len(crop_clean))

            if len(crop_clean) < min_samples:
                logger.warning("  Skipping after NaN drop (%d < %d).", len(crop_clean), min_samples)
                continue

            class_dist = crop_clean[TARGET_COL].value_counts().to_dict()
            logger.info("  Class distribution: %s", class_dist)

            if len(class_dist) < 2:
                logger.warning("  Skipping — only one class present.")
                continue

            train_df, test_df = temporal_train_test_split(crop_clean, "start_date", test_size)

            X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
            y_train = train_df[TARGET_COL].astype(int)
            X_test  = test_df[feature_cols].fillna(train_df[feature_cols].median())
            y_test  = test_df[TARGET_COL].astype(int)

            if len(y_test.unique()) < 2:
                logger.warning("  Skipping — test split has only one class.")
                continue

            pipe, metrics = self._evaluate(
                X_train, y_train, X_test, y_test,
                cv_folds=cfg["model"]["cv_folds"],
                crop=crop,
            )
            self._models[crop] = pipe

            model_path = models_dir / f"{crop.lower()}_model_variety.pkl"
            joblib.dump(pipe, model_path)

            import json as _json
            threshold_path = models_dir / f"{crop.lower()}_thresholds.json"
            threshold_path.write_text(_json.dumps({
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
                    f"Recall-constrained (variety model): threshold chosen to achieve "
                    f">={metrics['threshold_min_recall_target']*100:.0f}% recall "
                    f"(achieved {metrics['threshold_achieved_recall']*100:.1f}%) "
                    f"on train-only chronological threshold-tuning window."
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
                "model_type":                   "LogisticRegression (Pipeline: Imputer → Scaler → LR + Variety)",
                "n_features":                   len(feature_cols),
                "features":                     ", ".join(feature_cols),
            })

        return pd.DataFrame(results)

    def print_comparison(self, results_df: pd.DataFrame) -> None:
        print()
        print("=" * 70)
        print("DISEASE TRIANGLE — 3 / 3 COMPONENTS")
        print("=" * 70)
        print("Weather  ✓  |  Spore pressure  ✓  |  Crop variety  ✓")
        print()
        if not results_df.empty:
            print(results_df[["crop", "cv_auc_mean", "cv_auc_std", "test_auc",
                               "test_f1", "n_features"]].to_string(index=False))
        print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Spornado models with crop variety tolerance features (3/3 triangle)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        help="Path to integrated variety dataset (overrides config output.integrated_data)",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    _setup_logging(cfg)

    data_path = args.data or cfg["output"]["integrated_data"]

    if not Path(data_path).exists():
        logger.error(
            "Integrated dataset not found: %s\n"
            "Run integrate_variety_data.py first.",
            data_path,
        )
        sys.exit(1)

    builder    = VarietyAwareModelBuilder(cfg)
    results_df = builder.build_all_models(data_path)

    if results_df.empty:
        logger.error("No models were successfully trained.")
        sys.exit(1)

    out_path = Path(cfg["output"]["variety_results"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info("Results saved: %s", out_path)

    builder.print_comparison(results_df)


if __name__ == "__main__":
    main()
