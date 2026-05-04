#!/usr/bin/env python3
"""
Threshold sensitivity analysis for the Spornado disease risk models.

Purpose
-------
Helps agronomists and crop protection specialists choose the right
``min_recall`` value for their risk tolerance by showing the full
precision–recall tradeoff for each crop.

Key output
----------
For each trained crop model:
  • Precision–recall curve with annotated operating points
  • A table showing: at each recall target, what precision is achievable
    and what threshold is used
  • The current deployed threshold marked clearly

This script does NOT retrain models.  It loads the trained pipeline and
re-applies it to a fresh chronological hold-out split of the data.

Usage
-----
  python -m src.threshold_analysis
  python -m src.threshold_analysis --data data/raw/spornado_weather_spore_data.csv
  python -m src.threshold_analysis --crops Corn Soybean
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config_loader import load as load_config
from src.features import TARGET_COL, build_features, get_feature_columns
from src.build_crop_models import _disease_key, temporal_train_test_split

logger = logging.getLogger(__name__)

# Recall targets to evaluate in the sensitivity table
RECALL_TARGETS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]


def _load_model_and_thresholds(
    crop: str,
    models_dir: Path,
) -> tuple | None:
    """Load trained pipeline and threshold JSON for a crop. Returns None if not found."""
    for suffix, thr_stem in (
        ("_model_variety.pkl", "_thresholds"),
        ("_model_weather.pkl", "_weather_thresholds"),
        ("_model.pkl", "_thresholds"),
    ):
        model_path = models_dir / f"{crop.lower()}{suffix}"
        if model_path.exists():
            pipe = joblib.load(model_path)
            break
    else:
        logger.warning("No model found for '%s' in %s", crop, models_dir)
        return None

    threshold_path = models_dir / f"{crop.lower()}{thr_stem}.json"
    if not threshold_path.exists():
        logger.warning("No threshold JSON found for '%s' at %s", crop, threshold_path)
        return None

    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    return pipe, thresholds


def _get_model_feature_columns(pipe, cfg: dict, fallback_df: pd.DataFrame) -> list[str]:
    """Recover exact model feature order from fitted preprocessing steps."""
    inner = pipe
    if hasattr(pipe, "calibrated_classifiers_"):
        try:
            inner = pipe.calibrated_classifiers_[0].estimator
        except (IndexError, AttributeError):
            pass

    for step_name in ("imputer", "scaler"):
        step = inner.named_steps.get(step_name) if hasattr(inner, "named_steps") else None
        if step is not None and hasattr(step, "feature_names_in_"):
            return list(step.feature_names_in_)
    return get_feature_columns(cfg, fallback_df)


def analyse_crop(
    crop:        str,
    pipe,
    thresholds:  dict,
    X_test:      pd.DataFrame,
    y_test:      pd.Series,
    outputs_dir: Path,
) -> pd.DataFrame:
    """
    For a single crop: compute precision–recall curve, build sensitivity table,
    and save the visualisation.

    Returns
    -------
    DataFrame with columns [recall_target, threshold, precision, recall, feasible]
    """
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
    # Drop the trailing synthetic (1.0, 0.0) point
    precisions     = precisions[:-1]
    recalls        = recalls[:-1]
    pr_thresholds  = pr_thresholds        # already length n-1 from sklearn

    # ── Sensitivity table ────────────────────────────────────────────────────
    rows = []
    for target in RECALL_TARGETS:
        mask = recalls >= target
        if mask.any():
            best_idx = int(np.argmax(np.where(mask, precisions, -np.inf)))
            rows.append({
                "recall_target":    target,
                "threshold":        round(float(pr_thresholds[best_idx]), 4),
                "achieved_recall":  round(float(recalls[best_idx]),    4),
                "precision":        round(float(precisions[best_idx]), 4),
                "feasible":         True,
            })
        else:
            rows.append({
                "recall_target":    target,
                "threshold":        None,
                "achieved_recall":  None,
                "precision":        None,
                "feasible":         False,
            })

    table = pd.DataFrame(rows)

    # ── Print table to log ───────────────────────────────────────────────────
    logger.info("\n%s — Threshold Sensitivity (AUC = %.4f)", crop, auc)
    logger.info(
        "%-15s %-12s %-17s %-12s %s",
        "Recall Target", "Threshold", "Achieved Recall", "Precision", "Feasible"
    )
    logger.info("-" * 70)
    for _, r in table.iterrows():
        if r["feasible"]:
            marker = " ◄ DEPLOYED" if abs(r["recall_target"] - thresholds.get("min_recall_target", 0)) < 0.005 else ""
            logger.info(
                "%-15.0f%% %-12.4f %-17.1f%% %-12.1f%%%s",
                r["recall_target"] * 100,
                r["threshold"],
                r["achieved_recall"] * 100,
                r["precision"] * 100,
                marker,
            )
        else:
            logger.info("%-15.0f%% %-12s (not achievable)", r["recall_target"] * 100, "—")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(recalls, precisions, color="#2980b9", linewidth=2, label=f"PR curve (AUC={auc:.4f})")
    ax.fill_between(recalls, precisions, alpha=0.08, color="#2980b9")

    # Mark each evaluated recall target
    for r in rows:
        if r["feasible"]:
            is_deployed = abs(r["recall_target"] - thresholds.get("min_recall_target", 0)) < 0.005
            color  = "#e74c3c" if is_deployed else "#7f8c8d"
            marker = "*" if is_deployed else "o"
            size   = 14 if is_deployed else 7
            ax.plot(r["achieved_recall"], r["precision"],
                    marker=marker, color=color, markersize=size, zorder=5)
            ax.annotate(
                f"  {r['recall_target']:.0%}",
                (r["achieved_recall"], r["precision"]),
                fontsize=8, color=color,
            )

    # Mark currently deployed threshold
    deployed_recall = thresholds.get("achieved_recall")
    deployed_prec   = thresholds.get("achieved_precision")
    if deployed_recall and deployed_prec:
        ax.annotate(
            f"  DEPLOYED\n  threshold={thresholds['optimal']:.4f}\n  recall={deployed_recall:.1%}\n  prec={deployed_prec:.1%}",
            (deployed_recall, deployed_prec),
            fontsize=9, color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c"),
            xytext=(deployed_recall - 0.25, deployed_prec + 0.15),
        )

    ax.set_xlabel("Recall (outbreak catch rate)", fontsize=12)
    ax.set_ylabel("Precision (alert accuracy)", fontsize=12)
    ax.set_title(
        f"{crop} — Precision–Recall Tradeoff\n"
        f"★ = current deployed threshold  •  each label = min_recall target",
        fontsize=12,
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = outputs_dir / f"threshold_analysis_{crop.lower()}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved plot → %s", out_path)

    return table


def run(
    data_path:   str,
    cfg:         dict,
    crops:       list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run threshold analysis for all (or selected) crops.

    Returns dict mapping crop name → sensitivity DataFrame.
    """
    models_dir  = Path(cfg["paths"]["models_dir"])
    outputs_dir = Path(cfg["paths"].get("outputs_dir",
                        str(Path(cfg["_root"]) / "outputs")))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    target_crops = crops or cfg.get("crops", ["Corn", "Soybean", "Potato"])
    test_size    = cfg["data"]["test_size"]
    min_samples  = cfg["data"]["min_samples_per_crop"]

    logger.info("Loading dataset: %s", data_path)
    raw = pd.read_csv(data_path, low_memory=False)
    df  = build_features(raw, cfg)

    results      = {}

    for crop in target_crops:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Threshold analysis: %s", crop)
        logger.info("=" * 60)

        loaded = _load_model_and_thresholds(crop, models_dir)
        if loaded is None:
            continue
        pipe, thresholds = loaded
        feature_cols = _get_model_feature_columns(pipe, cfg, df)

        crop_df = df[df["crop_type"].str.strip().str.title() == crop].copy()
        if len(crop_df) < min_samples:
            logger.warning("  Skipping — insufficient data (%d rows).", len(crop_df))
            continue

        use_cols = [c for c in ["start_date", TARGET_COL, "test"] if c in crop_df.columns]
        crop_clean = crop_df[[*use_cols, *[c for c in feature_cols if c in crop_df.columns]]].copy()
        crop_clean = crop_clean.dropna(subset=[TARGET_COL, "start_date"])

        # Re-create disease one-hot columns expected by model, if present.
        disease_dummy_cols = [c for c in feature_cols if c.startswith("disease_")]
        if disease_dummy_cols:
            for col in disease_dummy_cols:
                crop_clean[col] = 0.0
            if "test" in crop_clean.columns:
                for idx, val in crop_clean["test"].items():
                    if pd.notna(val):
                        dk = _disease_key(str(val))
                        if dk in disease_dummy_cols:
                            crop_clean.at[idx, dk] = 1.0

        _, test_df = temporal_train_test_split(crop_clean, "start_date", test_size)
        X_test = test_df.reindex(columns=feature_cols)
        y_test = test_df[TARGET_COL].astype(int)

        if len(y_test.unique()) < 2:
            logger.warning("  Skipping — test set has only one class.")
            continue

        table = analyse_crop(crop, pipe, thresholds, X_test, y_test, outputs_dir)
        results[crop] = table

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threshold sensitivity analysis for Spornado crop models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",   default=None, help="Path to merged CSV (overrides config)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--crops",  nargs="+",    help="Crops to analyse (default: all)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    data_path = args.data or cfg["paths"]["data_merged"]
    if not Path(data_path).exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    results = run(data_path, cfg, crops=args.crops)

    if not results:
        logger.error("No crops analysed — check that models are trained and data is available.")
        sys.exit(1)

    print()
    print("=" * 70)
    print("THRESHOLD SENSITIVITY SUMMARY")
    print("=" * 70)
    print("How to read: each row shows what precision is achievable if you")
    print("require at least that recall level.")
    print("Higher recall = fewer missed outbreaks, but more false alarms.")
    print()
    for crop, table in results.items():
        print(f"  {crop}")
        feasible = table[table["feasible"]]
        if not feasible.empty:
            print(feasible[["recall_target", "threshold", "achieved_recall", "precision"]]
                  .to_string(index=False))
        print()
    print("=" * 70)
    print("Plots saved to: outputs/threshold_analysis_<crop>.png")
    print("To change the deployed threshold: edit model.min_recall in config/config.yaml")
    print("then retrain with: python -m src.build_crop_models")


if __name__ == "__main__":
    main()
