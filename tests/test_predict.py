import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.predict import Predictor


WEATHER_COLS = [
    "temperature_min_c",
    "temperature_max_c",
    "temperature_mean_c",
    "humidity_max_percent",
    "precipitation_mm",
    "wind_speed_max_kmh",
    "dew_point_min_c",
]
TEMPORAL_COLS = ["month", "day_of_year", "week_of_year"]


def _fit_model(feature_cols: list[str]) -> Pipeline:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, len(feature_cols)))
    y = (rng.random(80) > 0.4).astype(int)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    # Preserve feature names for Predictor feature alignment.
    pipe.named_steps["imputer"].feature_names_in_ = np.array(feature_cols, dtype=object)
    return pipe


def _base_cfg(models_dir: Path, logs_dir: Path, allow_threshold_fallback: bool = False) -> dict:
    return {
        "paths": {"models_dir": str(models_dir), "logs_dir": str(logs_dir)},
        "features": {"weather": WEATHER_COLS, "temporal": TEMPORAL_COLS},
        "prediction_logging": {"enabled": False},
        "prediction": {
            "allow_threshold_fallback": allow_threshold_fallback,
            "model_selection_file": str(models_dir / "model_selection.json"),
        },
    }


def _write_threshold(path: Path) -> None:
    payload = {
        "low_max": 0.30,
        "high_min": 0.60,
        "optimal": 0.60,
        "min_recall_target": 0.90,
        "achieved_recall": 0.92,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_forces_selected_model_suffix(tmp_path: Path):
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir()
    logs.mkdir()

    # Save both temporal and weather models for same crop.
    joblib.dump(_fit_model(TEMPORAL_COLS), models / "corn_model.pkl")
    joblib.dump(_fit_model(WEATHER_COLS + TEMPORAL_COLS), models / "corn_model_weather.pkl")
    _write_threshold(models / "corn_thresholds.json")

    # Explicitly select temporal model.
    selection = {"Corn": {"model_suffix": "_model.pkl", "threshold_stem": "_thresholds"}}
    (models / "model_selection.json").write_text(json.dumps(selection), encoding="utf-8")

    p = Predictor(_base_cfg(models, logs))
    out = p.predict(crop="Corn", month=8, day_of_year=220, week_of_year=32)

    assert out["crop"] == "Corn"
    assert p._model_key["Corn"] == "_model.pkl"


def test_manifest_weather_model_requires_complete_weather(tmp_path: Path):
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir()
    logs.mkdir()

    joblib.dump(_fit_model(WEATHER_COLS + TEMPORAL_COLS), models / "corn_model_weather.pkl")
    _write_threshold(models / "corn_weather_thresholds.json")
    selection = {
        "Corn": {
            "model_suffix": "_model_weather.pkl",
            "threshold_stem": "_weather_thresholds",
        }
    }
    (models / "model_selection.json").write_text(json.dumps(selection), encoding="utf-8")

    p = Predictor(_base_cfg(models, logs))
    with pytest.raises(ValueError, match="Missing required weather features"):
        p.predict(crop="Corn", month=8, day_of_year=220, week_of_year=32)


def test_missing_threshold_fails_fast_by_default(tmp_path: Path):
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir()
    logs.mkdir()

    joblib.dump(_fit_model(TEMPORAL_COLS), models / "corn_model.pkl")
    selection = {"Corn": {"model_suffix": "_model.pkl", "threshold_stem": "_thresholds"}}
    (models / "model_selection.json").write_text(json.dumps(selection), encoding="utf-8")

    p = Predictor(_base_cfg(models, logs, allow_threshold_fallback=False))
    with pytest.raises(FileNotFoundError, match="Threshold file is required"):
        p.predict(crop="Corn", month=8, day_of_year=220, week_of_year=32)


def test_missing_threshold_uses_fallback_when_enabled(tmp_path: Path):
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir()
    logs.mkdir()

    joblib.dump(_fit_model(TEMPORAL_COLS), models / "corn_model.pkl")
    selection = {"Corn": {"model_suffix": "_model.pkl", "threshold_stem": "_thresholds"}}
    (models / "model_selection.json").write_text(json.dumps(selection), encoding="utf-8")

    p = Predictor(_base_cfg(models, logs, allow_threshold_fallback=True))
    out = p.predict(crop="Corn", month=8, day_of_year=220, week_of_year=32)

    assert "risk_label" in out
    assert out["threshold_low_max"] == 0.25
    assert out["threshold_high_min"] == 0.40
