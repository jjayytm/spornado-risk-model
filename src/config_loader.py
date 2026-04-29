"""
Central config loader.

Usage
-----
from src.config_loader import load

cfg = load()                         # reads config/config.yaml relative to project root
cfg = load(Path("/custom/path.yaml"))

All values under cfg["paths"] and cfg["output"] are returned as absolute
strings resolved from the project root, so callers can safely do:

    Path(cfg["paths"]["models_dir"])
    Path(cfg["output"]["model_results"])
"""

from pathlib import Path
from typing import Optional

import yaml

# Project root = two levels up from this file  (src/config_loader.py → src/ → root/)
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _ROOT / "config" / "config.yaml"

# Config top-level sections whose string values are file / directory paths
# that should be resolved to absolute paths relative to the project root.
_PATH_SECTIONS = ("paths", "output")


def load(path: Optional[Path] = None) -> dict:
    config_path = Path(path) if path else _DEFAULT_CONFIG
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    for section in _PATH_SECTIONS:
        for key, val in cfg.get(section, {}).items():
            if isinstance(val, str):
                cfg[section][key] = str(_ROOT / val)

    # Expose root for callers that need to build ad-hoc paths
    cfg["_root"] = str(_ROOT)
    return cfg
