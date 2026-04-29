"""
Spornado Disease Risk Model

Production-grade ML system for crop disease risk prediction using the
disease triangle framework: weather conditions + spore pressure + crop variety.

Version: 2.0.0
"""

__version__ = "2.0.0"
__author__  = "Spornado Team"
__license__ = "MIT"

from .config_loader import load               # noqa: F401
from .features      import (                  # noqa: F401
    TARGET_COL,
    build_features,
    get_feature_columns,
)

# Heavy modules (inference, PDF extraction, variety integration) are NOT
# eagerly imported here.  They pull in large third-party libraries (joblib,
# pdfplumber, etc.) that slow down every `import src.*`.  Import directly:
#
#   from src.predict                import Predictor
#   from src.extract_pdf_to_csv     import PDFExtractor
#   from src.integrate_variety_data import VarietyDataIntegrator

__all__ = [
    "load",
    "build_features",
    "get_feature_columns",
    "TARGET_COL",
]
