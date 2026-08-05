"""
config.py
---------
Central place for paths, seeds, and constants shared across the notebook,
src/ modules, and the app. Avoid hard-coding these values elsewhere.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "CMAPSSData"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SEED = 42

HORIZONS = (10, 20, 30)

DEFAULT_COST_POLICY = dict(
    c_miss=10.0,
    c_late=3.0,
    c_early=1.0,
    horizon_h=10,
)

# Datasets used in this project
STAGE_1_DATASET = "FD001"
STAGE_2_DATASET = "FD003"
