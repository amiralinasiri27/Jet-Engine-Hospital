"""
data_loading.py
----------------
Load raw C-MAPSS text files (train/test/RUL) with explicit column naming.

Files are whitespace-separated, no header row, 26 columns for train/test files:
    engine_id, cycle, op_setting_1..3, sensor_1..21

RUL files (RUL_FD00x.txt) have a single column: one RUL value per test engine,
in the same order as engines appear in test_FD00x.txt (engine_id 1, 2, 3, ...).
"""

from pathlib import Path
import pandas as pd

COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_engine_file(path):
    """Load a train_FD00x.txt or test_FD00x.txt file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None)

    # CMAPSS files sometimes have trailing whitespace producing extra empty columns
    if df.shape[1] > len(COLUMN_NAMES):
        df = df.iloc[:, : len(COLUMN_NAMES)]
    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"{path.name}: expected {len(COLUMN_NAMES)} columns, got {df.shape[1]}"
        )

    df.columns = COLUMN_NAMES
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        raise ValueError(f"{path.name}: found {n_nan} unexpected NaN values after parsing")

    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def load_rul_file(path):
    """Load RUL_FD00x.txt into a Series indexed 1..N (matching test engine_id order)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape[1] != 1:
        raw = raw.iloc[:, [0]]
    series = raw.iloc[:, 0].astype(int)
    series.index = range(1, len(series) + 1)  # engine_id 1..N, official order
    series.name = "RUL_final"
    return series


def load_dataset(subset, data_dir):
    """Load train/test/RUL trio for a given subset (e.g. 'FD001').

    Returns:
        dict with keys: 'train', 'test', 'rul_final'
    """
    data_dir = Path(data_dir)
    train = load_engine_file(data_dir / f"train_{subset}.txt")
    test = load_engine_file(data_dir / f"test_{subset}.txt")
    rul_final = load_rul_file(data_dir / f"RUL_{subset}.txt")

    n_test_engines = test["engine_id"].nunique()
    if len(rul_final) != n_test_engines:
        raise AssertionError(
            f"{subset}: RUL file has {len(rul_final)} entries but test set has "
            f"{n_test_engines} distinct engines"
        )

    return {"train": train, "test": test, "rul_final": rul_final}
