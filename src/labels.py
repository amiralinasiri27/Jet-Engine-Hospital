"""
labels.py
---------
Construct RUL regression targets and horizon classification labels.

Rules (see project brief section 3.3):
  - Train RUL:   T(i) = max(cycle) per engine;  RUL(i,t) = T(i) - t
  - Test RUL:    T(i) = max(cycle) + RUL_final(i)  (from RUL_FD00x.txt), aligned by engine order
  - Classification: y(i,t,h) = 1 if RUL(i,t) <= h else 0, for h in (10, 20, 30)
"""

import pandas as pd
from src.config import HORIZONS


def add_train_rul(train_df, cap=None):
    """Add a 'RUL' column to the training DataFrame.

    Args:
        cap: optional piecewise cap (e.g. 125). If given, RUL is clipped at
             this value. Must be chosen/tuned only on train/validation engines.
    """
    df = train_df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_test_rul(test_df, rul_final):
    """Add a 'RUL' column to the test DataFrame using the final RUL per engine.

    Args:
        rul_final: pd.Series indexed by engine_id (1..N, official order),
                   as returned by data_loading.load_rul_file.
    """
    df = test_df.copy()
    engine_ids = df["engine_id"].unique()

    missing = set(engine_ids) - set(rul_final.index)
    if missing:
        raise AssertionError(f"engine_id(s) missing from rul_final: {sorted(missing)}")

    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    rul_final_per_row = df["engine_id"].map(rul_final)
    total_life = max_cycle + rul_final_per_row  # T(i) = last observed cycle + remaining RUL
    df["RUL"] = total_life - df["cycle"]
    return df


def add_classification_labels(df, horizons=HORIZONS):
    """Add one binary column per horizon: label_h10, label_h20, label_h30.

    Must be called AFTER RUL has been correctly constructed via add_train_rul
    or add_test_rul.
    """
    if "RUL" not in df.columns:
        raise ValueError("df must have a 'RUL' column before adding classification labels")
    df = df.copy()
    for h in horizons:
        df[f"label_h{h}"] = (df["RUL"] <= h).astype(int)
    return df
