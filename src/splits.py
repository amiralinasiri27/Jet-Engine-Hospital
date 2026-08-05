"""
splits.py
---------
Engine-level train/validation/test splitting. Must be called BEFORE any
preprocessing, feature fitting, windowing, threshold tuning, or calibration.

Test engines come from the official test_FD00x.txt file and are touched only
once for final reporting. This module handles splitting the TRAINING engines
into train/validation.
"""

import json
from pathlib import Path
import numpy as np
from src.config import SEED


def split_train_val_engines(engine_ids, val_fraction=0.2, seed=SEED):
    """Split training engine IDs into train/validation subsets."""
    engine_ids = sorted(set(int(e) for e in engine_ids))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(engine_ids)

    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = sorted(int(e) for e in shuffled[:n_val])
    train_ids = sorted(int(e) for e in shuffled[n_val:])

    return {"train": train_ids, "val": val_ids}


def save_split(split_dict, path):
    """Persist a split (engine ID lists) to disk as JSON for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split_dict, f, indent=2)


def load_split(path):
    """Load a previously saved split from disk."""
    with open(path) as f:
        return json.load(f)


def assert_disjoint(*engine_id_lists):
    """Sanity check: assert no engine_id appears in more than one split."""
    sets = [set(lst) for lst in engine_id_lists]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise AssertionError(
                    f"Splits {i} and {j} are not disjoint — shared engine_id(s): {sorted(overlap)}"
                )
