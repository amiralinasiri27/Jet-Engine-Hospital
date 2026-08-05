"""
features.py
-----------
Causal (leakage-safe) feature engineering. All rolling/window statistics are
TRAILING (use only cycles <= t), never centered.

FeaturePipeline is the single shared feature-construction object used by both
the training notebook and the deployed app (app parity).
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


@dataclass
class FeaturePipeline:
    window_sizes: tuple = (5, 10, 20)
    near_constant_std_threshold: float = 1e-3
    condition_aware: bool = False
    n_conditions: int = 6
    # SIMPLIFICATION: with 3 window sizes x (mean/std/min/max) the feature
    # matrix explodes to ~255 columns for FD001 (15 kept sensors), most of
    # them near-duplicates of each other (mean_w5 vs mean_w10 vs ewma all
    # carry almost the same signal; min/max add little for a monotonically
    # drifting sensor). That hurts the linear baselines badly (ridge/
    # ridge_poly) and, combined with a modest number of boosting rounds,
    # leaves the tree models unable to fully exploit the space either.
    # include_minmax=False drops the rolling min/max columns (least
    # informative for steady degradation trends) while keeping mean/std,
    # slope, EWMA, diff, and the expanding-mean residual. Default stays True
    # for backward compatibility with any code built against the wider
    # feature set; pipeline.run_stage() now passes False.
    include_minmax: bool = True
    sensor_cols: list = field(default_factory=list)
    op_cols: list = field(default_factory=lambda: ["op_setting_1", "op_setting_2", "op_setting_3"])
    scaler_: object = None
    dropped_sensors_: list = field(default_factory=list)
    kept_sensor_cols_: list = field(default_factory=list)
    feature_cols_: list = field(default_factory=list)
    kmeans_: object = None
    condition_stats_: dict = None  # cluster_id -> {sensor_col: (mean, std)}

    # ---------- fit ----------

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        all_sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
        self.sensor_cols = all_sensor_cols

        sensor_std = train_df[all_sensor_cols].std()
        self.dropped_sensors_ = sensor_std[sensor_std < self.near_constant_std_threshold].index.tolist()
        self.kept_sensor_cols_ = [c for c in all_sensor_cols if c not in self.dropped_sensors_]

        if self.condition_aware:
            self.kmeans_ = KMeans(n_clusters=self.n_conditions, random_state=0, n_init=10)
            clusters = self.kmeans_.fit_predict(train_df[self.op_cols].values)
            self.condition_stats_ = {}
            tmp = train_df[self.kept_sensor_cols_].copy()
            tmp["_cluster"] = clusters
            for cid, g in tmp.groupby("_cluster"):
                self.condition_stats_[cid] = {
                    col: (g[col].mean(), g[col].std() if g[col].std() > 1e-6 else 1.0)
                    for col in self.kept_sensor_cols_
                }

        # Build engineered features on train, then fit scaler on the resulting
        # feature matrix (fit only, never on val/test).
        raw_features = self._build_raw_features(train_df)
        self.feature_cols_ = [c for c in raw_features.columns if c not in ("engine_id", "cycle")]

        self.scaler_ = StandardScaler()
        self.scaler_.fit(raw_features[self.feature_cols_].values)

        return self

    # ---------- internal: causal feature construction ----------

    def _build_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all engineered (unscaled) features for every row of df.
        Must be called per-engine internally so rolling windows never cross
        engine boundaries; df here may contain multiple engines."""
        parts = []
        for eid, g in df.groupby("engine_id", sort=False):
            parts.append(self._build_raw_features_single_engine(g))
        return pd.concat(parts, ignore_index=True)

    def _build_raw_features_single_engine(self, g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("cycle").reset_index(drop=True)
        sensors = g[self.kept_sensor_cols_]
        cols = {}

        # 1. current-cycle raw sensor values
        for col in self.kept_sensor_cols_:
            cols[f"{col}_cur"] = sensors[col].values

        # 2. trailing rolling mean/std (+ optional min/max) per window
        for w in self.window_sizes:
            roll = sensors.rolling(window=w, min_periods=1)
            mean, std = roll.mean(), roll.std().fillna(0.0)
            if self.include_minmax:
                mn, mx = roll.min(), roll.max()
            for col in self.kept_sensor_cols_:
                cols[f"{col}_mean_w{w}"] = mean[col].values
                cols[f"{col}_std_w{w}"] = std[col].values
                if self.include_minmax:
                    cols[f"{col}_min_w{w}"] = mn[col].values
                    cols[f"{col}_max_w{w}"] = mx[col].values

        # 3. trailing slope over largest window (value - value w cycles ago, /w)
        w_slope = max(self.window_sizes)
        slope = (sensors - sensors.shift(w_slope)) / w_slope
        for col in self.kept_sensor_cols_:
            cols[f"{col}_slope_w{w_slope}"] = slope[col].fillna(0.0).values

        # 4. EWMA (causal by construction — only uses past + current)
        ewma = sensors.ewm(span=10, adjust=False).mean()
        for col in self.kept_sensor_cols_:
            cols[f"{col}_ewma"] = ewma[col].values

        # 5. first difference (trailing, current - previous)
        diff1 = sensors.diff().fillna(0.0)
        for col in self.kept_sensor_cols_:
            cols[f"{col}_diff1"] = diff1[col].values

        # 6. residual vs. engine's own trailing expanding mean (causal proxy
        #    for a condition-normalized residual)
        expanding_mean = sensors.expanding(min_periods=1).mean()
        for col in self.kept_sensor_cols_:
            cols[f"{col}_resid_vs_expanding_mean"] = (sensors[col] - expanding_mean[col]).values

        # 7. (optional) condition-normalized residual: sensor value minus the
        #    per-operating-regime mean/std, where regimes were discovered via
        #    KMeans fit on TRAIN op_settings only. Uses only the current row's
        #    own operating settings -- no future information.
        if self.condition_aware and self.kmeans_ is not None:
            row_clusters = self.kmeans_.predict(g[self.op_cols].values)
            for col in self.kept_sensor_cols_:
                means = np.array([self.condition_stats_[c][col][0] for c in row_clusters])
                stds = np.array([self.condition_stats_[c][col][1] for c in row_clusters])
                cols[f"{col}_cond_resid"] = (sensors[col].values - means) / stds

        out = pd.concat(
            [pd.DataFrame({"engine_id": g["engine_id"], "cycle": g["cycle"]}), pd.DataFrame(cols)],
            axis=1,
        )
        return out

    # ---------- transform ----------

    def transform_engine(self, engine_history: pd.DataFrame, at_cycle: int | None = None) -> pd.DataFrame:
        if self.scaler_ is None:
            raise RuntimeError("FeaturePipeline must be fit() before transform_engine()")

        engine_history = engine_history.sort_values("cycle").reset_index(drop=True)
        raw = self._build_raw_features_single_engine(engine_history)

        scaled_values = self.scaler_.transform(raw[self.feature_cols_].values)
        scaled = pd.DataFrame(scaled_values, columns=self.feature_cols_)
        result = pd.concat([raw[["engine_id", "cycle"]].reset_index(drop=True), scaled], axis=1)

        if at_cycle is not None:
            result = result[result["cycle"] == at_cycle].reset_index(drop=True)

        return result

    def assign_condition(self, df: pd.DataFrame) -> np.ndarray:
        """Return the operating-regime cluster id for each row (condition_aware only)."""
        if not self.condition_aware or self.kmeans_ is None:
            raise RuntimeError("assign_condition() requires condition_aware=True and a fitted pipeline")
        return self.kmeans_.predict(df[self.op_cols].values)

    def transform_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for eid, g in df.groupby("engine_id", sort=False):
            parts.append(self.transform_engine(g))
        return pd.concat(parts, ignore_index=True)


def causality_test(pipeline: FeaturePipeline, engine_history: pd.DataFrame, at_cycle: int) -> bool:
    """Mutate rows AFTER at_cycle and assert the feature row for at_cycle is unchanged."""
    engine_history = engine_history.sort_values("cycle").reset_index(drop=True)

    features_before = pipeline.transform_engine(engine_history, at_cycle=at_cycle)

    corrupted = engine_history.copy()
    future_mask = corrupted["cycle"] > at_cycle
    sensor_cols = [c for c in corrupted.columns if c.startswith("sensor_")]
    corrupted.loc[future_mask, sensor_cols] = corrupted.loc[future_mask, sensor_cols] + 999.0

    features_after = pipeline.transform_engine(corrupted, at_cycle=at_cycle)

    common_cols = [c for c in features_before.columns if c not in ("engine_id", "cycle")]
    return np.allclose(
        features_before[common_cols].values,
        features_after[common_cols].values,
        equal_nan=True,
    )
