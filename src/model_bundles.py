"""
model_bundles.py
-----------------
Plain container classes that get *pickled* (via joblib) as part of the
notebook's exported artifacts (rul_suite.pkl, clf_suite.pkl,
anomaly_suite.pkl) and later *unpickled* by app.py.

These classes used to be defined inline inside the training notebook.
That is fine for running the notebook itself, but it breaks the app:
joblib/pickle records a class's location as whatever module it was
defined in, and inside a notebook kernel that module is "__main__".
When app.py later calls joblib.load(...), Python looks for
sys.modules["__main__"].RulModelBundle -- but app.py's own __main__ is
a different program that never defined that class, so unpickling fails
with:

    AttributeError: module '__main__' has no attribute 'RulModelBundle'

Moving just these three data-holding classes here (and having both the
notebook and app.py import them from here) fixes that: the pickled
objects now point at "src.model_bundles.RulModelBundle", a real,
importable module both sides agree on. All the actual model-fitting /
training / selection logic stays in the notebook, unchanged -- nothing
about *how* models are trained moves out of it, only these three
serialization containers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RulModelBundle:
    """Holds the four RUL regression baselines (ridge, ridge_poly,
    random_forest, gradient_boosting) so predict/evaluate can address any of
    them by name."""

    def __init__(self, models, poly_cols, feat_cols):
        self.models = models            # {name: fitted estimator}
        self.poly_cols = poly_cols       # columns ridge_poly was fit on
        self.feat_cols = feat_cols

    def predict(self, name, X):
        if name == "ridge_poly":
            return self.models[name].predict(X[self.poly_cols].values)
        return self.models[name].predict(X[self.feat_cols].values)

    def evaluate(self, name, X, y_true):
        y_pred = self.predict(name, X)
        y_true_arr = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
        return {
            "MAE": mean_absolute_error(y_true_arr, y_pred),
            "RMSE": mean_squared_error(y_true_arr, y_pred) ** 0.5,
            "R2": r2_score(y_true_arr, y_pred),
        }

    def evaluate_by_life_region(self, name, X, y_true):
        """Break results into early-life / mid-life / near-failure using true
        RUL terciles (near-failure = low RUL)."""
        y_pred = self.predict(name, X)
        y_true_arr = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
        q1, q2 = np.quantile(y_true_arr, [1 / 3, 2 / 3])
        region = np.where(y_true_arr <= q1, "near_failure",
                  np.where(y_true_arr <= q2, "mid_life", "early_life"))
        rows = []
        for r in ["early_life", "mid_life", "near_failure"]:
            mask = region == r
            if mask.sum() == 0:
                continue
            rows.append({
                "region": r, "n": int(mask.sum()),
                "MAE": mean_absolute_error(y_true_arr[mask], y_pred[mask]),
                "RMSE": mean_squared_error(y_true_arr[mask], y_pred[mask]) ** 0.5,
            })
        return pd.DataFrame(rows)


class AnomalyDetectorBundle:
    """Holds the four fitted detectors + each one's validation score
    distribution (for percentile lookups)."""

    def __init__(self, detectors):
        self.detectors = detectors  # {name: fitted detector/PCA}
        self.val_score_distributions = {}

    def score(self, name, X):
        Xv = X.values
        det = self.detectors[name]
        if name in ("isolation_forest", "one_class_svm"):
            return -det.decision_function(Xv)          # larger = more normal -> flip
        if name == "lof":
            return -det.score_samples(Xv)               # larger = more normal -> flip
        if name == "pca_reconstruction":
            recon = det.inverse_transform(det.transform(Xv))
            return np.mean((Xv - recon) ** 2, axis=1)    # already larger = more abnormal
        raise ValueError(f"unknown detector: {name}")

    def fit_validation_distribution(self, name, X_val):
        self.val_score_distributions[name] = np.sort(self.score(name, X_val))

    def score_percentile(self, name, raw_score):
        dist = self.val_score_distributions[name]
        raw_score = np.atleast_1d(raw_score)
        return np.searchsorted(dist, raw_score) / len(dist) * 100.0


class ClassifierSuiteExport:
    """Adapter exposing the notebook's per-horizon classifier dict + Platt-
    calibrated wrapper through the interface app.py / src.decision.
    PrognosticsSystem expects: predict_proba(h, name, X),
    predict_proba_calibrated(h, name, X), and a calibrated_models_ dict."""

    def __init__(self, models_by_horizon, calibrated_by_horizon, chosen_model_name):
        self.models_ = models_by_horizon                 # {h: {name: model}}
        self.chosen_model_name = chosen_model_name
        self.calibrated_models_ = {
            h: {chosen_model_name: cal} for h, cal in calibrated_by_horizon.items()
        }

    def predict_proba(self, horizon, model_name, X):
        return self.models_[horizon][model_name].predict_proba(X.values)[:, 1]

    def predict_proba_calibrated(self, horizon, model_name, X):
        return self.calibrated_models_[horizon][model_name].predict_proba(X.values)[:, 1]
