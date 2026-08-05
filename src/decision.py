"""
decision.py
-----------
Maintenance decision policy shared between the training notebook and the
Streamlit app.

NOTE ON THE MISSING ORIGINAL FILE
----------------------------------
This file did not exist in the project as uploaded -- app.py imports
`from src.decision import PrognosticsSystem, DecisionPolicy`, but no
src/decision.py (source) shipped with the project, only a stale compiled
src/__pycache__/decision.cpython-314.pyc left over from an earlier version.
That mismatch is *part of* why the app couldn't start: the class doing the
recommend() logic and the PrognosticsSystem wrapper the app depends on
were simply gone.

PrognosticsSystem is reconstructed here to match exactly how the notebook
itself already exercises the same pipeline in its "decision policy" demo
cell (predict RUL -> failure risk from the CALIBRATED classifiers -> raw +
percentile anomaly score -> recommend()), so its behavior is consistent
with what the notebook trains, evaluates, and exports.

DecisionPolicy and recommend() are moved here unchanged from the notebook
(they used to be defined locally there); the notebook now imports them
from this module instead of redefining them, so the exact same code runs
in both places and decision_policy.json (a plain asdict(policy) dump)
round-trips correctly through DecisionPolicy(**policy_dict).
"""

from dataclasses import dataclass

from src.config import HORIZONS


@dataclass
class DecisionPolicy:
    rul_lower_bound_stop: float
    rul_lower_bound_inspect: float
    failure_prob_stop_h10: float
    failure_prob_inspect_h10: float
    anomaly_percentile_inspect: float


def recommend(rul, failure_risk, anomaly, policy: DecisionPolicy):
    """rul: {"point","lower","upper"}; failure_risk: {horizon: proba};
    anomaly: {"raw","percentile"}. Returns action + auditable trigger reason."""
    reasons = []
    rul_stop_fired = rul["lower"] <= policy.rul_lower_bound_stop
    risk_stop_fired = failure_risk[10] >= policy.failure_prob_stop_h10

    if rul_stop_fired:
        reasons.append(f"RUL lower bound ({rul['lower']:.1f}) <= critical threshold ({policy.rul_lower_bound_stop})")
    if risk_stop_fired:
        reasons.append(f"P(failure in 10 cycles)={failure_risk[10]:.2f} >= STOP threshold ({policy.failure_prob_stop_h10})")
    if reasons:
        confidence = max(failure_risk[10], 1.0 if rul_stop_fired else 0.0)
        return {"action": "STOP", "trigger_reason": "; ".join(reasons), "confidence": confidence}

    rul_inspect_fired = rul["lower"] <= policy.rul_lower_bound_inspect
    risk_inspect_fired = failure_risk[10] >= policy.failure_prob_inspect_h10
    anomaly_inspect_fired = anomaly["percentile"] >= policy.anomaly_percentile_inspect

    if rul_inspect_fired:
        reasons.append(f"RUL lower bound ({rul['lower']:.1f}) <= inspect threshold ({policy.rul_lower_bound_inspect})")
    if risk_inspect_fired:
        reasons.append(f"P(failure in 10 cycles)={failure_risk[10]:.2f} >= INSPECT threshold ({policy.failure_prob_inspect_h10})")
    if anomaly_inspect_fired:
        reasons.append(f"Anomaly percentile={anomaly['percentile']:.0f} >= INSPECT threshold ({policy.anomaly_percentile_inspect})")
    if anomaly_inspect_fired and not risk_inspect_fired:
        reasons.append("NOTE: anomaly score elevated while supervised failure risk remains low -- disagreement between signals")

    if reasons:
        confidence = max(failure_risk[10], 1.0 if rul_inspect_fired else 0.0,
                          anomaly["percentile"] / 100.0 if anomaly_inspect_fired else 0.0)
        return {"action": "INSPECT", "trigger_reason": "; ".join(reasons), "confidence": confidence}

    return {"action": "CONTINUE", "trigger_reason": "no threshold triggered", "confidence": 1 - failure_risk[10]}


class PrognosticsSystem:
    """Wraps one dataset's exported artifacts (feature pipeline, RUL suite,
    classifier suite, anomaly suite, decision policy) and answers the
    per-(engine, cycle) questions app.py needs: predict_rul, failure_risk,
    anomaly_score, and recommend."""

    def __init__(self, feature_pipeline, rul_suite, rul_model_name,
                 clf_suite, clf_model_name, anomaly_suite, anomaly_detector_name,
                 policy: DecisionPolicy, rul_interval_std=None):
        self.feature_pipeline = feature_pipeline
        self.rul_suite = rul_suite
        self.rul_model_name = rul_model_name
        self.clf_suite = clf_suite
        self.clf_model_name = clf_model_name
        self.anomaly_suite = anomaly_suite
        self.anomaly_detector_name = anomaly_detector_name
        self.policy = policy
        self.rul_interval_std = float(rul_interval_std) if rul_interval_std is not None else 0.0

    def predict_rul(self, X):
        point = float(self.rul_suite.predict(self.rul_model_name, X)[0])
        margin = 1.96 * self.rul_interval_std
        return {"point": point, "lower": max(0.0, point - margin), "upper": point + margin}

    def failure_risk(self, X):
        return {
            h: float(self.clf_suite.predict_proba_calibrated(h, self.clf_model_name, X)[0])
            for h in HORIZONS
        }

    def anomaly_score(self, X):
        raw = self.anomaly_suite.score(self.anomaly_detector_name, X)[0]
        pct = float(self.anomaly_suite.score_percentile(self.anomaly_detector_name, raw)[0])
        return {"raw": float(raw), "percentile": pct}

    def recommend(self, inputs: dict):
        return recommend(inputs["rul"], inputs["failure_risk"], inputs["anomaly"], self.policy)
