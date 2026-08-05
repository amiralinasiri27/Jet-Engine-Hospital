# Jet Engine Hospital — NASA C-MAPSS Predictive Maintenance

A multi-task early-warning system for NASA C-MAPSS turbofan engines. For every engine at every
cycle, the system produces:

1. **Remaining Useful Life (RUL)** — a point estimate with a 95% prediction interval, in cycles.
2. **Failure-horizon risk** — calibrated probabilities of failure within 10, 20, and 30 cycles.
3. **Anomaly score** — an unsupervised health signal, fitted without ever using failure labels.

These three signals are fused into one auditable maintenance recommendation —
**CONTINUE / INSPECT / STOP** — with a human-readable trigger reason and a confidence value.

Built and validated on **FD001** (single operating condition, single fault mode) as the
foundation stage, then stress-tested on **FD003** (single operating condition, two simultaneous
fault modes) as the Stage-2 challenge, using one identical, leakage-safe, engine-level
evaluation protocol throughout.

## Project layout

```
.
├── notebooks/main.ipynb
├── CMAPSSData/              # Raw C-MAPSS text files (train/test/RUL per subset)
├── artifacts/               # Exported per-dataset artifacts consumed by the app
│   ├── FD001/
│   └── FD003/
├── src/
│   ├── config.py             # Paths, seed, horizons, cost policy
│   ├── data_loading.py       # Load train/test/RUL text files with explicit column names
│   ├── labels.py              # RUL target and horizon classification label construction
│   ├── splits.py              # Engine-level train/validation split + disjointness checks
│   ├── features.py            # Causal (leakage-safe) feature pipeline + causality test
│   ├── model_bundles.py       # Picklable containers for RUL/classifier/anomaly model suites
│   ├── decision.py            # DecisionPolicy, recommend(), and PrognosticsSystem
│   └── plotting.py            # Shared plotting functions used by the notebook
├── app/
│   ├── app.py                 # Streamlit dashboard (reads exported artifacts only)
│   ├── requirements.txt
│   └── assets/
└── report/
    └── report.pdf
```

## Core design constraint

The train/validation/test split is made **by engine ID**, before any preprocessing is fit, any
feature is selected, any window is built, or any threshold is tuned. Every window belonging to
one engine lives in exactly one split. This is enforced in code (`splits.assert_disjoint`) and
verified by a dedicated causality test that perturbs future cycles and asserts that features
computed at the current cycle do not change.

## Models

| Task | Baselines compared | Selected |
|---|---|---|
| RUL regression | Ridge, Ridge+Polynomial, Random Forest, Gradient Boosting | **Gradient Boosting** |
| Failure-horizon classification | Logistic Regression, Gradient Boosting | **Logistic Regression** (Platt-calibrated) |
| Anomaly detection | Isolation Forest, LOF, One-Class SVM, PCA reconstruction | **LOF** (FD001) / **Isolation Forest** (FD003) |

Selection is always made on **validation** data using the metric appropriate to the task (MAE
for RUL, mean PR-AUC for classification, Spearman correlation with RUL for anomaly detection) —
never on test.

## Reproducing the results

1. Place the official C-MAPSS text files (`train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`)
   under `CMAPSSData/`.
2. Run `main.ipynb` top-to-bottom. It is fully reproducible: fixed random seed (42), documented
   data paths, and deterministic preprocessing throughout.
3. The notebook exports everything the app needs under `artifacts/<DATASET>/`:
   `feature_pipeline.pkl`, `rul_suite.pkl`, `clf_suite.pkl`, `anomaly_suite.pkl`,
   `decision_policy.json`, `metadata.json`.

## Running the dashboard

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

The app loads only the exported artifacts — it never re-runs training — so its inference path
is guaranteed to match the notebook exactly (**app parity**). Select a dataset and an engine ID
in the sidebar to see the full evidence chain: sensor timeline, RUL card, failure-risk card,
anomaly card, and the resulting action with its trigger reason.

## Evaluation protocol

- **Engine-level bootstrap** (not row-level) for all 95% confidence intervals.
- **Per-engine distributions**, not just fleet averages, so no single catastrophic engine is
  hidden inside an aggregate metric.
- **Cost-based threshold tuning**: classification and decision thresholds are chosen on
  validation engines by minimizing an asymmetric cost
  (`c_miss=10 > c_late=3 > c_early=1`), not the default 0.5 cut-off.
- **Warning lead time**: for each held-out engine, the first persistent alert cycle is compared
  against the true failure cycle to compute lead time, late-warning delay, early-warning burden,
  and miss rate.

## Known limitations

- The Gaussian RUL prediction interval undercovers its nominal 95% target on both datasets
  (empirical coverage ≈ 51% on FD001, ≈ 38% on FD003), concentrated in the early-life region
  where residual variance is highest. A conformal or per-life-region interval is the natural
  next step.
- FD003's RUL regression is markedly harder than FD001's (R² drops from 0.69 to 0.37) because
  two fault modes share one training set with no fault-mode label to separate them.
- FD004 (six operating conditions + two fault modes) is scoped out of this submission as the
  bonus benchmark — see the technical report's Challenges section for the reasoning.

## Technical report

See `report/report.tex` for the full write-up, including problem framing, data audit, feature
engineering, model mathematics, uncertainty analysis, early-warning cost analysis, the decision
policy, and a dedicated Challenges section.

## Data source

NASA Ames Prognostics Data Repository — C-MAPSS Jet Engine Simulated Data.
