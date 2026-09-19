# AI-Powered Digital Twin for Sustainable Textile Manufacturing

This existing Streamlit application combines a real textile weaving dataset, leakage-audited Random Forest models, a SimPy discrete-event digital twin, estimated sustainability indicators, recommendations, and an optional Gemini copilot.

## Problem and objectives

Textile production moves through Warping, Sizing, Weaving, and Quality Inspection. The dashboard helps inspect production records, estimate rejection risk and production volume, simulate factory flow, compare what-if settings, identify congestion, and assess resource indicators.

## Dataset and preprocessing

The primary dataset is `data/weaving_dataset_full.csv` with 121,148 raw records. `data/weaving_rejection_dataset.csv` remains an optional secondary dataset.

The reproducible preprocessing pipeline removes the export index, removes rows marked `TOTAL` as aggregate rows, normalizes `na`/`n/a`/`null`/blank sentinels, removes exact duplicate source rows, converts numeric fields, imputes missing numeric values with medians and categorical values with modes, and creates textile features such as cover factors and total fabric density.

Current primary-dataset accounting is 121,148 raw rows to 99,126 analytical rows: 22,022 aggregate rows removed, 15,218 missing sentinels normalized and imputed, and zero exact duplicates or invalid rows removed. The dashboard exposes this audit trail.

## Machine learning methodology

The classification target is `Has_Rejection`, derived from `Rej_and_cut_Piece > 0` in the full dataset. The regression target is `Total_pdn_per_order`. The train/test split happens before fitting separate `StandardScaler` instances. Random Forest models use a fixed random seed; the classifier uses class balancing because positive rejection cases are rare.

Target-derived and post-outcome fields are excluded, including rejection fields, production targets, `Total_pdn_m/c`, and `Rec_Beam_length(yds)`. Required material and beam quantities remain only as planning features under the assumption that they are available before production.

Measured deterministic holdout results from the current run:

| Model | Metrics |
|---|---|
| Rejection classifier | Accuracy 97.92%, precision 6.42%, recall 87.50%, F1 11.97% |
| Production regressor | MAE 1,220.11 yards, RMSE 17,306.51 yards, R2 0.017 |

The precision and regression score are limitations of the available data, not metrics optimized or fabricated for presentation. Model artifacts include dataset name and fingerprint, feature list, preprocessing version, timestamp, targets, and metrics. A changed dataset invalidates the old artifact.

## Digital twin and what-if simulation

The SimPy model preserves the flow `Raw Material -> Warping -> Sizing -> Weaving -> Quality Inspection -> Finished Fabric`. It records batches, processing and waiting time, queues, utilization, event logs, production, rejection, and bottlenecks. What-if simulation compares baseline and altered loom count, weaving time, defect factor, and sizing capacity using the same measured outputs.

ML predictions are decision-support signals passed to recommendations and AI context. They are not treated as causal inputs to the SimPy model. Simulation outputs remain simulation results.

## Sustainability and Eco Score

Energy, water, material waste, and carbon are **estimated indicators**, not measurements from the dataset. The assumptions are:

- Energy: stage machine power multiplied by active hours plus 15% standby power during idle hours.
- Water: total produced yards multiplied by 1.2 L/yard.
- Waste: rejected yards plus 2% selvage trim, multiplied by 0.18 kg/yard fabric weight.
- Carbon: energy multiplied by 0.52 kg CO2e/kWh plus 0.0003 kg CO2e/L water processing.

Eco Score is a 0-100 calculation with four equal components worth 25 points each: Energy Efficiency, Waste Minimization, Carbon Intensity, and Throughput Health. It is not an ML prediction.

## Recommendations and AI Copilot

Recommendations use current utilization, queue delay, rejection, energy intensity, waste, Eco Score, live ML predictions, and top model-associated features. Language is intentionally non-causal: model-associated, contributing factor, or simulation suggests. Gemini receives current telemetry, ML outputs, model context, and the estimated-data status. Without an API key or when Gemini is unavailable, the offline expert mode remains available.

## Technology and architecture

- Streamlit dashboard: `app.py`
- Pandas and NumPy: loading, cleaning, feature engineering
- Scikit-learn: Random Forest classification and regression
- SimPy: discrete-event simulation
- Plotly: dashboard charts
- Joblib: model persistence and metadata
- Gemini SDKs: optional AI copilot with offline fallback

See [AUDIT.md](AUDIT.md) for the requirement-by-requirement audit and leakage decision.

## Run locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Train/retrain models from the ML Prediction tab. The end-to-end checks can be run with:

```bash
.venv\Scripts\python.exe test_pipeline.py
```

## Limitations and future work

The digital twin is a calibrated software simulation, not a real-time IoT system. Sustainability values are engineering estimates, not measured energy, water, or carbon. The current feature set does not establish causal defect drivers, and the regression model has weak holdout explanatory power. Future work should add time/order-aware validation, richer pre-production features, measured utility data, failure scenarios, and production feedback from deployed operations.