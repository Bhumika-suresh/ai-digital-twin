# Implementation audit

| Requirement | Existing implementation | Status | Problem found | Applied fix |
|---|---|---|---|---|
| Primary weaving dataset | `load_data()` selected the rejection CSV first | INCORRECT | Full dataset was not the default | `weaving_dataset_full.csv` is now explicit and selected first |
| Transparent preprocessing | Dropped unnamed columns and duplicates only | PARTIAL | Row changes and `TOTAL`/`na` handling were not reported | Added preprocessing accounting and documented aggregate-row removal |
| Data leakage control | Scaler fit before train/test split | INCORRECT | Test statistics influenced preprocessing | Fit separate scalers on training folds only |
| Rejection model | Random Forest classification | PARTIAL | Severe class imbalance produced zero recall | Added `class_weight="balanced"`; metrics remain test-derived |
| Production model | Random Forest regression | PARTIAL | Feature and target availability was undocumented | Excluded post-outcome fields and saved target/feature metadata |
| Model versioning | Joblib model without provenance | INCORRECT | Old artifacts could be reused silently | Saved dataset fingerprint, target, features, preprocessing version, timestamp, and metrics |
| Digital Twin | SimPy Warping -> Sizing -> Weaving -> Inspection | MATCH | Existing simulation was preserved | ML outputs now reach decision-support recommendations and AI context |
| What-if simulation | Baseline/scenario simulation | MATCH | Existing comparison was retained | End-to-end test covers both runs |
| Sustainability | Formula-based energy/water/waste/carbon estimates | MATCH | Values are estimates, not measurements | UI and README label assumptions explicitly |
| Eco Score | Four 25-point components | MATCH | Formula existed in one module | Kept the single calculation source |
| Bottleneck detection | Utilization and queue metrics | MATCH | Severity thresholds were already present | Preserved and tested |
| Recommendations | Simulation-only rule engine | PARTIAL | Did not use ML outputs and claimed historical causality | Added ML prediction, feature-importance context, and supporting metrics |
| AI Copilot | Gemini plus offline fallback | PARTIAL | Context omitted live ML and estimate status | Added ML outputs, top features, and estimated-data status |
| End-to-end testing | Print-only smoke script | INCORRECT | No assertions and false all-passed message | Added assertion-based pipeline test |

## Data findings

The primary file has 121,148 raw rows. The reproducible preprocessing result is 99,126 analytical rows:

- 1 export index column removed (`H1`), not a business feature.
- 22,022 rows containing `TOTAL` aggregate markers removed.
- 15,218 `na` sentinels normalized to missing values and imputed; these rows were not removed.
- 0 exact duplicate source rows removed.
- 0 invalid rows removed by the current target-preserving rules.
- `Total_pdn_m/c` has 13,630 missing values, but it is excluded as a post-outcome field and is not used as a target.

## Leakage decision

The classifier target is `Rej_and_cut_Piece > 0` through `Has_Rejection`. The regression target is `Total_pdn_per_order` for the full dataset. Target fields and post-outcome fields are excluded, including `Rejection`, `Rej_and_cut_Piece`, `Rejection_Qty`, `Has_Rejection`, `Total_pdn_m/c`, `Total_pdn_per_order`, and `Rec_Beam_length(yds)`. Planning fields such as required grey fabric and required beam length remain only under the assumption that they are available before production; operational deployment should verify that assumption.

## Current measured metrics

Using a deterministic 80/20 split (`random_state=42`) and a training-only scaler:

- Classification: accuracy 97.92%, precision 6.42%, recall 87.50%, F1 11.97%.
- Regression: MAE 1,220.11 yards, RMSE 17,306.51 yards, R2 0.017.

The low precision and regression R2 are limitations of the available features and target distribution, not hidden or fabricated metrics.
