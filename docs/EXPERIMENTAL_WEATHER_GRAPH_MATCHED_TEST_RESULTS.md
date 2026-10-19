# UrbanFlow: Weather + Graph Model — Matched-Test Evaluation

**Status:** Completed
**Evaluation date:** 19 October 2026
**Device:** Apple MPS
**Evaluation type:** Matched multi-horizon test evaluation
**Training performed during evaluation:** No

## 1. Purpose

This experiment compares the weather-aware graph model, a traffic-only
graph-aware counterpart, and a persistence baseline on the same test
samples and valid raw traffic observations.

These experimental results are separate from the main UrbanFlow model
benchmark. They do not replace the main model's verified results.

## 2. Preprocessing correction

The initial evaluation supplied `weather_test_sequences.npy` directly
to a model trained using standardized weather features. This was an
input-preprocessing mismatch, so the initial weather-model metrics are
not valid for model comparison.

The corrected evaluation transforms test weather using the existing
training-fitted scaler at `data/processed/weather_scaler.pkl`.

The transformed test weather has shape `(6838, 12, 6)`, mean `0.0232`,
standard deviation `0.9253`, minimum `-2.1228`, maximum `2.6347`, and
contains only finite values.

No scaler was fitted on test data. Neither model was retrained.

## 3. Evaluation methodology

- Traffic input: 12 time steps across 207 sensors.
- Weather input: 12 time steps across 6 features.
- Forecast horizons: 15, 30, 45, and 60 minutes.
- Both checkpoints loaded with `strict=True`.
- Target prefixes checked against the saved input sequences.
- Valid observations masked using finite, non-zero raw METR-LA values,
  aligned by sensor IDs.
- Predictions and targets inverse-transformed using the project's
  fitted traffic scaler.
- Persistence predicts the last observed traffic frame.
- Metrics: mean absolute error (MAE) and root mean squared error (RMSE),
  in mph.

Each horizon uses the same samples and observation mask for all three
models. The valid observation counts differ between horizons because
the available target arrays have different lengths.

## 4. Corrected matched-test results

Lower values are better. All metrics are in mph.

| Horizon | Model | MAE | RMSE | Valid observations |
|---|---|---:|---:|---:|
| 15 min | Weather + graph | 3.291670 | 5.749414 | 1,243,168 |
| 15 min | Traffic-only | 3.144881 | 5.844688 | 1,243,168 |
| 15 min | Persistence | 3.483514 | 6.374143 | 1,243,168 |
| 30 min | Weather + graph | 3.916554 | 7.045038 | 1,243,750 |
| 30 min | Traffic-only | 3.770267 | 7.215984 | 1,243,750 |
| 30 min | Persistence | 4.175889 | 7.983383 | 1,243,750 |
| 45 min | Weather + graph | 4.450642 | 8.013977 | 1,241,932 |
| 45 min | Traffic-only | 4.276558 | 8.210785 | 1,241,932 |
| 45 min | Persistence | 4.778861 | 9.222697 | 1,241,932 |
| 60 min | Weather + graph | 4.942532 | 8.706656 | 1,241,314 |
| 60 min | Traffic-only | 4.746236 | 8.975672 | 1,241,314 |
| 60 min | Persistence | 5.331637 | 10.263740 | 1,241,314 |

## 5. Interpretation

### Compared with persistence

The weather + graph model reduces MAE by approximately 5.51%,
6.21%, 6.87%, and 7.30% at the 15-, 30-, 45-, and 60-minute
horizons, respectively.

Its RMSE is lower than persistence at every horizon, with approximate
reductions of 9.81%, 11.75%, 13.10%, and 15.17%, respectively.

### Compared with the traffic-only checkpoint

The weather + graph model has lower RMSE at all four horizons, but
its MAE is higher at all four horizons. Therefore, it does not
outperform the traffic-only checkpoint on every metric.

This comparison alone does not establish that weather features caused
the difference: the checkpoints came from separate training runs and
may differ in training dynamics and checkpoint selection.

## 6. Checkpoint provenance

- Weather + graph checkpoint:
  `models/weather_graph_aware_multi_horizon_best.pth`
  - Selected epoch: 6
  - Validation 30-minute MAE: 0.3964211 in the training metric scale
- Traffic-only checkpoint:
  `models/experiments/weather_ablation_traffic_only_best.pth`
  - Selected epoch: 5
  - Validation 30-minute MAE: 0.3807777 in the training metric scale

These validation metrics are reported in their saved training scale;
they should not be directly compared numerically with the mph test metrics.

## 7. Reproducibility artifacts

Corrected full-horizon results:
- `results/experiments/weather_ablation_matched_test_scaled_all_horizons_mph.csv`
- `results/experiments/weather_ablation_matched_test_scaled_all_horizons_audit.json`

Earlier outputs were retained as historical artifacts. Results produced
with unscaled test weather must not be presented as valid benchmark
metrics.

## 8. Limitations and next steps

1. This is a comparison of existing checkpoints, not a controlled
   retraining experiment with matched random seeds and training runs.
2. The results do not establish weather-feature causality.
3. The weather + graph model improves on persistence across all tested
   horizons but has slightly worse MAE than the traffic-only checkpoint.
4. The main UrbanFlow model's benchmark remains separate and unchanged.

A stronger causal ablation would require controlled training conditions
and validation-based checkpoint selection, followed by a single final
test evaluation. No further training was performed as part of this report.
