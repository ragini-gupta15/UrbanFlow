# UrbanFlow — Consolidated Model Comparison

## Purpose and scope

This report summarizes the verified original UrbanFlow final-model benchmark
alongside a separate matched-test comparison of graph-aware weather,
traffic-only, and persistence experimental approaches. Experimental results
do not replace the original model benchmark.

All MAE and RMSE values are in mph. Lower is better.

## 1. Original UrbanFlow final-model benchmark

| Horizon | UrbanFlow MAE | UrbanFlow RMSE | Persistence MAE | Persistence RMSE | MAE improvement vs persistence |
|---|---:|---:|---:|---:|---:|
| 15 min | 3.7314 | 6.5828 | 3.4835 | 6.3741 | -7.12% |
| 30 min | 4.2100 | 7.5942 | 4.1759 | 7.9834 | -0.82% |
| 45 min | 4.6255 | 8.4162 | 4.7789 | 9.2227 | 3.21% |
| 60 min | 4.9884 | 9.0749 | 5.3316 | 10.2637 | 6.44% |

The original model is worse than persistence on MAE at 15 and 30 minutes,
and better on MAE at 45 and 60 minutes. Its benchmark uses raw METR-LA
observations that are finite and non-zero; interpolated values are excluded.

Source: `results/urbanflow_verified_test_results.csv`.

## 2. Matched-test experimental comparison

| Horizon | Model | MAE (mph) | RMSE (mph) | Valid observations |
|---|---|---:|---:|---:|
| 15min | Graph-aware weather model | 3.2917 | 5.7494 | 1,243,168 |
| 15min | Traffic-only experimental model | 3.1449 | 5.8447 | 1,243,168 |
| 15min | Persistence baseline | 3.4835 | 6.3741 | 1,243,168 |
| 30min | Graph-aware weather model | 3.9166 | 7.0450 | 1,243,750 |
| 30min | Traffic-only experimental model | 3.7703 | 7.2160 | 1,243,750 |
| 30min | Persistence baseline | 4.1759 | 7.9834 | 1,243,750 |
| 45min | Graph-aware weather model | 4.4506 | 8.0140 | 1,241,932 |
| 45min | Traffic-only experimental model | 4.2766 | 8.2108 | 1,241,932 |
| 45min | Persistence baseline | 4.7789 | 9.2227 | 1,241,932 |
| 60min | Graph-aware weather model | 4.9425 | 8.7067 | 1,241,314 |
| 60min | Traffic-only experimental model | 4.7462 | 8.9757 | 1,241,314 |
| 60min | Persistence baseline | 5.3316 | 10.2637 | 1,241,314 |

Source: `results/experiments/weather_ablation_matched_test_scaled_all_horizons_mph.csv`.

## 3. Interpretation

- In the matched-test experiment, the traffic-only model has the lowest MAE
  at all four horizons.
- The graph-aware weather model has the lowest RMSE at all four horizons.
- Both experimental models outperform persistence on both metrics in this
  matched-test evaluation.
- The original UrbanFlow model remains a separate benchmark and should not
  be ranked as if all model rows came from the same experimental run.
- These results do not establish that adding weather features caused better
  performance. The weather and traffic-only models are separately trained
  checkpoints; causal attribution requires a carefully controlled ablation.

## 4. Limitations

- These are historical held-out test results, not live traffic forecasts.
- Do not mix the experimental results with the original model's benchmark
  without clearly labeling their different evaluation provenance.
- Model choice depends on the metric: MAE emphasizes typical absolute error,
  while RMSE penalizes larger errors more strongly.

## 5. Reproducibility artifacts

- Original benchmark: `results/urbanflow_verified_test_results.csv`
- Matched experimental comparison:
  `results/experiments/weather_ablation_matched_test_scaled_all_horizons_mph.csv`
- Experimental report:
  `docs/EXPERIMENTAL_WEATHER_GRAPH_MATCHED_TEST_RESULTS.md`


## 6. Validation-selected blend: held-out test evaluation

A separate held-out test evaluation combined the existing UrbanFlow model
with the persistence baseline. The blend weights were selected on validation
data and fixed before test evaluation. The test set was not used to select or
retune weights.

| Horizon | Model weight | Persistence weight | Model MAE (mph) | Persistence MAE (mph) | Blend MAE (mph) | Blend RMSE (mph) |
|---|---:|---:|---:|---:|---:|---:|
| 15min | 0.35 | 0.65 | 3.7314 | 3.4835 | 3.3049 | 5.9740 |
| 30min | 0.45 | 0.55 | 4.2100 | 4.1759 | 3.9160 | 7.3297 |
| 45min | 0.55 | 0.45 | 4.6255 | 4.7789 | 4.4157 | 8.2581 |
| 60min | 0.60 | 0.40 | 4.9884 | 5.3316 | 4.8412 | 8.9702 |

The blend has lower MAE and RMSE than both standalone baselines at each
reported horizon in this evaluation. This is a result on this specific
historical test split, not evidence of live-traffic performance or a guarantee
of generalization to other datasets.

### Evaluation safeguards and artifacts

- Test target alignment was checked against shifted stored input sequences;
  maximum absolute scaled difference was 0.0 for each verified prefix.
- At 30 minutes, 6,832 of the 6,838 evaluated target rows were in the
  independently verified shifted-sequence prefix. The audit records this
  distinction explicitly.
- Metrics use finite, nonzero raw METR-LA observations in model sensor order.
- The original benchmark file was not modified.
- Test results: `results/urbanflow_blend_all_horizons_test.csv`
- Test audit: `results/urbanflow_blend_all_horizons_test_audit.json`
- Validation weight-grid source:
  `results/urbanflow_validation_blend_all_horizons.csv`
- Evaluator: `scripts/evaluate_test_blend_all_horizons.py`
