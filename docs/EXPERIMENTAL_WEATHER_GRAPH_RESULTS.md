# Experimental Graph-Aware Weather Model — Test Results

## Purpose

This document reports a separate graph-aware, weather-aware multi-horizon model experiment. These results do not replace the original UrbanFlow final-model benchmark.

## Evaluation setup

- Traffic network: 207 METR-LA sensors.
- Historical input: 12 time steps at 5-minute intervals.
- Forecast horizons: 15, 30, 45, and 60 minutes.
- Baseline: persistence, carrying forward the last observed traffic speed.
- Metrics: mean absolute error (MAE) and root mean squared error (RMSE), in mph.

Source: `results/experiments/weather_graph_aware_test_comparison.csv`.

## Test results

| Horizon | Experimental MAE (mph) | Persistence MAE (mph) | MAE improvement | RMSE improvement |
|---|---:|---:|---:|---:|
| 15 min | 3.2917 | 3.4835 | 5.51% | 9.80% |
| 30 min | 3.9307 | 4.1883 | 6.15% | 11.71% |
| 45 min | 4.4506 | 4.7789 | 6.87% | 13.11% |
| 60 min | 4.9425 | 5.3316 | 7.30% | 15.17% |

The model and persistence baseline use the same number of valid observations within each horizon comparison.

## Interpretation and limitations

In this experiment, the model has lower MAE and RMSE than persistence at all four forecast horizons. The relative improvement increases with the forecast horizon.

This is a result for the separate experimental checkpoint. It must not be presented as the original final model's performance. The comparison also does not isolate the causal contribution of weather features; that requires a controlled ablation study.

These are historical held-out test results, not live traffic forecasts.

## Reproducibility

The detailed comparison is stored in `results/experiments/weather_graph_aware_test_comparison.csv`. Related training, validation, and corrected test outputs are retained separately under `results/experiments/`.
