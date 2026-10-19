# UrbanFlow — Verified Test Results

## Evaluation setup

UrbanFlow forecasts traffic speeds across 207 METR-LA sensors at four horizons:
15, 30, 45, and 60 minutes.

- Historical input: 12 time steps at 5-minute intervals.
- Evaluation: held-out test sequences.
- Evaluation mask: raw METR-LA observations that are finite and non-zero.
- Interpolated values are excluded.
- Baseline: persistence, which carries forward the last observed speed.
- Units: mean absolute error (MAE) and root mean squared error (RMSE), in mph.

Source: `results/urbanflow_verified_test_results.csv`.

## Results

| Horizon | Model MAE (mph) | Persistence MAE (mph) | MAE change vs. persistence | Model RMSE (mph) | Persistence RMSE (mph) |
|---|---:|---:|---:|---:|---:|
| 15 min | 3.7314 | 3.4835 | 7.12% worse | 6.5828 | 6.3741 |
| 30 min | 4.2100 | 4.1759 | 0.82% worse | 7.5942 | 7.9834 |
| 45 min | 4.6255 | 4.7789 | 3.21% better | 8.4162 | 9.2227 |
| 60 min | 4.9884 | 5.3316 | 6.44% better | 9.0749 | 10.2637 |

## Interpretation

The raw model does not outperform persistence at every horizon. Persistence has lower MAE at 15 and 30 minutes, while the raw model has lower MAE and RMSE at 45 and 60 minutes.

These results suggest that the model is more useful relative to persistence at the longer evaluated horizons. They do not establish that weather features independently improve accuracy, nor do they demonstrate live traffic-monitoring performance.

The reported 30-minute model–persistence blend is a separate experiment and should not be conflated with these raw-model results.

## Reproducibility

The verified evaluation script is `scripts/evaluate_multihorizon_verified.py`. The procedure and input requirements are documented in `docs/VERIFIED_EVALUATION_GUIDE.md`.

The reported metrics depend on the specified observation mask and evaluation procedure. Results using different masks or sample populations should not be compared as if they were measured identically.
