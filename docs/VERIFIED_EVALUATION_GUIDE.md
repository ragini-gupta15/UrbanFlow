# Verified UrbanFlow test evaluation

This is an additive evaluation path. It does not replace the model checkpoint or overwrite the original target arrays, masks, or historical result files.

## Files added

- `scripts/evaluate_multihorizon_verified.py` — verifies horizon target alignment, evaluates the existing checkpoint, builds a raw-observation mask from METR-LA (`finite` and non-zero), compares against persistence on the same mask, and writes new result/audit files.
- `notebooks/12_verified_evaluation.ipynb` — runs or loads that evaluation and plots MAE/RMSE against persistence.

## Requirements

Run from the UrbanFlow project root with the project's Python environment. Required inputs include:

- `data/raw/metr-la.h5`
- `data/processed/test_sequences.npz`
- `data/processed/weather_test_sequences.npy`
- `data/processed/traffic_scaler.pkl`
- `data/processed/adjacency_matrix.npy`
- `data/processed/metadata.json`
- `data/processed/y_test_{15,30,45,60}min.npy`
- `models/urbanflow_final_gnn_transformer_weather_multihorizon.pth`

If any required file is missing, the script stops before writing metrics. It checks that target rows agree with the future test-sequence values over the verifiable prefix. It does not guess or fabricate missing targets.

## Run

```bash
./.venv/bin/python scripts/evaluate_multihorizon_verified.py
```

Or open `notebooks/12_verified_evaluation.ipynb` and run its cells in order.

## Output

New files under `results/`:

- `urbanflow_verified_test_results.csv`
- `urbanflow_verified_test_audit.json`
- `urbanflow_verified_test_mae.png`
- `urbanflow_verified_test_rmse.png`

The evaluator refuses to overwrite an existing verified CSV/audit pair. Move those files aside manually if an intentional rerun is needed. Historical result files are preserved.

## Important interpretation

The evaluation mask uses raw METR-LA values that are finite and non-zero, so interpolated values are excluded. Historical metrics may have used different masks or sample counts; do not compare them as if they were measured under identical conditions. This evaluation alone does not prove that weather improves the model.
