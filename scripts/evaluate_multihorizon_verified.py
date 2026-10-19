#!/usr/bin/env python3
"""Evaluate UrbanFlow on genuinely observed METR-LA targets, without retraining.

Inputs are read-only. Outputs are written to new filenames under results/.
This script deliberately refuses to evaluate if target alignment cannot be
verified against the stored test sequences.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "metr-la.h5"
MODEL_PATH = ROOT / "models" / "urbanflow_final_gnn_transformer_weather_multihorizon.pth"
RESULTS = ROOT / "results"
HORIZONS = (15, 30, 45, 60)
INPUT_STEPS = 12
SAMPLE_MINUTES = 5
N_SENSORS = 207
BATCH_SIZE = 64


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path.relative_to(ROOT)}")


def load_raw_observation_mask(sensor_ids: list[str], target_positions: np.ndarray) -> np.ndarray:
    """Return raw-observation validity at target rows, aligned to model sensor order."""
    require(RAW)
    with h5py.File(RAW, "r") as h5:
        raw_values = np.asarray(h5["df/block0_values"][:], dtype=np.float32)
        raw_sensor_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                          for x in h5["df/block0_items"][:]]

    if len(raw_sensor_ids) != raw_values.shape[1]:
        raise ValueError("Raw METR-LA sensor-ID count does not match its data columns.")
    if len(sensor_ids) != N_SENSORS or len(set(sensor_ids)) != N_SENSORS:
        raise ValueError("metadata.json must contain 207 unique sensor IDs in model order.")
    if set(sensor_ids) != set(raw_sensor_ids):
        raise ValueError("Sensor IDs in metadata.json do not match raw METR-LA sensor IDs.")
    if target_positions.size == 0 or target_positions.min() < 0 or target_positions.max() >= len(raw_values):
        raise ValueError("Computed target positions fall outside raw METR-LA timestamps.")

    raw_col = {sid: i for i, sid in enumerate(raw_sensor_ids)}
    col_order = [raw_col[sid] for sid in sensor_ids]
    selected = raw_values[target_positions][:, col_order]
    # The documented preprocessing treats zero-speed values as invalid.
    return np.isfinite(selected) & (selected != 0)


def metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    valid = mask & np.isfinite(pred) & np.isfinite(truth)
    count = int(valid.sum())
    if count == 0:
        raise ValueError("No valid observed targets remain for metric calculation.")
    error = pred[valid] - truth[valid]
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(np.square(error)))), count


def main() -> int:
    required = [
        DATA / "test_sequences.npz",
        DATA / "weather_test_sequences.npy",
        DATA / "traffic_scaler.pkl",
        DATA / "adjacency_matrix.npy",
        DATA / "metadata.json",
        RAW,
        MODEL_PATH,
    ] + [DATA / f"y_test_{h}min.npy" for h in HORIZONS]
    for path in required:
        require(path)

    with np.load(DATA / "test_sequences.npz") as archive:
        X = np.asarray(archive["X"], dtype=np.float32)
    W = np.asarray(np.load(DATA / "weather_test_sequences.npy"), dtype=np.float32)
    metadata = json.loads((DATA / "metadata.json").read_text())
    sensor_ids = [str(x) for x in metadata["sensor_ids"]]
    scaler = joblib.load(DATA / "traffic_scaler.pkl")
    adjacency = np.asarray(np.load(DATA / "adjacency_matrix.npy"), dtype=np.float32)

    if X.ndim != 3 or X.shape[1:] != (INPUT_STEPS, N_SENSORS):
        raise ValueError(f"Unexpected test traffic shape: {X.shape}")
    if W.shape != (len(X), INPUT_STEPS, 6):
        raise ValueError(f"Weather/test alignment mismatch: X={X.shape}, W={W.shape}")
    if adjacency.shape != (N_SENSORS, N_SENSORS) or not np.isfinite(adjacency).all():
        raise ValueError("Adjacency matrix has invalid shape or non-finite values.")

    # Chronological split boundary in Notebook 02.
    with h5py.File(RAW, "r") as h5:
        raw_length = int(h5["df/block0_values"].shape[0])
    test_start = int(raw_length * 0.80)

    A = adjacency + np.eye(N_SENSORS, dtype=np.float32)
    degree = A.sum(axis=1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    positive = degree > 0
    inv_sqrt[positive] = degree[positive] ** -0.5
    graph = torch.as_tensor(inv_sqrt[:, None] * A * inv_sqrt[None, :], dtype=torch.float32)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Import the exact architecture used by the API/checkpoint.
    sys.path.insert(0, str(ROOT))
    from src.model import UrbanFlowFinalModel

    model = UrbanFlowFinalModel().to(device)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    except TypeError:  # compatibility with older PyTorch
        checkpoint = torch.load(MODEL_PATH, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    graph = graph.to(device)

    rows = []
    alignment_checks = []
    for horizon in HORIZONS:
        steps = horizon // SAMPLE_MINUTES
        Y = np.asarray(np.load(DATA / f"y_test_{horizon}min.npy"), dtype=np.float32)
        if Y.ndim != 2 or Y.shape[1] != N_SENSORS:
            raise ValueError(f"Unexpected {horizon}-min target shape: {Y.shape}")
        n = min(len(X), len(W), len(Y))
        if n == 0:
            raise ValueError(f"No samples available for {horizon}-minute evaluation.")

        # Verify the saved target convention where a future sequence is available.
        check_n = min(n, max(0, len(X) - steps))
        if check_n == 0:
            raise ValueError(f"No alignable target prefix for {horizon}-minute horizon.")
        expected = X[steps:steps + check_n, -1, :]
        actual = Y[:check_n]
        finite_pair = np.isfinite(expected) & np.isfinite(actual)
        if not finite_pair.any():
            raise ValueError(f"Cannot verify target alignment for {horizon}-minute horizon.")
        max_abs_diff = float(np.max(np.abs(expected[finite_pair] - actual[finite_pair])))
        alignment_checks.append({
            "horizon_min": horizon,
            "target_rows": int(len(Y)),
            "evaluated_rows": int(n),
            "verified_prefix_rows": int(check_n),
            "max_abs_scaled_difference": max_abs_diff,
        })
        if max_abs_diff > 1e-5:
            raise ValueError(
                f"Target alignment check failed for {horizon} min: "
                f"max scaled difference {max_abs_diff:.8g}. No metrics were written."
            )

        target_positions = test_start + INPUT_STEPS - 1 + steps + np.arange(n, dtype=np.int64)
        observed_mask = load_raw_observation_mask(sensor_ids, target_positions)

        pred_batches = []
        with torch.no_grad():
            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                xb = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
                wb = torch.as_tensor(W[start:end], dtype=torch.float32, device=device)
                out = model(xb, wb, graph)[horizon].detach().cpu().numpy()
                pred_batches.append(out)
        pred_scaled = np.concatenate(pred_batches, axis=0)

        pred_mph = scaler.inverse_transform(pred_scaled)
        target_mph = scaler.inverse_transform(Y[:n])
        persistence_mph = scaler.inverse_transform(X[:n, -1, :])

        model_mae, model_rmse, valid_count = metrics(pred_mph, target_mph, observed_mask)
        persistence_mae, persistence_rmse, baseline_count = metrics(
            persistence_mph, target_mph, observed_mask
        )
        if valid_count != baseline_count:
            raise RuntimeError("Model and persistence baseline were not evaluated on the same mask.")
        improvement = (persistence_mae - model_mae) / persistence_mae * 100 if persistence_mae else float("nan")
        rows.append({
            "horizon_min": horizon,
            "target_rows_available": int(len(Y)),
            "samples_scored": int(n),
            "observed_sensor_targets": valid_count,
            "observed_fraction": valid_count / int(n * N_SENSORS),
            "MAE_mph": model_mae,
            "RMSE_mph": model_rmse,
            "persistence_MAE_mph": persistence_mae,
            "persistence_RMSE_mph": persistence_rmse,
            "MAE_improvement_vs_persistence_pct": improvement,
            "mask_semantics": "raw METR-LA observation (finite and non-zero); interpolated values excluded",
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "urbanflow_verified_test_results.csv"
    audit_path = RESULTS / "urbanflow_verified_test_audit.json"
    if csv_path.exists() or audit_path.exists():
        raise FileExistsError(
            "Verified-result output already exists. Move it aside manually before rerunning; "
            "the script will not overwrite previous results."
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    audit = {
        "project": "UrbanFlow",
        "evaluation": "multi-horizon model vs persistence baseline",
        "device": str(device),
        "checkpoint": str(MODEL_PATH.relative_to(ROOT)),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "test_start_raw_row": test_start,
        "input_steps": INPUT_STEPS,
        "sampling_interval_minutes": SAMPLE_MINUTES,
        "target_mask": "raw METR-LA values finite and non-zero; interpolated values excluded",
        "alignment_checks": alignment_checks,
        "metrics_csv": str(csv_path.relative_to(ROOT)),
        "warning": "This is a new evaluation; do not substitute these metrics for historical results until reviewed.",
    }
    audit_path.write_text(json.dumps(audit, indent=2))
    print("VERIFIED MULTI-HORIZON EVALUATION COMPLETE")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved new files:\n  {csv_path.relative_to(ROOT)}\n  {audit_path.relative_to(ROOT)}")
    print("Original targets, masks, checkpoint, and historical result files were not modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"EVALUATION STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
