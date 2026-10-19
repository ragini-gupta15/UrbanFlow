#!/usr/bin/env python3
"""Evaluate validation-selected blends on held-out test data; never retrain or retune."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
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
VALIDATION_CSV = ROOT / "results" / "urbanflow_validation_blend_all_horizons.csv"
OUT = ROOT / "results" / "urbanflow_blend_all_horizons_test.csv"
AUDIT = ROOT / "results" / "urbanflow_blend_all_horizons_test_audit.json"

HORIZONS = (15, 30, 45, 60)
WEIGHTS = {15: 0.35, 30: 0.45, 45: 0.55, 60: 0.60}
INPUT_STEPS = 12
SAMPLE_MINUTES = 5
N_SENSORS = 207
BATCH_SIZE = 64


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path.relative_to(ROOT)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_mask(sensor_ids: list[str], positions: np.ndarray) -> np.ndarray:
    with h5py.File(RAW, "r") as h5:
        values = np.asarray(h5["df/block0_values"][:], dtype=np.float32)
        raw_ids = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in h5["df/block0_items"][:]
        ]
    if len(raw_ids) != values.shape[1] or set(raw_ids) != set(sensor_ids):
        raise ValueError("Raw METR-LA sensor IDs do not match model sensor order.")
    if positions.size == 0 or positions.min() < 0 or positions.max() >= len(values):
        raise ValueError("Target positions fall outside raw METR-LA timestamps.")
    order = {sid: i for i, sid in enumerate(raw_ids)}
    selected = values[positions][:, [order[sid] for sid in sensor_ids]]
    return np.isfinite(selected) & (selected != 0)


def score(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    valid = mask & np.isfinite(pred) & np.isfinite(truth)
    count = int(valid.sum())
    if count == 0:
        raise ValueError("No valid observed targets available for scoring.")
    error = pred[valid] - truth[valid]
    return (
        float(np.mean(np.abs(error))),
        float(np.sqrt(np.mean(np.square(error)))),
        count,
    )


def main() -> int:
    # Fail closed: never overwrite results or the original benchmark.
    if OUT.exists() or AUDIT.exists():
        raise FileExistsError(
            "New test output already exists. Nothing was overwritten: "
            f"{OUT.name}, {AUDIT.name}"
        )
    if not VALIDATION_CSV.is_file():
        raise FileNotFoundError(
            f"Validation-selected weight record is missing: {VALIDATION_CSV}"
        )

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

    validation = pd.read_csv(VALIDATION_CSV)
    if not {"horizon_min", "alpha_model"}.issubset(validation.columns):
        raise ValueError("Validation CSV lacks horizon_min/alpha_model columns.")
    # Confirm each fixed weight appears in the validation alpha grid.
    for horizon, alpha in WEIGHTS.items():
        part = validation[validation["horizon_min"] == horizon]
        if part.empty or not np.isclose(
            part["alpha_model"].astype(float).to_numpy(), alpha, atol=1e-8
        ).any():
            raise ValueError(
                f"Fixed alpha {alpha} for {horizon} min is not recorded "
                "in the validation results. No test metrics were written."
            )

    with np.load(DATA / "test_sequences.npz") as archive:
        X = np.asarray(archive["X"], dtype=np.float32)
    W = np.asarray(np.load(DATA / "weather_test_sequences.npy"), dtype=np.float32)
    metadata = json.loads((DATA / "metadata.json").read_text())
    sensor_ids = [str(x) for x in metadata["sensor_ids"]]
    scaler = joblib.load(DATA / "traffic_scaler.pkl")
    adjacency = np.asarray(np.load(DATA / "adjacency_matrix.npy"), dtype=np.float32)

    if X.shape != (len(X), INPUT_STEPS, N_SENSORS):
        raise ValueError(f"Unexpected test traffic shape: {X.shape}")
    if W.shape != (len(X), INPUT_STEPS, 6):
        raise ValueError(f"Weather/test alignment mismatch: {W.shape}")
    if len(sensor_ids) != N_SENSORS or len(set(sensor_ids)) != N_SENSORS:
        raise ValueError("Expected 207 unique sensor IDs in model order.")
    if adjacency.shape != (N_SENSORS, N_SENSORS) or not np.isfinite(adjacency).all():
        raise ValueError("Invalid adjacency matrix.")

    with h5py.File(RAW, "r") as h5:
        raw_length = int(h5["df/block0_values"].shape[0])
    test_start = int(raw_length * 0.80)

    A = adjacency + np.eye(N_SENSORS, dtype=np.float32)
    degree = A.sum(axis=1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    positive = degree > 0
    inv_sqrt[positive] = degree[positive] ** -0.5
    graph = torch.as_tensor(inv_sqrt[:, None] * A * inv_sqrt[None, :],
                            dtype=torch.float32)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sys.path.insert(0, str(ROOT))
    from src.model import UrbanFlowFinalModel

    model = UrbanFlowFinalModel().to(device)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint lacks model_state_dict.")
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
        check_n = min(n, max(0, len(X) - steps))
        if n == 0 or check_n == 0:
            raise ValueError(f"No alignable samples for {horizon}-min horizon.")

        expected, actual = X[steps:steps + check_n, -1, :], Y[:check_n]
        finite_pair = np.isfinite(expected) & np.isfinite(actual)
        if not finite_pair.any():
            raise ValueError(f"Cannot verify target alignment at {horizon} min.")
        max_diff = float(np.max(np.abs(expected[finite_pair] - actual[finite_pair])))
        alignment_checks.append({
            "horizon_min": horizon,
            "target_rows": int(len(Y)),
            "evaluated_rows": int(n),
            "verified_prefix_rows": int(check_n),
            "max_abs_scaled_difference": max_diff,
        })
        if max_diff > 1e-5:
            raise ValueError(
                f"{horizon}-min target alignment failed ({max_diff:.8g}); "
                "no metrics were written."
            )

        positions = test_start + INPUT_STEPS - 1 + steps + np.arange(n, dtype=np.int64)
        mask = observed_mask(sensor_ids, positions)
        truth = np.asarray(scaler.inverse_transform(Y[:n]), dtype=np.float32)
        persistence_scaled = X[:n, -1, :]
        persistence = np.asarray(scaler.inverse_transform(persistence_scaled), dtype=np.float32)

        predictions = []
        with torch.no_grad():
            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                xb = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
                wb = torch.as_tensor(W[start:end], dtype=torch.float32, device=device)
                output = model(xb, wb, graph)
                if horizon not in output:
                    raise ValueError(f"Model output lacks horizon key {horizon}.")
                predictions.append(output[horizon].detach().cpu().numpy())

        model_scaled = np.concatenate(predictions, axis=0)
        if model_scaled.shape != Y[:n].shape:
            raise ValueError(
                f"Model output shape mismatch at {horizon} min: {model_scaled.shape}"
            )
        model_pred = np.asarray(scaler.inverse_transform(model_scaled), dtype=np.float32)
        alpha = WEIGHTS[horizon]
        blend_pred = alpha * model_pred + (1.0 - alpha) * persistence

        for name, pred in (
            ("model", model_pred),
            ("persistence", persistence),
            ("validation_selected_blend", blend_pred),
        ):
            mae, rmse, count = score(pred, truth, mask)
            rows.append({
                "split": "test",
                "horizon_min": horizon,
                "method": name,
                "model_weight_alpha": alpha if name == "validation_selected_blend" else np.nan,
                "persistence_weight": 1.0 - alpha if name == "validation_selected_blend" else np.nan,
                "mae_mph": mae,
                "rmse_mph": rmse,
                "observed_target_count": count,
            })

    # Write outputs only after every horizon passes alignment and scoring.
    result = pd.DataFrame(rows)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "held-out test",
        "model_checkpoint": str(MODEL_PATH.relative_to(ROOT)),
        "model_checkpoint_sha256": sha256(MODEL_PATH),
        "validation_weight_source": str(VALIDATION_CSV.relative_to(ROOT)),
        "fixed_validation_selected_model_weights": WEIGHTS,
        "test_used_for_weight_selection": False,
        "target_mask": "finite and nonzero raw METR-LA values in model sensor order",
        "test_start_index": test_start,
        "input_shape": list(X.shape),
        "weather_shape": list(W.shape),
        "alignment_checks": alignment_checks,
        "output_csv": str(OUT.relative_to(ROOT)),
        "original_verified_benchmark_modified": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {OUT.relative_to(ROOT)}")
    print(f"Audit: {AUDIT.relative_to(ROOT)}")
    print("Validation-selected weights were fixed before test evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
