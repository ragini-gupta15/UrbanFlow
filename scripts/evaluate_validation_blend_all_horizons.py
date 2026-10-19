import hashlib
import json
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
RAW = ROOT / "data/raw/metr-la.h5"
MODEL_PATH = ROOT / "models/urbanflow_final_gnn_transformer_weather_multihorizon.pth"
OUT = ROOT / "results/urbanflow_validation_blend_all_horizons.csv"
AUDIT = ROOT / "results/urbanflow_validation_blend_all_horizons_audit.json"

HORIZONS = (15, 30, 45, 60)
INPUT_STEPS = 12
N_SENSORS = 207
BATCH_SIZE = 64
VAL_START = 23990
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
TOL = 1e-5


def main():
    if OUT.exists() or AUDIT.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment output.")

    required = [
        DATA / "validation_sequences.npz",
        DATA / "weather_validation_sequences.npy",
        DATA / "traffic_scaler.pkl",
        DATA / "adjacency_matrix.npy",
        DATA / "metadata.json",
        RAW, MODEL_PATH,
        *[DATA / f"y_val_{h}min.npy" for h in HORIZONS],
    ]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)

    with np.load(DATA / "validation_sequences.npz") as z:
        X = np.asarray(z["X"], dtype=np.float32)
    W = np.asarray(np.load(DATA / "weather_validation_sequences.npy"), dtype=np.float32)
    scaler = joblib.load(DATA / "traffic_scaler.pkl")
    adjacency = np.asarray(np.load(DATA / "adjacency_matrix.npy"), dtype=np.float32)
    meta = json.loads((DATA / "metadata.json").read_text())
    sensor_ids = [str(s) for s in meta["sensor_ids"]]

    if X.shape != (3410, 12, N_SENSORS) or W.shape != (len(X), 12, 6):
        raise ValueError(f"Unexpected validation input shapes: X={X.shape}, W={W.shape}")
    if len(sensor_ids) != N_SENSORS or len(set(sensor_ids)) != N_SENSORS:
        raise ValueError("Invalid sensor IDs.")
    if adjacency.shape != (N_SENSORS, N_SENSORS) or not np.isfinite(adjacency).all():
        raise ValueError("Invalid adjacency matrix.")

    with h5py.File(RAW, "r") as f:
        raw = np.asarray(f["df/block0_values"][:], dtype=np.float32)
        raw_ids = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in f["df/block0_items"][:]
        ]
    if len(set(raw_ids)) != len(raw_ids) or set(raw_ids) != set(sensor_ids):
        raise ValueError("Raw and model sensor IDs do not match.")
    cols = [raw_ids.index(s) for s in sensor_ids]

    A = adjacency + np.eye(N_SENSORS, dtype=np.float32)
    degree = A.sum(axis=1)
    inv = np.zeros_like(degree)
    positive = degree > 0
    inv[positive] = degree[positive] ** -0.5
    graph_np = inv[:, None] * A * inv[None, :]

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    sys.path.insert(0, str(ROOT))
    from src.model import UrbanFlowFinalModel

    model = UrbanFlowFinalModel().to(device)
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    graph = torch.as_tensor(graph_np, dtype=torch.float32, device=device)
    predictions = {h: [] for h in HORIZONS}
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X))
            xb = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
            wb = torch.as_tensor(W[start:end], dtype=torch.float32, device=device)
            out = model(xb, wb, graph)
            for h in HORIZONS:
                predictions[h].append(out[h].detach().cpu().numpy())
    predictions = {h: np.concatenate(predictions[h]) for h in HORIZONS}

    rows, audit_horizons = [], {}
    for h in HORIZONS:
        steps = h // 5
        Y = np.asarray(np.load(DATA / f"y_val_{h}min.npy"), dtype=np.float32)
        n = len(Y)
        if Y.shape != (n, N_SENSORS) or n > len(X):
            raise ValueError(f"Unexpected {h}-minute target shape: {Y.shape}")

        # Verify the saved target against the shifted input sequence.
        check_n = min(n, len(X) - steps)
        pair = (
            np.isfinite(X[steps:steps + check_n, -1, :])
            & np.isfinite(Y[:check_n])
        )
        if not pair.any():
            raise ValueError(f"No comparable alignment values for {h} minutes.")
        alignment_error = float(np.max(np.abs(
            X[steps:steps + check_n, -1, :][pair] - Y[:check_n][pair]
        )))
        if alignment_error > TOL:
            raise ValueError(f"{h}-minute alignment failed: {alignment_error}")

        positions = VAL_START + np.arange(n) + INPUT_STEPS - 1 + steps
        if positions.min() < 0 or positions.max() >= len(raw):
            raise ValueError(f"{h}-minute raw target positions out of range.")

        raw_truth = raw[positions][:, cols]
        raw_mask = np.isfinite(raw_truth) & (raw_truth != 0)
        truth = scaler.inverse_transform(Y)
        model_pred = scaler.inverse_transform(predictions[h][:n])
        persistence = scaler.inverse_transform(X[:n, -1, :])
        valid = (
            raw_mask & np.isfinite(truth)
            & np.isfinite(model_pred) & np.isfinite(persistence)
        )
        if not valid.any():
            raise ValueError(f"No valid observed targets for {h} minutes.")

        target_error = float(np.max(np.abs(truth[valid] - raw_truth[valid])))
        if target_error > 1e-3:
            raise ValueError(
                f"{h}-minute target/raw mismatch: {target_error:.6g} mph. "
                "No results will be written."
            )

        def score(pred):
            err = pred[valid] - truth[valid]
            return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))

        model_mae, model_rmse = score(model_pred)
        persist_mae, persist_rmse = score(persistence)
        horizon_rows = []

        for alpha in GRID:
            blended = alpha * model_pred + (1 - alpha) * persistence
            mae, rmse = score(blended)
            row = {
                "horizon_min": h,
                "alpha_model": float(alpha),
                "alpha_persistence": float(1 - alpha),
                "MAE_mph": mae,
                "RMSE_mph": rmse,
                "model_only_MAE_mph": model_mae,
                "model_only_RMSE_mph": model_rmse,
                "persistence_MAE_mph": persist_mae,
                "persistence_RMSE_mph": persist_rmse,
                "target_rows": n,
                "observed_sensor_targets": int(valid.sum()),
                "alignment_prefix_rows": check_n,
                "alignment_max_abs_scaled_diff": alignment_error,
                "target_raw_max_abs_error_mph": target_error,
            }
            rows.append(row)
            horizon_rows.append(row)

        best_mae = min(horizon_rows, key=lambda r: (r["MAE_mph"], r["RMSE_mph"]))
        best_rmse = min(horizon_rows, key=lambda r: (r["RMSE_mph"], r["MAE_mph"]))
        audit_horizons[str(h)] = {
            "target_rows": n,
            "observed_sensor_targets": int(valid.sum()),
            "alignment_prefix_rows": check_n,
            "alignment_max_abs_scaled_diff": alignment_error,
            "target_raw_max_abs_error_mph": target_error,
            "model_only": {"MAE_mph": model_mae, "RMSE_mph": model_rmse},
            "persistence_only": {"MAE_mph": persist_mae, "RMSE_mph": persist_rmse},
            "best_grid_alpha_by_MAE": best_mae["alpha_model"],
            "best_grid_alpha_by_RMSE": best_rmse["alpha_model"],
            "raw_mask": "finite, non-zero METR-LA observations in model sensor order",
        }
        print(
            f"{h:2} min | observed={int(valid.sum()):,} | "
            f"model MAE/RMSE={model_mae:.4f}/{model_rmse:.4f} | "
            f"persistence={persist_mae:.4f}/{persist_rmse:.4f} | "
            f"best MAE alpha={best_mae['alpha_model']:.2f}"
        )

    if OUT.exists() or AUDIT.exists():
        raise FileExistsError("Refusing to overwrite existing outputs.")
    audit = {
        "experiment": "validation-only model/persistence blend sweep",
        "split": "validation only; test data not loaded",
        "device": str(device),
        "checkpoint": str(MODEL_PATH.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "checkpoint_best_epoch": ckpt.get("best_epoch"),
        "validation_sequence_start_raw_row": VAL_START,
        "input_steps": INPUT_STEPS,
        "grid_alpha_model": [float(x) for x in GRID],
        "selection": "Select alpha separately per horizon using validation MAE; report RMSE optimum separately.",
        "horizons": audit_horizons,
        "outputs": {"csv": str(OUT.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT))},
        "notes": [
            "No training or optimizer steps performed.",
            "Raw observation masks used for scoring; saved mask files left untouched.",
            "Existing benchmarks and checkpoints were not modified.",
            "Weights must be fixed before any held-out test evaluation.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"\nSaved {OUT.relative_to(ROOT)}")
    print(f"Saved {AUDIT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
