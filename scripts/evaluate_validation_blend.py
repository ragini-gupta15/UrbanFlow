
import json
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
OUT = ROOT / "results" / "urbanflow_validation_blend.csv"
AUDIT = ROOT / "results" / "urbanflow_validation_blend_audit.json"

HORIZON = 30
STEPS = 6
BATCH_SIZE = 64
N_SENSORS = 207
INPUT_STEPS = 12

for p in [
    DATA / "validation_sequences.npz",
    DATA / "weather_validation_sequences.npy",
    DATA / "y_val_30min.npy",
    DATA / "traffic_scaler.pkl",
    DATA / "adjacency_matrix.npy",
    DATA / "metadata.json",
    RAW,
    MODEL_PATH,
]:
    if not p.is_file():
        raise FileNotFoundError(p)

if OUT.exists() or AUDIT.exists():
    raise FileExistsError("Output exists; refusing to overwrite.")

with np.load(DATA / "validation_sequences.npz") as z:
    X = np.asarray(z["X"], dtype=np.float32)
    y_archive = np.asarray(z["y"], dtype=np.float32)

Y = np.asarray(np.load(DATA / "y_val_30min.npy"), dtype=np.float32)
W = np.asarray(np.load(DATA / "weather_validation_sequences.npy"), dtype=np.float32)
scaler = joblib.load(DATA / "traffic_scaler.pkl")
adjacency = np.asarray(np.load(DATA / "adjacency_matrix.npy"), dtype=np.float32)
metadata = json.loads((DATA / "metadata.json").read_text())
sensor_ids = [str(s) for s in metadata["sensor_ids"]]

if X.shape != (3410, INPUT_STEPS, N_SENSORS):
    raise ValueError(f"Unexpected X shape: {X.shape}")
if W.shape != (len(X), INPUT_STEPS, 6):
    raise ValueError(f"Unexpected W shape: {W.shape}")
if Y.shape != (len(X), N_SENSORS) or y_archive.shape != Y.shape:
    raise ValueError("Validation target shapes do not match.")

finite = np.isfinite(Y) & np.isfinite(y_archive)
if not finite.any() or np.max(np.abs(Y[finite] - y_archive[finite])) > 1e-5:
    raise ValueError("Validation target arrays disagree.")

# Verify 30-minute target convention before evaluating.
check_n = len(X) - STEPS
pair = np.isfinite(X[STEPS:, -1, :]) & np.isfinite(Y[:check_n])
if not pair.any():
    raise ValueError("Cannot verify target alignment.")
alignment_diff = float(np.max(np.abs(X[STEPS:, -1, :][pair] - Y[:check_n][pair])))
if alignment_diff > 1e-5:
    raise ValueError(f"Target alignment failed: {alignment_diff}")

with h5py.File(RAW, "r") as h5:
    raw = np.asarray(h5["df/block0_values"][:], dtype=np.float32)
    raw_ids = [
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in h5["df/block0_items"][:]
    ]

if set(sensor_ids) != set(raw_ids) or len(sensor_ids) != N_SENSORS:
    raise ValueError("Raw and model sensor IDs do not match.")

val_start = int(len(raw) * 0.70)
target_positions = val_start + INPUT_STEPS - 1 + STEPS + np.arange(len(X))
if target_positions.max() >= len(raw):
    raise ValueError("Target positions exceed raw data.")

raw_col = {sid: i for i, sid in enumerate(raw_ids)}
columns = [raw_col[sid] for sid in sensor_ids]
raw_targets = raw[target_positions][:, columns]
mask = np.isfinite(raw_targets) & (raw_targets != 0)

A = adjacency + np.eye(N_SENSORS, dtype=np.float32)
degree = A.sum(axis=1)
inv = np.zeros_like(degree)
positive = degree > 0
inv[positive] = degree[positive] ** -0.5
graph_np = inv[:, None] * A * inv[None, :]

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import sys
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
pred_batches = []

with torch.no_grad():
    for start in range(0, len(X), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X))
        xb = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
        wb = torch.as_tensor(W[start:end], dtype=torch.float32, device=device)
        pred_batches.append(
            model(xb, wb, graph)[HORIZON].detach().cpu().numpy()
        )

model_pred = scaler.inverse_transform(np.concatenate(pred_batches))
truth = scaler.inverse_transform(Y)
persistence = scaler.inverse_transform(X[:, -1, :])

valid = mask & np.isfinite(truth) & np.isfinite(model_pred) & np.isfinite(persistence)
if not valid.any():
    raise ValueError("No valid observed targets.")

rows = []
for alpha in np.round(np.arange(0, 1.01, 0.1), 2):
    blended = alpha * model_pred + (1 - alpha) * persistence
    err = blended[valid] - truth[valid]
    rows.append({
        "alpha_model": float(alpha),
        "alpha_persistence": float(1 - alpha),
        "MAE_mph": float(np.mean(np.abs(err))),
        "RMSE_mph": float(np.sqrt(np.mean(err ** 2))),
        "observed_targets": int(valid.sum()),
    })

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)

audit = {
    "split": "validation",
    "horizon_min": HORIZON,
    "device": str(device),
    "checkpoint_best_epoch": ckpt.get("best_epoch"),
    "alignment_max_abs_scaled_difference": alignment_diff,
    "mask": "raw METR-LA finite and non-zero observations",
    "validation_weather_shape": list(W.shape),
    "selection_rule": "lowest validation MAE; RMSE reported alongside",
    "test_set_used_for_selection": False,
    "checkpoint_modified": False,
}
AUDIT.write_text(json.dumps(audit, indent=2))

print(result.to_string(index=False))
print("\nSaved:", OUT)
print("Audit:", AUDIT)
print("\nBest validation MAE weight:")
print(result.loc[result["MAE_mph"].idxmin()].to_string())
print("\nBest validation RMSE weight:")
print(result.loc[result["RMSE_mph"].idxmin()].to_string())
