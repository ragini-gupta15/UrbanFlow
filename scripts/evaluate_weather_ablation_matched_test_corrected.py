#!/usr/bin/env python3
"""Strict matched-test evaluation for UrbanFlow weather ablation checkpoints.

Architecture is copied from scripts/run_weather_ablation.py.
Alignment, sensor-ID ordering, and raw observation masks follow
scripts/evaluate_multihorizon_verified.py.
No training is performed. Existing output files are never overwritten.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
RAW = ROOT / "data/raw/metr-la.h5"
WEATHER_CKPT = ROOT / "models/weather_graph_aware_multi_horizon_best.pth"
TRAFFIC_CKPT = ROOT / "models/experiments/weather_ablation_traffic_only_best.pth"
OUT_CSV = ROOT / "results/experiments/weather_ablation_matched_test_scaled_all_horizons_mph.csv"
OUT_AUDIT = ROOT / "results/experiments/weather_ablation_matched_test_scaled_all_horizons_audit.json"
HORIZONS = (15, 30, 45, 60)
N_SENSORS = 207
INPUT_STEPS = 12
SAMPLE_MINUTES = 5
D_MODEL = 128
NHEAD = 4
FF_DIM = 256
N_LAYERS = 2
DROPOUT = 0.0
BATCH_SIZE = 64
ALIGNMENT_TOLERANCE = 1e-5


def stop(message: str) -> None:
    raise RuntimeError(message)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional encoding "
                f"length {self.pe.size(1)}."
            )
        return x + self.pe[:, : x.size(1), :]


class WeatherGraphAwareUrbanFlow(nn.Module):
    """Exact model structure/forward logic from scripts/run_weather_ablation.py."""

    def __init__(self, adjacency: torch.Tensor, use_weather: bool, positional_length: int):
        super().__init__()
        self.num_sensors = N_SENSORS
        self.d_model = D_MODEL
        self.use_weather = use_weather
        self.traffic_projection = nn.Linear(1, D_MODEL)
        self.position_encoding = PositionalEncoding(D_MODEL, positional_length)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.spatial_query = nn.Linear(D_MODEL, D_MODEL)
        self.spatial_key = nn.Linear(D_MODEL, D_MODEL)
        self.spatial_value = nn.Linear(D_MODEL, D_MODEL)
        if use_weather:
            self.weather_encoder = nn.Sequential(
                nn.Linear(6, 32), nn.GELU(), nn.Linear(32, D_MODEL)
            )
        self.fusion = nn.Sequential(
            nn.Linear(D_MODEL * (2 if use_weather else 1), D_MODEL),
            nn.GELU(),
            nn.LayerNorm(D_MODEL),
        )
        self.head_15 = nn.Linear(D_MODEL, 1)
        self.head_30 = nn.Linear(D_MODEL, 1)
        self.head_45 = nn.Linear(D_MODEL, 1)
        self.head_60 = nn.Linear(D_MODEL, 1)
        # The training script registers graph_mask for both models. The weather
        # checkpoint's state dict omits it, so keep the same forward value as a
        # plain tensor for that checkpoint to allow strict loading.
        if not use_weather:
            self.register_buffer("graph_mask", adjacency.float())
        else:
            # Move with the model to MPS/CPU, but omit it from state_dict because
            # the saved weather checkpoint does not contain this buffer.
            self.register_buffer("graph_mask", adjacency.float(), persistent=False)

    def forward(self, traffic, weather=None):
        b, t, s = traffic.shape
        x = traffic.permute(0, 2, 1).unsqueeze(-1).reshape(b * s, t, 1)
        x = self.traffic_projection(x)
        x = self.position_encoding(x)
        x = self.temporal_transformer(x)
        x = x[:, -1, :].reshape(b, s, self.d_model)

        q, k, v = self.spatial_query(x), self.spatial_key(x), self.spatial_value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        scores = scores + torch.log(self.graph_mask.clamp_min(1e-6))
        spatial_context = torch.matmul(torch.softmax(scores, dim=-1), v)

        if self.use_weather:
            if weather is None:
                raise ValueError("Weather input is required for weather-enabled model.")
            weather_context = self.weather_encoder(weather[:, -1, :]).unsqueeze(1)
            weather_context = weather_context.expand(-1, s, -1)
            fused_input = torch.cat([spatial_context, weather_context], dim=-1)
        else:
            fused_input = spatial_context

        fused = self.fusion(fused_input)
        return {
            "15min": self.head_15(fused).squeeze(-1),
            "30min": self.head_30(fused).squeeze(-1),
            "45min": self.head_45(fused).squeeze(-1),
            "60min": self.head_60(fused).squeeze(-1),
        }


def require_files(paths: list[Path]) -> None:
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        stop("Required file(s) missing: " + ", ".join(missing))


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        stop(f"Unexpected checkpoint structure: {path.relative_to(ROOT)}")
    return checkpoint


def load_raw_observation_mask(sensor_ids: list[str], target_positions: np.ndarray) -> np.ndarray:
    """Exact sensor-order and validity convention from verified evaluator."""
    with h5py.File(RAW, "r") as h5:
        if "df/block0_values" not in h5 or "df/block0_items" not in h5:
            stop("Raw HDF5 file lacks df/block0_values or df/block0_items.")
        raw_values = np.asarray(h5["df/block0_values"][:], dtype=np.float32)
        raw_sensor_ids = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in h5["df/block0_items"][:]
        ]

    if len(raw_sensor_ids) != raw_values.shape[1]:
        stop("Raw sensor-ID count does not match raw data columns.")
    if len(sensor_ids) != N_SENSORS or len(set(sensor_ids)) != N_SENSORS:
        stop("metadata.json must contain 207 unique sensor IDs in model order.")
    if set(sensor_ids) != set(raw_sensor_ids):
        stop("Sensor IDs in metadata.json do not match raw METR-LA sensor IDs.")
    if target_positions.size == 0 or target_positions.min() < 0 or target_positions.max() >= len(raw_values):
        stop("Computed target positions fall outside raw METR-LA timestamps.")

    raw_col = {sid: i for i, sid in enumerate(raw_sensor_ids)}
    col_order = [raw_col[sid] for sid in sensor_ids]
    selected = raw_values[target_positions][:, col_order]
    return np.isfinite(selected) & (selected != 0)


def calculate_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    valid = mask & np.isfinite(pred) & np.isfinite(truth)
    count = int(valid.sum())
    if count == 0:
        stop("No valid observed targets remain for metric calculation.")
    error = pred[valid] - truth[valid]
    return {
        "mae_mph": float(np.mean(np.abs(error))),
        "rmse_mph": float(np.sqrt(np.mean(np.square(error)))),
        "valid_observations": count,
    }


def main() -> int:
    if OUT_CSV.exists() or OUT_AUDIT.exists():
        found = [str(p.relative_to(ROOT)) for p in (OUT_CSV, OUT_AUDIT) if p.exists()]
        stop("Refusing to overwrite existing output(s): " + ", ".join(found))

    require_files([
        DATA / "test_sequences.npz",
        DATA / "weather_test_sequences.npy",
        DATA / "traffic_scaler.pkl",
        DATA / "adjacency_matrix.npy",
        DATA / "metadata.json",
        RAW,
        DATA / "y_test_15min.npy",
        DATA / "y_test_30min.npy",
        DATA / "y_test_45min.npy",
        DATA / "y_test_60min.npy",
        WEATHER_CKPT,
        TRAFFIC_CKPT,
    ])

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    metadata = json.loads((DATA / "metadata.json").read_text())
    if "sensor_ids" not in metadata:
        stop("metadata.json is missing sensor_ids.")

    sensor_ids = [str(x) for x in metadata["sensor_ids"]]

    weather_ckpt = load_checkpoint(WEATHER_CKPT)
    traffic_ckpt = load_checkpoint(TRAFFIC_CKPT)
    weather_sd = weather_ckpt["model_state_dict"]
    traffic_sd = traffic_ckpt["model_state_dict"]
    weather_pe = weather_sd.get("position_encoding.pe")
    traffic_pe = traffic_sd.get("position_encoding.pe")
    if weather_pe is None or traffic_pe is None:
        stop("A checkpoint is missing position_encoding.pe.")

    adjacency = np.asarray(np.load(DATA / "adjacency_matrix.npy"), dtype=np.float32)
    if adjacency.shape != (N_SENSORS, N_SENSORS):
        stop(f"Expected adjacency shape (207,207), got {adjacency.shape}.")
    # Match training script's row normalization.
    adjacency = adjacency / (adjacency.sum(axis=1, keepdims=True) + 1e-8)
    adjacency_tensor = torch.from_numpy(adjacency)

    weather_model = WeatherGraphAwareUrbanFlow(
        adjacency_tensor, True, int(weather_pe.shape[1])
    )
    traffic_model = WeatherGraphAwareUrbanFlow(
        adjacency_tensor, False, int(traffic_pe.shape[1])
    )
    weather_model.load_state_dict(weather_sd, strict=True)
    traffic_model.load_state_dict(traffic_sd, strict=True)
    weather_model.to(device).eval()
    traffic_model.to(device).eval()
    print("Both checkpoints loaded with strict=True.")

    x_archive = np.load(DATA / "test_sequences.npz")
    if "X" not in x_archive:
        stop("test_sequences.npz does not contain key X.")
    X = np.asarray(x_archive["X"], dtype=np.float32)
    weather_scaler = joblib.load(DATA / "weather_scaler.pkl")
    W_raw = np.asarray(np.load(DATA / "weather_test_sequences.npy"), dtype=np.float32)
    W = weather_scaler.transform(
    W_raw.reshape(-1, W_raw.shape[-1])
).reshape(W_raw.shape).astype(np.float32)
    if X.ndim != 3 or X.shape[1:] != (INPUT_STEPS, N_SENSORS):
        stop(f"Unexpected X shape {X.shape}; expected (N,12,207).")
    if W.ndim != 3 or W.shape[0] != len(X) or W.shape[1] != INPUT_STEPS or W.shape[2] != 6:
        stop(f"Unexpected weather shape {W.shape}; expected (N,12,6).")
    if len(sensor_ids) != N_SENSORS or len(set(sensor_ids)) != N_SENSORS:
        stop("metadata.json must contain 207 unique sensor IDs.")

    scaler = joblib.load(DATA / "traffic_scaler.pkl")
    with h5py.File(RAW, "r") as h5:
        if "df/block0_values" not in h5:
            stop("Raw HDF5 lacks df/block0_values.")
        raw_length = int(h5["df/block0_values"].shape[0])
    test_start = int(raw_length * 0.80)

    # Keep all results in memory until every horizon has passed alignment/mask
    # validation and inference; output files are created only at the very end.
    rows: list[dict] = []
    alignment_audit: list[dict] = []
    checkpoint_audit = {
        "weather_graph_aware": {
            "path": str(WEATHER_CKPT.relative_to(ROOT)),
            "architecture": weather_ckpt.get("architecture"),
            "best_epoch": weather_ckpt.get("best_epoch"),
            "best_val_mae_30": weather_ckpt.get("best_val_mae_30"),
            "positional_encoding_length": int(weather_pe.shape[1]),
        },
        "traffic_only": {
            "path": str(TRAFFIC_CKPT.relative_to(ROOT)),
            "architecture": traffic_ckpt.get("architecture"),
            "best_epoch": traffic_ckpt.get("best_epoch"),
            "best_val_mae_30": traffic_ckpt.get("best_val_mae_30"),
            "positional_encoding_length": int(traffic_pe.shape[1]),
        },
    }

    with torch.no_grad():
        for horizon in HORIZONS:
            steps = horizon // SAMPLE_MINUTES
            Y = np.asarray(np.load(DATA / f"y_test_{horizon}min.npy"), dtype=np.float32)
            if Y.ndim != 2 or Y.shape[1] != N_SENSORS:
                stop(f"Unexpected {horizon}-min target shape: {Y.shape}")
            n = min(len(X), len(W), len(Y))
            if n == 0:
                stop(f"No samples available for {horizon}-minute evaluation.")

            # Exact prefix alignment convention from evaluate_multihorizon_verified.py.
            check_n = min(n, max(0, len(X) - steps))
            if check_n == 0:
                stop(f"No alignable target prefix for {horizon}-minute horizon.")
            expected = X[steps:steps + check_n, -1, :]
            actual = Y[:check_n]
            finite_pair = np.isfinite(expected) & np.isfinite(actual)
            if not finite_pair.any():
                stop(f"Cannot verify target alignment for {horizon}-minute horizon.")
            max_abs_diff = float(np.max(np.abs(expected[finite_pair] - actual[finite_pair])))
            alignment_audit.append({
                "horizon_min": horizon,
                "target_rows": int(len(Y)),
                "evaluated_rows": int(n),
                "verified_prefix_rows": int(check_n),
                "max_abs_scaled_difference": max_abs_diff,
            })
            if max_abs_diff > ALIGNMENT_TOLERANCE:
                stop(
                    f"Target alignment failed at {horizon} min: "
                    f"max scaled difference {max_abs_diff:.8g} > "
                    f"{ALIGNMENT_TOLERANCE}. No metrics were written."
                )

            target_positions = (
                test_start + INPUT_STEPS - 1 + steps + np.arange(n, dtype=np.int64)
            )
            mask = load_raw_observation_mask(sensor_ids, target_positions)

            pred_weather_batches = []
            pred_traffic_batches = []
            pred_persistence_batches = []
            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                xb = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
                wb = torch.as_tensor(W[start:end], dtype=torch.float32, device=device)
                pw = weather_model(xb, wb)[f"{horizon}min"].detach().cpu().numpy()
                pt = traffic_model(xb)[f"{horizon}min"].detach().cpu().numpy()
                pred_weather_batches.append(pw)
                pred_traffic_batches.append(pt)
                pred_persistence_batches.append(X[start:end, -1, :])

            predictions_scaled = {
                "weather_graph_aware": np.concatenate(pred_weather_batches, axis=0),
                "traffic_only": np.concatenate(pred_traffic_batches, axis=0),
                "persistence": np.concatenate(pred_persistence_batches, axis=0),
            }

                                # Inverse-transform each sensor using the project's fitted scaler.
            truth_mph = scaler.inverse_transform(
                Y[:n].reshape(n, N_SENSORS)
            ).reshape(n, N_SENSORS)

            for model_name, pred_scaled in predictions_scaled.items():
                pred_mph = scaler.inverse_transform(
                    pred_scaled.reshape(-1, N_SENSORS)
                ).reshape(n, N_SENSORS)

                metrics = calculate_metrics(pred_mph, truth_mph, mask)
                rows.append({
                    "model": model_name,
                    "horizon": f"{horizon}min",
                    **metrics,
                })

    df = pd.DataFrame(rows)
    audit = {
        "evaluation": "matched_test_weather_vs_traffic_only",
        "device": str(device),
        "checkpoints": checkpoint_audit,
        "n_test_input_samples": int(len(X)),
        "n_weather_samples": int(len(W)),
        "n_sensors": N_SENSORS,
        "input_steps": INPUT_STEPS,
        "sample_interval_minutes": SAMPLE_MINUTES,
        "test_start_raw_row": test_start,
        "target_alignment_tolerance_scaled": ALIGNMENT_TOLERANCE,
        "alignment_checks": alignment_audit,
        "mask_definition": "finite and non-zero raw METR-LA observations, aligned by sensor IDs",
        "metrics_units": "mph",
        "notes": [
            "No training was performed.",
            "Both checkpoint state dictionaries loaded with strict=True.",
            "Each horizon used n=min(len(X), len(W), len(Y)) and its saved target prefix.",
            "Targets were checked against X[steps:steps+check_n, -1, :] before metrics.",
            "Weather-aware and traffic-only models were evaluated on identical samples and masks within each horizon.",
            "Persistence predicts the last observed input frame.",
            "A model comparison alone does not isolate weather causality from differences in training runs or checkpoint selection.",
        ],
    }

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwrites. If either path exists, fail before
    # inference at startup; exclusive mode also protects against race conditions.
    with OUT_CSV.open("x", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
    try:
        with OUT_AUDIT.open("x", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
    except Exception:
        # If audit creation fails after CSV creation, don't delete or overwrite the
        # CSV automatically. Report exactly what was created for manual recovery.
        print(f"WARNING: CSV created at {OUT_CSV}; audit write failed.", file=sys.stderr)
        raise

    print("\nMatched test results (mph):")
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_CSV.relative_to(ROOT)}")
    print(f"Saved: {OUT_AUDIT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"EVALUATION STOPPED SAFELY: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
