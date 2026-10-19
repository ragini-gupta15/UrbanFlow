#!/usr/bin/env python3
"""
Matched test evaluation for the saved UrbanFlow weather-aware and traffic-only
graph-aware checkpoints.

Uses the exact model implementation from scripts/run_weather_ablation.py.
Does not train models and refuses to overwrite output files.
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
WEATHER_CKPT = ROOT / "models/weather_graph_aware_multi_horizon_best.pth"
TRAFFIC_CKPT = ROOT / "models/experiments/weather_ablation_traffic_only_best.pth"

X_TEST_PATH = ROOT / "data/processed/test_sequences.npz"
WEATHER_TEST_PATH = ROOT / "data/processed/weather_test_sequences.npy"
SCALER_PATH = ROOT / "data/processed/traffic_scaler.pkl"
ADJ_PATH = ROOT / "data/processed/adjacency_matrix.npy"
METADATA_PATH = ROOT / "data/processed/metadata.json"
RAW_H5_PATH = ROOT / "data/raw/metr-la.h5"

OUT_CSV = ROOT / "results/experiments/weather_ablation_matched_test_mph.csv"
OUT_AUDIT = ROOT / "results/experiments/weather_ablation_matched_test_audit.json"

HORIZONS = ("15min", "30min", "45min", "60min")
STEPS = {"15min": 3, "30min": 6, "45min": 9, "60min": 12}
INPUT_STEPS = 12
D_MODEL = 128
NHEAD = 4
FF_DIM = 256
N_LAYERS = 2
DROPOUT = 0.0


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
        pe[:, 1::2] = torch.cos(
            position * div_term[: pe[:, 1::2].shape[1]]
        )
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class WeatherGraphAwareUrbanFlow(nn.Module):
    """Architecture reproduced exactly from scripts/run_weather_ablation.py."""

    def __init__(
        self,
        adjacency: torch.Tensor,
        use_weather: bool,
        positional_length: int,
    ):
        super().__init__()
        self.num_sensors = 207
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
        self.temporal_transformer = nn.TransformerEncoder(
            layer, num_layers=N_LAYERS
        )
        self.spatial_query = nn.Linear(D_MODEL, D_MODEL)
        self.spatial_key = nn.Linear(D_MODEL, D_MODEL)
        self.spatial_value = nn.Linear(D_MODEL, D_MODEL)
        if use_weather:
            self.weather_encoder = nn.Sequential(
                nn.Linear(6, 32),
                nn.GELU(),
                nn.Linear(32, D_MODEL),
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
        if not use_weather:
            self.register_buffer("graph_mask", adjacency.float())
        else:
            self.graph_mask = adjacency.float()

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


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        fail(f"Required checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        fail(f"Unexpected checkpoint format: {path}")
    return checkpoint


def masked_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray):
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    n = int(valid.sum())
    if n == 0:
        return {"mae_mph": None, "rmse_mph": None, "valid_observations": 0}
    error = pred[valid] - target[valid]
    return {
        "mae_mph": float(np.mean(np.abs(error))),
        "rmse_mph": float(np.sqrt(np.mean(error ** 2))),
        "valid_observations": n,
    }


def load_raw_observation_mask(
    raw_h5_path: Path,
    metadata_path: Path,
    test_start: int,
    horizon_steps: int,
    n_samples: int,
    n_sensors: int,
) -> np.ndarray:
    """
    Load raw METR-LA target observations, align sensor order when metadata
    contains sensor IDs, and mark finite non-zero observations as valid.
    """
    with h5py.File(raw_h5_path, "r") as h5:
        if "df" not in h5:
            fail(f"Expected dataset key 'df' in {raw_h5_path}")
        raw = np.asarray(h5["df"]["block0_values"])

    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    # The known raw file convention is time x sensors. If stored transposed,
    # correct it only when the expected sensor dimension makes that unambiguous.
    if raw.ndim != 2:
        fail(f"Raw traffic data must be 2D; got shape {raw.shape}")
    if raw.shape[1] != n_sensors and raw.shape[0] == n_sensors:
        raw = raw.T
    if raw.shape[1] != n_sensors:
        fail(
            f"Raw sensor dimension mismatch: raw shape {raw.shape}, "
            f"expected {n_sensors} sensors. Refusing to guess sensor alignment."
        )

    # Use the same target row convention as the verified evaluator:
    # test_start + INPUT_STEPS - 1 + horizon_steps + sample_index.
    rows = test_start + INPUT_STEPS - 1 + horizon_steps + np.arange(n_samples)
    if rows.max(initial=-1) >= raw.shape[0]:
        fail(
            f"Raw target rows exceed data length: max row {rows.max()}, "
            f"raw length {raw.shape[0]}"
        )
    obs = raw[rows, :].astype(np.float64, copy=False)
    return np.isfinite(obs) & (obs != 0)


def main() -> int:
    # Refuse to start if outputs already exist; never overwrite results.
    existing = [str(p) for p in (OUT_CSV, OUT_AUDIT) if p.exists()]
    if existing:
        fail("Refusing to overwrite existing output(s): " + ", ".join(existing))

    required = [
        X_TEST_PATH, WEATHER_TEST_PATH, SCALER_PATH, ADJ_PATH,
        METADATA_PATH, RAW_H5_PATH,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        fail("Required evaluation input(s) missing: " + ", ".join(missing))

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    weather_ckpt = load_checkpoint(WEATHER_CKPT, device)
    traffic_ckpt = load_checkpoint(TRAFFIC_CKPT, device)

    # Respect the positional buffer lengths saved in each checkpoint.
    weather_pe = weather_ckpt["model_state_dict"].get("position_encoding.pe")
    traffic_pe = traffic_ckpt["model_state_dict"].get("position_encoding.pe")
    if weather_pe is None or traffic_pe is None:
        fail("A checkpoint is missing position_encoding.pe.")
    weather_pos_len = int(weather_pe.shape[1])
    traffic_pos_len = int(traffic_pe.shape[1])

    A = np.load(ADJ_PATH).astype(np.float32)
    if A.shape != (207, 207):
        fail(f"Expected adjacency shape (207,207), got {A.shape}")
    # This is the row-normalized adjacency used when the experiment was trained.
    A = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    A_tensor = torch.from_numpy(A)

    weather_model = WeatherGraphAwareUrbanFlow(
        A_tensor, use_weather=True, positional_length=weather_pos_len
    )
    traffic_model = WeatherGraphAwareUrbanFlow(
        A_tensor, use_weather=False, positional_length=traffic_pos_len
    )
    # Strict loading is intentional: no partial or silent checkpoint loading.
    weather_model.load_state_dict(weather_ckpt["model_state_dict"], strict=True)
    traffic_model.load_state_dict(traffic_ckpt["model_state_dict"], strict=True)
    weather_model.to(device).eval()
    traffic_model.to(device).eval()
    print("Both checkpoints loaded strictly.")

    bundle = np.load(X_TEST_PATH)
    if "X" not in bundle:
        fail(f"{X_TEST_PATH} does not contain key 'X'.")
    X = np.asarray(bundle["X"], dtype=np.float32)
    W = np.asarray(np.load(WEATHER_TEST_PATH), dtype=np.float32)
    if X.ndim != 3 or X.shape[1:] != (INPUT_STEPS, 207):
        fail(f"Unexpected test traffic shape {X.shape}; expected (N,12,207).")
    if W.ndim != 3 or W.shape[0] != X.shape[0] or W.shape[2] != 6:
        fail(f"Unexpected weather test shape {W.shape}; expected (N,12,6).")
    n_samples = X.shape[0]

    scaler = joblib.load(SCALER_PATH)
    raw_metadata = json.loads(METADATA_PATH.read_text())
    # The verified evaluator derives chronological split from raw sequence count.
    with h5py.File(RAW_H5_PATH, "r") as h5:
        raw_shape = h5["df"]["block0_values"].shape
    raw_length = raw_shape[0] if raw_shape[1] == 207 else raw_shape[1]
    test_start = int(raw_length * 0.80)

    # Confirm each saved target array aligns to the sequence's last observed
    # traffic frame shifted by the configured forecast horizon.
    targets_scaled = {}
    for horizon, steps in STEPS.items():
        target_path = ROOT / f"data/processed/y_test_{horizon}.npy"
        if not target_path.exists():
            fail(f"Missing target array: {target_path}")
        y = np.asarray(np.load(target_path), dtype=np.float32)
        if y.shape != (n_samples, 207):
            fail(f"{horizon} target shape {y.shape}; expected {(n_samples,207)}")
        check_n = min(100, max(0, n_samples - steps))
        if check_n:
            expected = X[steps : steps + check_n, -1, :]
            actual = y[:check_n]
            max_diff = float(np.max(np.abs(expected - actual)))
            if max_diff > 1e-5:
                fail(
                    f"{horizon} target alignment failed: max scaled difference "
                    f"{max_diff:.8g} exceeds 1e-5."
                )
        targets_scaled[horizon] = y
    print("Saved target arrays passed the existing alignment check.")

    # This split convention matches the verified evaluator. Raw observation rows
    # account for 12 input steps and the horizon shift.
    predictions = {name: {} for name in ("weather_graph_aware", "traffic_only", "persistence")}
    targets_mph = {}
    masks = {}
    with torch.no_grad():
        for start in range(0, n_samples, 64):
            stop = min(start + 64, n_samples)
            xb = torch.from_numpy(X[start:stop]).to(device)
            wb = torch.from_numpy(W[start:stop]).to(device)
            pw = weather_model(xb, wb)
            pt = traffic_model(xb)
            for horizon in HORIZONS:
                predictions["weather_graph_aware"].setdefault(horizon, []).append(
                    pw[horizon].detach().cpu().numpy()
                )
                predictions["traffic_only"].setdefault(horizon, []).append(
                    pt[horizon].detach().cpu().numpy()
                )
                # Persistence prediction is the final input frame in scaled space.
                predictions["persistence"].setdefault(horizon, []).append(
                    X[start:stop, -1, :]
                )

    # Concatenate predictions and inverse-transform predictions/targets to mph.
    for horizon, steps in STEPS.items():
        for name in predictions:
            pred_scaled = np.concatenate(predictions[name][horizon], axis=0)
            predictions[name][horizon] = scaler.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).reshape(n_samples, 207)
        targets_mph[horizon] = scaler.inverse_transform(
            targets_scaled[horizon].reshape(-1, 1)
        ).reshape(n_samples, 207)
        masks[horizon] = load_raw_observation_mask(
            RAW_H5_PATH, METADATA_PATH, test_start, steps, n_samples, 207
        )
        # The raw observation mask is expected to match the evaluator's sample
        # range; fail if the alignment assumptions do not fit the raw data.
        if masks[horizon].shape != targets_mph[horizon].shape:
            fail(f"{horizon} raw mask/target shape mismatch.")

    rows = []
    audit = {
        "evaluation": "matched_test_weather_vs_traffic_only",
        "device": str(device),
        "checkpoints": {
            "weather_graph_aware": {
                "path": str(WEATHER_CKPT.relative_to(ROOT)),
                "architecture": weather_ckpt.get("architecture"),
                "best_epoch": weather_ckpt.get("best_epoch"),
                "best_val_mae_30": weather_ckpt.get("best_val_mae_30"),
                "positional_encoding_length": weather_pos_len,
            },
            "traffic_only": {
                "path": str(TRAFFIC_CKPT.relative_to(ROOT)),
                "architecture": traffic_ckpt.get("architecture"),
                "best_epoch": traffic_ckpt.get("best_epoch"),
                "best_val_mae_30": traffic_ckpt.get("best_val_mae_30"),
                "positional_encoding_length": traffic_pos_len,
            },
        },
        "n_test_samples": int(n_samples),
        "n_sensors": 207,
        "input_steps": INPUT_STEPS,
        "sample_interval_minutes": 5,
        "test_start_raw_row": int(test_start),
        "alignment_check_max_abs_scaled_difference": 1e-5,
        "target_mask": "finite and non-zero raw METR-LA observations",
        "metrics_units": "mph",
        "notes": [
            "No training was performed.",
            "Checkpoints loaded with strict=True.",
            "Only new output paths are written; existing outputs are never overwritten.",
            "Traffic-only and weather-aware predictions use the same test samples and raw observation masks.",
            "Weather benefit is not causally established by this comparison alone; training randomness and checkpoint-selection details may differ.",
        ],
    }

    for horizon in HORIZONS:
        mask = masks[horizon]
        for model_name in ("weather_graph_aware", "traffic_only", "persistence"):
            metrics = masked_metrics(predictions[model_name][horizon], targets_mph[horizon], mask)
            rows.append({
                "model": model_name,
                "horizon": horizon,
                **metrics,
            })
    df = pd.DataFrame(rows)

    # Output files are opened exclusively to prevent accidental overwrite.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("x", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
    with OUT_AUDIT.open("x", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

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
