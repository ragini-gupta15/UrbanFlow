#!/usr/bin/env python3
"""
UrbanFlow controlled weather ablation.

Trains a traffic-only counterpart to WeatherGraphAwareUrbanFlow using the
existing common-prefix train/validation splits and the same training protocol.
It never edits notebooks, source model files, input arrays, or existing outputs.

Run from the Urbanflow repository root:
    ./.venv/bin/python scripts/run_weather_ablation.py

This script is deliberately fail-closed: if any required input/output path is
missing or an output already exists, it exits before training.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models" / "experiments"
RESULTS = ROOT / "results" / "experiments"

HORIZONS = ("15min", "30min", "45min", "60min")
HORIZON_MINUTES = {"15min": 15, "30min": 30, "45min": 45, "60min": 60}
N_TRAIN, N_VAL = 23961, 3398
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 64, 20, 4
LEARNING_RATE, HUBER_DELTA, SEED = 5e-4, 1.0, 42
D_MODEL, NHEAD, N_LAYERS, FF_DIM = 128, 4, 2, 256
DROPOUT = 0.0

MODEL_OUT = MODELS / "weather_ablation_traffic_only_best.pth"
HISTORY_OUT = RESULTS / "weather_ablation_traffic_only_training.csv"
VAL_OUT = RESULTS / "weather_ablation_traffic_only_validation.csv"
AUDIT_OUT = RESULTS / "weather_ablation_traffic_only_audit.json"

def require_files(paths):
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Ablation preflight failed; no training was started. Missing files:\n"
            + "\n".join(missing)
        )

def refuse_existing(paths):
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError(
            "Ablation stopped before training because these outputs already exist. "
            "Nothing was overwritten:\n" + "\n".join(existing)
        )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class WeatherGraphAwareUrbanFlow(nn.Module):
    """Architecture reproduced from the existing graph-aware experiment."""
    def __init__(self, adjacency, use_weather: bool):
        super().__init__()
        self.num_sensors = 207
        self.d_model = D_MODEL
        self.use_weather = use_weather
        self.traffic_projection = nn.Linear(1, D_MODEL)
        self.position_encoding = PositionalEncoding(D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=FF_DIM,
            dropout=DROPOUT, activation="gelu", batch_first=True
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
            nn.LayerNorm(D_MODEL)
        )
        self.head_15 = nn.Linear(D_MODEL, 1)
        self.head_30 = nn.Linear(D_MODEL, 1)
        self.head_45 = nn.Linear(D_MODEL, 1)
        self.head_60 = nn.Linear(D_MODEL, 1)
        self.register_buffer("graph_mask", adjacency.float())

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

def masked_huber_loss(pred, target, mask, delta=HUBER_DELTA):
    mask_f = mask.to(dtype=pred.dtype)
    error = (pred - target).abs()
    quadratic = torch.minimum(error, torch.tensor(delta, device=pred.device))
    linear = error - quadratic
    loss = (0.5 * quadratic.square() + delta * linear) * mask_f
    count = mask_f.sum()
    if count.item() == 0:
        raise RuntimeError("A target batch contains no valid masked observations.")
    return loss.sum() / count

def metrics(pred, target, mask):
    valid = mask.bool() & torch.isfinite(target) & torch.isfinite(pred)
    count = int(valid.sum().item())
    if count == 0:
        raise RuntimeError("No valid values available for metric calculation.")
    err = pred[valid] - target[valid]
    return {
        "mae": float(err.abs().mean().item()),
        "rmse": float(torch.sqrt(err.square().mean()).item()),
        "valid_observations": count,
    }

def load_array(path):
    arr = np.load(path, allow_pickle=False)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in input array: {path}")
    return arr

def main():
    output_paths = [MODEL_OUT, HISTORY_OUT, VAL_OUT, AUDIT_OUT]
    refuse_existing(output_paths)
    for p in output_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
    # Re-check after directory creation, before any expensive work.
    refuse_existing(output_paths)

    required = [
        DATA / "train_sequences.npz",
        DATA / "validation_sequences.npz",
        DATA / "weather_train_scaled.npy",
        DATA / "weather_validation_scaled.npy",
        DATA / "adjacency_matrix.npy",
    ]
    for h in HORIZONS:
        required.extend([
            DATA / f"y_train_{h}.npy",
            DATA / f"y_val_{h}.npy",
            DATA / f"train_target_mask_{h}.npy",
            DATA / f"val_target_mask_{h}.npy",
        ])
    require_files(required)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    if device.type == "mps":
        torch.mps.manual_seed(SEED)

    train_npz = np.load(DATA / "train_sequences.npz", allow_pickle=False)
    val_npz = np.load(DATA / "validation_sequences.npz", allow_pickle=False)
    x_train = np.asarray(train_npz["X"][:N_TRAIN], dtype=np.float32)
    x_val = np.asarray(val_npz["X"][:N_VAL], dtype=np.float32)
    w_train = np.asarray(load_array(DATA / "weather_train_scaled.npy")[:N_TRAIN], dtype=np.float32)
    w_val = np.asarray(load_array(DATA / "weather_validation_scaled.npy")[:N_VAL], dtype=np.float32)
    adjacency = np.asarray(load_array(DATA / "adjacency_matrix.npy"), dtype=np.float32)

    if x_train.shape != (N_TRAIN, 12, 207) or x_val.shape != (N_VAL, 12, 207):
        raise ValueError(f"Unexpected traffic shapes: train={x_train.shape}, val={x_val.shape}")
    if w_train.shape != (N_TRAIN, 12, 6) or w_val.shape != (N_VAL, 12, 6):
        raise ValueError(f"Unexpected scaled weather shapes: train={w_train.shape}, val={w_val.shape}")
    if adjacency.shape != (207, 207):
        raise ValueError(f"Unexpected adjacency shape: {adjacency.shape}")

    y_train, y_val, m_train, m_val = {}, {}, {}, {}
    for h in HORIZONS:
        # Targets may contain NaN at invalid sensor observations; their saved masks
        # determine which entries contribute to loss/metrics. Do not reject masked NaNs.
        y_train[h] = np.asarray(np.load(DATA / f"y_train_{h}.npy", allow_pickle=False)[:N_TRAIN], dtype=np.float32)
        y_val[h] = np.asarray(np.load(DATA / f"y_val_{h}.npy", allow_pickle=False)[:N_VAL], dtype=np.float32)
        m_train[h] = np.asarray(np.load(DATA / f"train_target_mask_{h}.npy", allow_pickle=False)[:N_TRAIN], dtype=bool)
        m_val[h] = np.asarray(np.load(DATA / f"val_target_mask_{h}.npy", allow_pickle=False)[:N_VAL], dtype=bool)
        if y_train[h].shape != (N_TRAIN, 207) or m_train[h].shape != y_train[h].shape:
            raise ValueError(f"Train target/mask shape mismatch for {h}")
        if y_val[h].shape != (N_VAL, 207) or m_val[h].shape != y_val[h].shape:
            raise ValueError(f"Validation target/mask shape mismatch for {h}")
        if not np.isfinite(y_train[h][m_train[h]]).all():
            raise ValueError(f"Non-finite unmasked training targets for {h}")
        if not np.isfinite(y_val[h][m_val[h]]).all():
            raise ValueError(f"Non-finite unmasked validation targets for {h}")

    # Match notebook's row-normalized adjacency.
    adjacency = np.maximum(adjacency, 0.0)
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency_norm = adjacency / (row_sums + 1e-8)

    # Keep CPU tensors and move only batches to device.
    xt = torch.from_numpy(x_train)
    xv = torch.from_numpy(x_val)
    yt = {h: torch.from_numpy(y_train[h]) for h in HORIZONS}
    yv = {h: torch.from_numpy(y_val[h]) for h in HORIZONS}
    mt = {h: torch.from_numpy(m_train[h]) for h in HORIZONS}
    mv = {h: torch.from_numpy(m_val[h]) for h in HORIZONS}

    model = WeatherGraphAwareUrbanFlow(torch.from_numpy(adjacency_norm), use_weather=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    best_val_mae_30 = float("inf")
    best_state = None
    best_epoch = None
    stale = 0
    print(f"Device: {device}; traffic-only ablation; parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Train={N_TRAIN}; validation={N_VAL}; batch={BATCH_SIZE}; max epochs={MAX_EPOCHS}; patience={PATIENCE}")
    print("Test data will not be loaded or evaluated in this training run.")

    for epoch in range(1, MAX_EPOCHS + 1):
        start_time = time.time()
        model.train()
        perm = torch.randperm(N_TRAIN)
        loss_sum = 0.0
        batches = 0
        for start in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            traffic = xt[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(traffic)
            losses = [
                masked_huber_loss(pred[h], yt[h][idx].to(device), mt[h][idx].to(device))
                for h in HORIZONS
            ]
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1

        model.eval()
        val_preds = {h: [] for h in HORIZONS}
        with torch.no_grad():
            for start in range(0, N_VAL, BATCH_SIZE):
                end = min(start + BATCH_SIZE, N_VAL)
                pred = model(xv[start:end].to(device))
                for h in HORIZONS:
                    val_preds[h].append(pred[h].cpu())
        epoch_row = {"epoch": epoch, "train_loss": loss_sum / batches}
        epoch_metrics = {}
        for h in HORIZONS:
            pred_h = torch.cat(val_preds[h], dim=0)
            met = metrics(pred_h, yv[h], mv[h])
            epoch_metrics[h] = met
            epoch_row[f"mae_{h}"] = met["mae"]
            epoch_row[f"rmse_{h}"] = met["rmse"]
        history.append(epoch_row)
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | loss={epoch_row['train_loss']:.4f} | "
            + " | ".join(f"{h} MAE={epoch_metrics[h]['mae']:.4f}" for h in HORIZONS)
            + f" | {time.time() - start_time:.1f}s"
        )

        val_mae_30 = epoch_metrics["30min"]["mae"]
        if val_mae_30 < best_val_mae_30:
            best_val_mae_30 = val_mae_30
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"Early stopping after epoch {epoch}; best epoch={best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No valid traffic-only checkpoint was selected.")
    # Save only after successful training. Outputs were preflighted as absent.
    torch.save({
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_mae_30": best_val_mae_30,
        "architecture": "WeatherGraphAwareUrbanFlowTrafficOnly",
        "config": {
            "num_sensors": 207, "d_model": D_MODEL, "nhead": NHEAD,
            "num_layers": N_LAYERS, "dim_feedforward": FF_DIM,
            "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "horizons": list(HORIZONS),
            "loss": "masked_huber_delta_1.0",
            "gradient_clip_norm": 1.0,
            "selection_metric": "30min validation MAE",
            "seed": SEED, "weather_input": False,
            "train_samples": N_TRAIN, "validation_samples": N_VAL,
        },
        "test_used_for_selection": False,
    }, MODEL_OUT)
    pd.DataFrame(history).to_csv(HISTORY_OUT, index=False)

    # Restore best state and produce the selected checkpoint's validation metrics.
    model.load_state_dict(best_state)
    model.eval()
    val_preds = {h: [] for h in HORIZONS}
    with torch.no_grad():
        for start in range(0, N_VAL, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_VAL)
            pred = model(xv[start:end].to(device))
            for h in HORIZONS:
                val_preds[h].append(pred[h].cpu())
    val_rows = []
    for h in HORIZONS:
        met = metrics(torch.cat(val_preds[h]), yv[h], mv[h])
        val_rows.append({"model": "traffic_only", "horizon": h, **met})
    pd.DataFrame(val_rows).to_csv(VAL_OUT, index=False)

    audit = {
        "experiment": "controlled_weather_ablation_traffic_only",
        "status": "trained_validation_evaluated",
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_validation_mae_30": float(best_val_mae_30),
        "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "training_config": {
            "seed": SEED, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "learning_rate": LEARNING_RATE,
            "loss": "masked_huber_delta_1.0", "gradient_clip_norm": 1.0,
        },
        "data": {
            "train_samples": N_TRAIN, "validation_samples": N_VAL,
            "traffic_train_shape": list(x_train.shape),
            "traffic_validation_shape": list(x_val.shape),
            "weather_train_shape_checked_but_not_used_by_model": list(w_train.shape),
            "weather_validation_shape_checked_but_not_used_by_model": list(w_val.shape),
            "horizons": list(HORIZONS),
            "adjacency_normalization": "row_normalized A/(row_sum+1e-8)",
        },
        "test_used_for_selection": False,
        "test_evaluated": False,
        "outputs": {
            "checkpoint": str(MODEL_OUT.relative_to(ROOT)),
            "training_history": str(HISTORY_OUT.relative_to(ROOT)),
            "validation_metrics": str(VAL_OUT.relative_to(ROOT)),
        },
        "notes": [
            "Existing notebook, main checkpoint, weather-model checkpoint, and input arrays were not modified.",
            "Traffic-only model has no weather encoder or weather contribution to fusion.",
            "Architecture parameter count differs because the weather branch is removed.",
            "Held-out test evaluation is intentionally a separate next step after reviewing validation output.",
        ],
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("\nTraffic-only ablation training completed.")
    print(f"Best epoch: {best_epoch}; best 30-min validation MAE: {best_val_mae_30:.4f}")
    print(f"Checkpoint: {MODEL_OUT}")
    print(f"Validation metrics: {VAL_OUT}")
    print(f"Audit: {AUDIT_OUT}")
    print("Held-out test set was not loaded or evaluated.")

if __name__ == "__main__":
    main()
