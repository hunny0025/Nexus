"""
core/learning/tft_model.py

Temporal Fusion Transformer for Indian Railways delay cascade prediction.

Architecture follows the 2021 Google Brain paper (Lim et al., NeurIPS 2021):
  - Variable Selection Networks for feature gating
  - Gated Residual Networks (GRN) throughout
  - LSTM-based sequence encoder
  - Multi-head Self-Attention with interpretable attention weights
  - Quantile output (p10, p50, p90) --- enables uncertainty-aware decisions

Training target: predict delay at time t+1 ... t+6 (next 6 hours) given
    - past 24h delay sequence for this station
    - known covariates: hour, day-of-week, month, is_junction, train_count

Usage:
    from core.learning.tft_model import TFTTrainer
    trainer = TFTTrainer()
    trainer.train("data/historical/delays.csv")
    trainer.save("data/models/tft.pt")

    model = TFTTrainer.load("data/models/tft.pt")
    preds = model.predict(station_id="NDLS", recent_delays=[...])
"""

import os
import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Hyper-parameters ----------------------------------------------------------
LOOKBACK   = 24   # hours of history fed to encoder
HORIZON    = 6    # hours ahead to forecast
D_MODEL    = 64   # hidden dimension
N_HEADS    = 4    # attention heads
DROPOUT    = 0.1
QUANTILES  = [0.1, 0.5, 0.9]
BATCH_SIZE = 256
EPOCHS     = 40
LR         = 1e-3

# -- Gated Residual Network ----------------------------------------------------

class GRN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=DROPOUT):
        super().__init__()
        self.fc1  = nn.Linear(d_in, d_hidden)
        self.fc2  = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h  = F.elu(self.fc1(x))
        h  = self.drop(h)
        g  = torch.sigmoid(self.gate(h))
        h2 = self.fc2(h)
        return self.norm(g * h2 + (1 - g) * self.skip(x))

# -- Variable Selection Network ------------------------------------------------

class VSN(nn.Module):
    """Selects which input features are relevant at each timestep."""
    def __init__(self, n_features, d_model):
        super().__init__()
        self.feature_grns = nn.ModuleList([GRN(1, d_model, d_model) for _ in range(n_features)])
        self.softmax_grn  = GRN(n_features * d_model, d_model, n_features)

    def forward(self, x):
        # x: (B, T, n_features)
        B, T, nF = x.shape
        processed = []
        for i, grn in enumerate(self.feature_grns):
            processed.append(grn(x[..., i:i+1]))      # each: (B, T, d_model)
        stacked = torch.stack(processed, dim=-2)       # (B, T, nF, d_model)

        # Softmax gate operates on concatenated processed features
        concat = torch.cat(processed, dim=-1)          # (B, T, nF * d_model)
        weights = F.softmax(self.softmax_grn(concat), dim=-1)  # (B, T, nF)

        # Weighted sum over feature dimension
        out = (stacked * weights.unsqueeze(-1)).sum(dim=-2)  # (B, T, d_model)
        return out, weights

# -- TFT Core ------------------------------------------------------------------

class TFT(nn.Module):
    """Temporal Fusion Transformer (simplified but publication-faithful)."""

    def __init__(self, n_past_features=6, n_future_features=3, d_model=D_MODEL,
                 n_heads=N_HEADS, n_quantiles=len(QUANTILES)):
        super().__init__()
        self.d_model = d_model

        # Encoders
        self.past_vsn   = VSN(n_past_features, d_model)
        self.future_vsn = VSN(n_future_features, d_model)

        # LSTM encoder for past observations
        self.encoder_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        # LSTM decoder for future known covariates
        self.decoder_lstm = nn.LSTM(d_model, d_model, batch_first=True)

        # Static enrichment (station embedding)
        self.n_stations = 16   # updated dynamically
        self.station_emb = nn.Embedding(self.n_stations + 1, d_model)
        self.static_grn  = GRN(d_model, d_model, d_model)

        # Temporal self-attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=DROPOUT, batch_first=True)
        self.attn_grn = GRN(d_model, d_model, d_model)

        # Output projection -> quantile predictions per horizon step
        self.out_fc = nn.Linear(d_model, n_quantiles)

    def forward(self, past_x, future_x, station_idx):
        """
        past_x   : (B, LOOKBACK, n_past_features)
        future_x : (B, HORIZON,  n_future_features)
        station_idx: (B,)
        Returns  : (B, HORIZON, n_quantiles)
        """
        B = past_x.size(0)

        # Static context from station embedding
        s_emb = self.station_emb(station_idx)           # (B, d)
        s_ctx = self.static_grn(s_emb).unsqueeze(1)    # (B, 1, d)

        # Variable selection
        past_enc,   _ = self.past_vsn(past_x)           # (B, L, d)
        future_enc, _ = self.future_vsn(future_x)       # (B, H, d)

        # Enrich with static context
        past_enc   = past_enc   + s_ctx
        future_enc = future_enc + s_ctx

        # LSTM encode past -> decode future
        enc_out, state   = self.encoder_lstm(past_enc)
        dec_out, _       = self.decoder_lstm(future_enc, state)

        # Concatenate all temporal representations for attention
        all_seq = torch.cat([enc_out, dec_out], dim=1)  # (B, L+H, d)

        # Temporal self-attention (decoder queries over full sequence)
        attn_out, attn_weights = self.attn(dec_out, all_seq, all_seq)
        attn_out = self.attn_grn(attn_out + dec_out)    # residual

        # Per-timestep quantile output: (B, H, d) -> (B, H, n_q)
        out = self.out_fc(attn_out)
        return out, attn_weights

# -- Dataset -------------------------------------------------------------------

class DelayDataset(Dataset):
    def __init__(self, records, station2idx):
        """
        records: list of dicts from delays.csv grouped by station
        Each sample: LOOKBACK past + HORIZON future
        """
        self.samples = []
        self.station2idx = station2idx

        # Group by station
        by_station = defaultdict(list)
        for r in records:
            by_station[r["station_id"]].append(r)

        for sid, rows in by_station.items():
            rows.sort(key=lambda x: x["timestamp"])
            n = len(rows)
            idx = station2idx.get(sid, 0)

            for i in range(n - LOOKBACK - HORIZON):
                past  = rows[i : i + LOOKBACK]
                future = rows[i + LOOKBACK : i + LOOKBACK + HORIZON]

                past_x = np.array([
                    [float(r["delay_min"]) / 180.0,
                     float(r["hour"]) / 23.0,
                     float(r["day_of_week"]) / 6.0,
                     float(r["month"]) / 12.0,
                     float(r["is_junction"]),
                     float(float(r["delay_min"]) > 15)]      # binary: significant delay
                    for r in past
                ], dtype=np.float32)  # (LOOKBACK, 6)

                future_x = np.array([
                    [float(r["hour"]) / 23.0,
                     float(r["day_of_week"]) / 6.0,
                     float(r["month"]) / 12.0]
                    for r in future
                ], dtype=np.float32)  # (HORIZON, 3)

                # Target: normalised delays at each future horizon
                target = np.array(
                    [min(float(r["delay_min"]) / 180.0, 1.0) for r in future],
                    dtype=np.float32
                )  # (HORIZON,)

                self.samples.append((past_x, future_x, idx, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        past_x, future_x, sidx, target = self.samples[i]
        return (torch.tensor(past_x),
                torch.tensor(future_x),
                torch.tensor(sidx, dtype=torch.long),
                torch.tensor(target))

# -- Quantile Loss -------------------------------------------------------------

def quantile_loss(preds, target, quantiles=QUANTILES):
    """
    preds  : (B, HORIZON, n_q)
    target : (B, HORIZON)
    """
    target = target.unsqueeze(-1)   # (B, H, 1)
    qs = torch.tensor(quantiles, device=preds.device).view(1, 1, -1)
    err = target - preds
    loss = torch.max(qs * err, (qs - 1) * err)
    return loss.mean()

# -- Trainer -------------------------------------------------------------------

class TFTTrainer:
    def __init__(self):
        self.model = None
        self.station2idx = {}
        self.metrics = {}

    def train(self, delays_csv: str, epochs=EPOCHS, max_rows=None):
        import csv
        print("Loading delay data ...")
        records = []
        with open(delays_csv, newline="") as f:
            for row in csv.DictReader(f):
                records.append(row)

        if max_rows and len(records) > max_rows:
            # Subsample evenly across the dataset
            step = len(records) // max_rows
            records = records[::step][:max_rows]
            print(f"   Subsampled to {len(records):,} records (--max-rows {max_rows})")

        stations = sorted(set(r["station_id"] for r in records))
        self.station2idx = {s: i for i, s in enumerate(stations)}

        print(f"   {len(records):,} records | {len(stations)} stations")
        print("Building dataset ...")
        dataset = DelayDataset(records, self.station2idx)
        print(f"   {len(dataset):,} training windows")

        split = int(0.85 * len(dataset))
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [split, len(dataset) - split])

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        self.model = TFT(
            n_past_features=6,
            n_future_features=3,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_quantiles=len(QUANTILES),
        ).to(DEVICE)

        # Resize station embedding
        n_st = len(stations)
        self.model.station_emb = nn.Embedding(n_st + 1, D_MODEL).to(DEVICE)
        self.model.n_stations  = n_st

        optimiser = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

        best_val  = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            # -- Train --
            self.model.train()
            train_loss = 0.0
            for past_x, future_x, sidx, target in train_dl:
                past_x   = past_x.to(DEVICE)
                future_x = future_x.to(DEVICE)
                sidx     = sidx.to(DEVICE)
                target   = target.to(DEVICE)

                preds, _ = self.model(past_x, future_x, sidx)
                loss = quantile_loss(preds, target)

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                train_loss += loss.item()

            scheduler.step()
            train_loss /= len(train_dl)

            # -- Validate --
            self.model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for past_x, future_x, sidx, target in val_dl:
                    past_x   = past_x.to(DEVICE)
                    future_x = future_x.to(DEVICE)
                    sidx     = sidx.to(DEVICE)
                    target   = target.to(DEVICE)
                    preds, _ = self.model(past_x, future_x, sidx)
                    val_loss += quantile_loss(preds, target).item()
                    # p50 for MAE
                    all_preds.append(preds[:, :, 1].cpu().numpy())
                    all_targets.append(target.cpu().numpy())

            val_loss /= len(val_dl)
            preds_np   = np.concatenate(all_preds)    * 180
            targets_np = np.concatenate(all_targets)  * 180
            mae = float(np.mean(np.abs(preds_np - targets_np)))

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3}/{epochs}  "
                      f"train_ql={train_loss:.4f}  val_ql={val_loss:.4f}  "
                      f"MAE={mae:.2f} min")

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)
        self.metrics = {"best_val_ql": round(best_val, 4), "mae_minutes": round(mae, 2)}
        print(f"\n[OK] TFT training complete  |  best val_ql={best_val:.4f}  MAE={mae:.1f} min")
        return self.metrics

    def predict(self, station_id: str, recent_delays: list, future_hours: list = None):
        """
        station_id    : e.g. "NDLS"
        recent_delays : list of dicts [{delay_min, hour, day_of_week, month, is_junction}, ...]
                        Must have at least LOOKBACK (24) entries.
        future_hours  : list of {hour, day_of_week, month} for next HORIZON steps.
                        If None, auto-generated.
        Returns dict with p10/p50/p90 forecasts in minutes.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .train() or .load() first.")

        recent = recent_delays[-LOOKBACK:]
        while len(recent) < LOOKBACK:
            recent = [recent[0]] + recent   # pad left

        if future_hours is None:
            last = recent[-1]
            future_hours = [
                {"hour": (last["hour"] + i) % 24,
                 "day_of_week": last.get("day_of_week", 0),
                 "month": last.get("month", 6)}
                for i in range(1, HORIZON + 1)
            ]

        past_x = np.array([
            [min(float(r.get("delay_min", 0)) / 180.0, 1.0),
             float(r.get("hour", 12)) / 23.0,
             float(r.get("day_of_week", 0)) / 6.0,
             float(r.get("month", 6)) / 12.0,
             float(r.get("is_junction", 0)),
             float(float(r.get("delay_min", 0)) > 15)]
            for r in recent
        ], dtype=np.float32)

        future_x = np.array([
            [float(h["hour"]) / 23.0,
             float(h["day_of_week"]) / 6.0,
             float(h["month"]) / 12.0]
            for h in future_hours
        ], dtype=np.float32)

        sidx = self.station2idx.get(station_id, 0)

        self.model.eval()
        with torch.no_grad():
            pt = torch.tensor(past_x).unsqueeze(0).to(DEVICE)
            ft = torch.tensor(future_x).unsqueeze(0).to(DEVICE)
            st = torch.tensor([sidx], dtype=torch.long).to(DEVICE)
            preds, attn_weights = self.model(pt, ft, st)

        preds_min = (preds[0].cpu().numpy() * 180).clip(0, 180)  # (H, 3)
        # attn_weights: (B, H, L+H) -> average over queries -> (L+H,)
        attn_np = attn_weights[0].cpu().numpy().mean(axis=0)      # (L+H,)

        return {
            "station_id": station_id,
            "horizon_hours": HORIZON,
            "p10": preds_min[:, 0].tolist(),
            "p50": preds_min[:, 1].tolist(),
            "p90": preds_min[:, 2].tolist(),
            "attention_weights": attn_np.tolist(),
            "interpretation": _interpret_attention(attn_np, recent),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict":  self.model.state_dict(),
            "station2idx": self.station2idx,
            "metrics":     self.metrics,
            "config": {"d_model": D_MODEL, "n_heads": N_HEADS,
                       "lookback": LOOKBACK, "horizon": HORIZON},
        }, path)
        print(f"[OK] TFT saved -> {path}")

    @classmethod
    def load(cls, path: str):
        obj = cls()
        ckpt = torch.load(path, map_location=DEVICE)
        obj.station2idx = ckpt["station2idx"]
        obj.metrics     = ckpt.get("metrics", {})
        n_st = len(obj.station2idx)
        obj.model = TFT(n_past_features=6, n_future_features=3,
                        d_model=D_MODEL, n_heads=N_HEADS).to(DEVICE)
        obj.model.station_emb = nn.Embedding(n_st + 1, D_MODEL).to(DEVICE)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.model.eval()
        print(f"[OK] TFT loaded from {path}  |  metrics={obj.metrics}")
        return obj


def _interpret_attention(attn_weights, recent_delays):
    """Return human-readable interpretation of what the model is attending to."""
    n = len(attn_weights)
    if n == 0:
        return "No attention data."
    peak_t = int(np.argmax(attn_weights))
    if peak_t < len(recent_delays):
        rec = recent_delays[peak_t]
        return (f"Model focusing most on t-{len(recent_delays)-peak_t}h "
                f"(delay={rec.get('delay_min',0):.0f} min at hour {rec.get('hour','?')})")
    return f"Model attending to future step t+{peak_t - len(recent_delays) + 1}h"
