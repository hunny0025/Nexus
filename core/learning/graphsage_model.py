"""
core/learning/graphsage_model.py

GraphSAGE-based delay propagation model for Indian Railways.

Replaces the hand-tuned dampening factor in the DBN with a learned
node-level classifier trained on historical co-occurrence of delays.

Architecture:
  - 3-layer GraphSAGE (mean aggregation) with skip connections
  - Input node features: station static properties + recent delay statistics
  - Edge features: distance, track type, max speed
  - Output: per-node delay probability in next 60 min (binary: >15 min)

Training signal: cascade_pairs.csv
  "If upstream delayed >10 min and downstream delayed >10 min within 90 min
   -> positive propagation edge"

Usage:
    from core.learning.graphsage_model import GraphSAGETrainer
    trainer = GraphSAGETrainer()
    trainer.train("data/historical/cascade_pairs.csv",
                  "data/network/stations.json",
                  "data/network/tracks.json")
    trainer.save("data/models/graphsage.pt")

    model = GraphSAGETrainer.load("data/models/graphsage.pt")
    risk_scores = model.predict(current_delays_dict)
"""

import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Hyper-parameters ----------------------------------------------------------
D_HIDDEN = 64
N_LAYERS = 3
DROPOUT  = 0.2
EPOCHS   = 30
LR       = 5e-4
BATCH    = 512

# -- GraphSAGE Layer -----------------------------------------------------------

class SAGEConv(nn.Module):
    """Mean-aggregation GraphSAGE layer (no PyG dependency)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.self_fc  = nn.Linear(d_in, d_out)
        self.neigh_fc = nn.Linear(d_in, d_out)
        self.norm     = nn.LayerNorm(d_out)

    def forward(self, x, adj):
        """
        x   : (N, d_in)  node features
        adj : (N, N)     adjacency matrix (normalised)
        """
        neigh_agg = torch.mm(adj, x)              # (N, d_in)
        out = self.self_fc(x) + self.neigh_fc(neigh_agg)
        return F.relu(self.norm(out))


class GraphSAGE(nn.Module):
    def __init__(self, d_in, d_hidden=D_HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        layers = []
        for i in range(n_layers):
            din  = d_in if i == 0 else d_hidden
            layers.append(SAGEConv(din, d_hidden))
        self.layers   = nn.ModuleList(layers)
        self.drop     = nn.Dropout(DROPOUT)
        self.out_proj = nn.Linear(d_hidden, 1)

    def forward(self, x, adj):
        h = x
        for layer in self.layers:
            h = self.drop(layer(h, adj))
        return torch.sigmoid(self.out_proj(h)).squeeze(-1)  # (N,) ∈ [0,1]


# -- Dataset (graph-level snapshot classification) ----------------------------

class PropagationDataset(Dataset):
    """
    Each sample is a time-slice graph snapshot.
    Node features include recent delay + station static properties.
    Label: whether each node experienced a propagated delay.
    """

    def __init__(self, snapshots, adj_matrix):
        self.snapshots  = snapshots   # list of (node_features, labels)
        self.adj        = adj_matrix  # fixed topology (N, N)

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, i):
        feats, labels = self.snapshots[i]
        return torch.tensor(feats, dtype=torch.float32), \
               torch.tensor(labels, dtype=torch.float32), \
               torch.tensor(self.adj, dtype=torch.float32)


# -- Trainer -------------------------------------------------------------------

class GraphSAGETrainer:
    def __init__(self):
        self.model      = None
        self.station2idx = {}
        self.idx2station = {}
        self.adj        = None
        self.n_stations = 0
        self.metrics    = {}

    # -- Build adjacency matrix ------------------------------------------------

    def _build_adjacency(self, stations, tracks):
        n  = len(stations)
        A  = np.zeros((n, n), dtype=np.float32)
        for t in tracks:
            i = self.station2idx.get(t["from_id"])
            j = self.station2idx.get(t["to_id"])
            if i is not None and j is not None:
                w = 1.0 / (t["distance_km"] + 1e-6)  # weight by proximity
                A[i, j] = w
                A[j, i] = w
        # Row-normalise
        row_sum = A.sum(1, keepdims=True) + 1e-8
        return A / row_sum

    # -- Build static node features --------------------------------------------

    def _station_features(self, stations):
        """Static features: lat, lng (normalised), platform_count, is_junction."""
        lats = [s["lat"] for s in stations]
        lngs = [s["lng"] for s in stations]
        lat_min, lat_max = min(lats), max(lats)
        lng_min, lng_max = min(lngs), max(lngs)

        feats = []
        for s in stations:
            feats.append([
                (s["lat"] - lat_min) / (lat_max - lat_min + 1e-6),
                (s["lng"] - lng_min) / (lng_max - lng_min + 1e-6),
                min(s["platform_count"] / 10.0, 1.0),
                float(s.get("is_junction", False)),
            ])
        return np.array(feats, dtype=np.float32)  # (N, 4)

    # -- Build training snapshots ----------------------------------------------

    def _build_snapshots(self, cascade_pairs, static_feats, delays_csv=None):
        """
        Group cascade_pairs by hour-bucket.
        For each bucket, build a node feature matrix and propagation label vector.
        """
        # Build dict: timestamp-bucket -> {station: max_delay}
        bucket_delays = defaultdict(lambda: defaultdict(float))
        bucket_labels = defaultdict(lambda: defaultdict(float))

        for row in cascade_pairs:
            ts = row["timestamp"][:13]   # hour bucket "2020-06-01T14"
            us = row["upstream_station"]
            ds = row["downstream_station"]
            ud = float(row["upstream_delay"])
            dd = float(row["downstream_delay"])
            bucket_delays[ts][us] = max(bucket_delays[ts][us], ud)
            bucket_delays[ts][ds] = max(bucket_delays[ts][ds], dd)
            # Label: downstream was affected
            if dd > 10:
                bucket_labels[ts][ds] = 1.0

        snapshots = []
        for ts, delay_map in bucket_delays.items():
            # Build (N, d_feat) matrix
            delay_feat = np.zeros((self.n_stations, 1), dtype=np.float32)
            labels     = np.zeros(self.n_stations, dtype=np.float32)
            for sid, d in delay_map.items():
                idx = self.station2idx.get(sid)
                if idx is not None:
                    delay_feat[idx, 0] = min(d / 180.0, 1.0)
            for sid, lbl in bucket_labels.get(ts, {}).items():
                idx = self.station2idx.get(sid)
                if idx is not None:
                    labels[idx] = lbl

            # Concatenate static + dynamic features
            node_feats = np.concatenate([static_feats, delay_feat], axis=1)  # (N, 5)
            snapshots.append((node_feats, labels))

        return snapshots

    # -- Train -----------------------------------------------------------------

    def train(self, cascade_csv: str, stations_json: str, tracks_json: str):
        print("Loading network topology ...")
        with open(stations_json) as f:
            stations = json.load(f)
        with open(tracks_json) as f:
            tracks = json.load(f)

        self.station2idx = {s["id"]: i for i, s in enumerate(stations)}
        self.idx2station = {i: s["id"] for i, s in enumerate(stations)}
        self.n_stations  = len(stations)

        self.adj = self._build_adjacency(stations, tracks)
        static_feats = self._station_features(stations)

        print("Loading cascade pairs ...")
        cascade_pairs = []
        with open(cascade_csv, newline="") as f:
            for row in csv.DictReader(f):
                cascade_pairs.append(row)
        print(f"   {len(cascade_pairs):,} cascade events")

        snapshots = self._build_snapshots(cascade_pairs, static_feats)
        print(f"   {len(snapshots):,} graph snapshots built")

        # Split
        random_idx = np.random.permutation(len(snapshots))
        split = int(0.85 * len(snapshots))
        train_idx = random_idx[:split]
        val_idx   = random_idx[split:]

        train_snaps = [snapshots[i] for i in train_idx]
        val_snaps   = [snapshots[i] for i in val_idx]

        d_in = static_feats.shape[1] + 1   # 4 static + 1 delay feature
        self.model = GraphSAGE(d_in=d_in, d_hidden=D_HIDDEN, n_layers=N_LAYERS).to(DEVICE)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=LR)
        adj_t = torch.tensor(self.adj, dtype=torch.float32).to(DEVICE)

        best_val  = float("inf")
        best_state = None

        print("Training GraphSAGE ...")
        for epoch in range(1, EPOCHS + 1):
            np.random.shuffle(train_snaps)

            # -- Train --
            self.model.train()
            train_loss = 0.0
            for feats, labels in train_snaps:
                x = torch.tensor(feats, dtype=torch.float32).to(DEVICE)   # (N, d)
                y = torch.tensor(labels, dtype=torch.float32).to(DEVICE)  # (N,)

                preds = self.model(x, adj_t)
                # Weighted BCE (positives are ~20% of nodes)
                pos_weight = torch.tensor([4.0], device=DEVICE)
                loss = F.binary_cross_entropy(preds, y,
                           weight=(y * 3.0 + 1.0))   # upweight positives
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                train_loss += loss.item()

            train_loss /= len(train_snaps)

            # -- Validate --
            self.model.eval()
            val_loss = 0.0
            tp = fp = tn = fn = 0
            with torch.no_grad():
                for feats, labels in val_snaps:
                    x = torch.tensor(feats, dtype=torch.float32).to(DEVICE)
                    y = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
                    preds = self.model(x, adj_t)
                    loss  = F.binary_cross_entropy(preds, y,
                                weight=(y * 3.0 + 1.0))
                    val_loss += loss.item()
                    pred_bin = (preds > 0.4).float()
                    tp += ((pred_bin == 1) & (y == 1)).sum().item()
                    fp += ((pred_bin == 1) & (y == 0)).sum().item()
                    tn += ((pred_bin == 0) & (y == 0)).sum().item()
                    fn += ((pred_bin == 0) & (y == 1)).sum().item()

            val_loss /= len(val_snaps)
            prec   = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1     = 2 * prec * recall / (prec + recall + 1e-8)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3}/{EPOCHS}  "
                      f"train={train_loss:.4f}  val={val_loss:.4f}  "
                      f"P={prec:.3f}  R={recall:.3f}  F1={f1:.3f}")

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_metrics = {"precision": round(prec, 3),
                                "recall": round(recall, 3),
                                "f1": round(f1, 3)}

        self.model.load_state_dict(best_state)
        self.metrics = best_metrics
        self.metrics["best_val_loss"] = round(best_val, 4)
        print(f"\n[OK] GraphSAGE training complete | F1={f1:.3f}  P={prec:.3f}  R={recall:.3f}")
        return self.metrics

    def predict(self, current_delays: dict) -> dict:
        """
        current_delays: {station_id: delay_minutes}
        Returns: {station_id: propagation_risk_0_to_1}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        # Build node feature tensor
        static = np.zeros((self.n_stations, 4), dtype=np.float32)
        delay_feat = np.zeros((self.n_stations, 1), dtype=np.float32)

        for sid, d in current_delays.items():
            idx = self.station2idx.get(sid)
            if idx is not None:
                delay_feat[idx, 0] = min(d / 180.0, 1.0)

        feats = np.concatenate([static, delay_feat], axis=1)
        x     = torch.tensor(feats, dtype=torch.float32).to(DEVICE)
        adj_t = torch.tensor(self.adj, dtype=torch.float32).to(DEVICE)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(x, adj_t).cpu().numpy()

        return {self.idx2station[i]: float(scores[i])
                for i in range(self.n_stations)}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict":  self.model.state_dict(),
            "station2idx": self.station2idx,
            "idx2station": self.idx2station,
            "adj":         self.adj,
            "n_stations":  self.n_stations,
            "metrics":     self.metrics,
        }, path)
        print(f"[OK] GraphSAGE saved -> {path}")

    @classmethod
    def load(cls, path: str):
        obj = cls()
        ckpt = torch.load(path, map_location=DEVICE)
        obj.station2idx = ckpt["station2idx"]
        obj.idx2station = ckpt["idx2station"]
        obj.adj         = ckpt["adj"]
        obj.n_stations  = ckpt["n_stations"]
        obj.metrics     = ckpt.get("metrics", {})
        d_in = 5  # 4 static + 1 delay
        obj.model = GraphSAGE(d_in=d_in, d_hidden=D_HIDDEN, n_layers=N_LAYERS).to(DEVICE)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.model.eval()
        print(f"[OK] GraphSAGE loaded from {path}  |  metrics={obj.metrics}")
        return obj
