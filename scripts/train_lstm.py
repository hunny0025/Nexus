"""NEXUS — Train the LSTM autoencoder on synthetic sensor data.

Generates normal and fault sequences, trains for 50 epochs, calibrates
the anomaly threshold, and saves weights to data/models/lstm_weights.pt.
Runs in under 5 minutes on CPU.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.detection.lstm_detector import LSTMAnomalyDetector

# ---------------------------------------------------------------------------
# Sensor distribution parameters (mirrored from simulator)
# ---------------------------------------------------------------------------

NORMAL_PARAMS = {
    # (mean, std) — order: vibration, temperature, brake_pressure,
    #                       wheel_impact, track_stress
    "vibration": (0.5, 0.08),
    "temperature": (45.0, 3.0),
    "brake_pressure": (6.0, 0.4),
    "wheel_impact": (1.2, 0.15),
    "track_stress": (100.0, 12.0),
}

FAULT_MULTIPLIERS = {
    "vibration": 4.0,
    "temperature": 2.0,
    "brake_pressure": 0.3,
    "wheel_impact": 5.0,
    "track_stress": 2.5,
}

SENSOR_ORDER = ["vibration", "temperature", "brake_pressure", "wheel_impact", "track_stress"]
SEQ_LEN = 30
INPUT_DIM = 5


def generate_normal_sequence() -> np.ndarray:
    """Generate one normal sequence: shape (SEQ_LEN, INPUT_DIM)."""
    seq = np.zeros((SEQ_LEN, INPUT_DIM), dtype=np.float32)
    for i, name in enumerate(SENSOR_ORDER):
        mean, std = NORMAL_PARAMS[name]
        seq[:, i] = np.random.normal(mean, std, size=SEQ_LEN)
    return seq


def generate_fault_sequence() -> np.ndarray:
    """Generate one fault sequence with gradual severity ramp."""
    seq = np.zeros((SEQ_LEN, INPUT_DIM), dtype=np.float32)
    severity_ramp = np.linspace(0.3, 1.0, SEQ_LEN)
    for i, name in enumerate(SENSOR_ORDER):
        mean, std = NORMAL_PARAMS[name]
        base = np.random.normal(mean, std, size=SEQ_LEN)
        mult = FAULT_MULTIPLIERS[name]
        if mult < 1.0:
            shift = mean * (1.0 - mult) * severity_ramp * -1
        else:
            shift = mean * (mult - 1.0) * severity_ramp
        seq[:, i] = base + shift
    return seq


def main():
    model_dir = Path(__file__).resolve().parent.parent / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "lstm_weights.pt"

    print("=" * 60)
    print("NEXUS — LSTM Autoencoder Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    print("\n1. Generating synthetic training data …")
    n_normal = 5000
    n_fault = 500

    normal_seqs = np.stack([generate_normal_sequence() for _ in range(n_normal)])
    fault_seqs = np.stack([generate_fault_sequence() for _ in range(n_fault)])

    # Training set: normal only (autoencoder learns normal distribution)
    X_train = torch.tensor(normal_seqs, dtype=torch.float32)

    # Validation: mix of normal and fault for threshold calibration
    X_val_normal = torch.tensor(normal_seqs[:500], dtype=torch.float32)
    X_val_fault = torch.tensor(fault_seqs, dtype=torch.float32)

    print(f"   Normal sequences : {n_normal}")
    print(f"   Fault sequences  : {n_fault}")
    print(f"   Sequence length  : {SEQ_LEN}")
    print(f"   Input dimensions : {INPUT_DIM}")

    # ------------------------------------------------------------------
    # 2. Create model & dataloader
    # ------------------------------------------------------------------
    model = LSTMAnomalyDetector(
        input_dim=INPUT_DIM, hidden_dim=64, num_layers=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_train, X_train)  # autoencoder: target = input
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print("\n2. Training for 50 epochs …")
    epochs = 50
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(X_train)
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"   Epoch {epoch:3d}/50  loss={epoch_loss:.6f}  ({elapsed:.1f}s)")

    train_time = time.time() - t0
    print(f"\n   Training completed in {train_time:.1f}s")

    # ------------------------------------------------------------------
    # 4. Calibrate threshold
    # ------------------------------------------------------------------
    print("\n3. Calibrating anomaly threshold …")
    model.eval()
    normal_scores = model.compute_anomaly_score(X_val_normal)
    fault_scores = model.compute_anomaly_score(X_val_fault)

    mean_normal = float(np.mean(normal_scores))
    std_normal = float(np.std(normal_scores))
    threshold = mean_normal + 3.0 * std_normal
    model.threshold = threshold

    print(f"   Normal scores — mean: {mean_normal:.6f}, std: {std_normal:.6f}")
    print(f"   Threshold (μ + 3σ)  : {threshold:.6f}")
    print(f"\n   Example normal scores : {normal_scores[:5]}")
    print(f"   Example fault scores  : {fault_scores[:5]}")

    # Detection stats
    normal_detected = np.sum(normal_scores > threshold)
    fault_detected = np.sum(fault_scores > threshold)
    print(f"\n   False positives (normal flagged)  : {normal_detected}/{len(normal_scores)}")
    print(f"   True positives  (fault  flagged)  : {fault_detected}/{len(fault_scores)}")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "threshold": model.threshold,
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
        },
        weights_path,
    )
    print(f"\n✅ Model saved to {weights_path}")
    print(f"   Threshold: {model.threshold:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
