"""NEXUS — LSTM autoencoder anomaly detector with sliding-window pipeline."""

from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LSTMAnomalyDetector(nn.Module):
    """LSTM autoencoder for multivariate sensor anomaly detection.

    Architecture
    ------------
    Encoder: LSTM(input_dim → hidden_dim, num_layers)
    Decoder: LSTM(hidden_dim → hidden_dim, num_layers)
    Output:  Linear(hidden_dim → input_dim)

    Parameters
    ----------
    input_dim : int
        Number of sensor channels (default 5).
    hidden_dim : int
        LSTM hidden state size (default 64).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    threshold : float
        MSE threshold above which a sequence is anomalous.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        threshold: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.threshold = threshold

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode–decode a batch of sequences.

        Parameters
        ----------
        x : Tensor
            Shape (batch, seq_len, input_dim).

        Returns
        -------
        Tensor
            Reconstructed sequences, same shape as input.
        """
        # Encode
        encoded, (h_n, c_n) = self.encoder(x)

        # Use encoder output as decoder input
        decoded, _ = self.decoder(encoded, (h_n, c_n))

        # Project back to input space
        reconstructed = self.output_layer(decoded)
        return reconstructed

    def compute_anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """Compute per-sample MSE reconstruction error.

        Parameters
        ----------
        x : Tensor
            Shape (batch, seq_len, input_dim).

        Returns
        -------
        np.ndarray
            Shape (batch,) — MSE score per sequence.
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Per-sample mean squared error
            mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return mse.cpu().numpy()

    def is_anomalous(self, x: torch.Tensor) -> np.ndarray:
        """Check if sequences exceed the anomaly threshold.

        Returns
        -------
        np.ndarray
            Boolean array, shape (batch,).
        """
        scores = self.compute_anomaly_score(x)
        return scores > self.threshold


class AnomalyDetectionPipeline:
    """Maintains per-location sliding windows and scores via LSTM.

    Accumulates sensor vectors per location. Once a window is full
    (``window_size`` readings), ``score_location()`` returns an anomaly
    assessment.

    Parameters
    ----------
    model : LSTMAnomalyDetector
        A trained LSTM autoencoder.
    window_size : int
        Sliding window length (default 30).
    """

    def __init__(self, model: LSTMAnomalyDetector, window_size: int = 30):
        self.model = model
        self.window_size = window_size

        # location_id → deque of sensor vectors (each a list of 5 floats)
        self._windows: dict[str, deque] = {}

    def _ensure_window(self, location_id: str):
        if location_id not in self._windows:
            self._windows[location_id] = deque(maxlen=self.window_size)

    def add_reading(self, location_id: str, sensor_vector: list[float]):
        """Append a sensor vector [vibration, temperature, brake_pressure,
        wheel_impact, track_stress] for the given location.

        Parameters
        ----------
        location_id : str
            Track or train identifier.
        sensor_vector : list[float]
            Length-5 vector of sensor values.
        """
        self._ensure_window(location_id)
        self._windows[location_id].append(sensor_vector)

    def score_location(self, location_id: str) -> Optional[dict]:
        """Score the current window for a location.

        Returns
        -------
        dict or None
            If the window is full: {location_id, anomaly_score,
            is_anomalous, confidence}. Returns None if window not
            full yet.
        """
        self._ensure_window(location_id)
        window = self._windows[location_id]

        if len(window) < self.window_size:
            return None

        # Build tensor: (1, window_size, input_dim)
        data = np.array(list(window), dtype=np.float32)
        x = torch.tensor(data).unsqueeze(0)

        score = float(self.model.compute_anomaly_score(x)[0])
        is_anom = score > self.model.threshold

        # Confidence: how far above/below threshold (sigmoid-mapped)
        ratio = score / max(self.model.threshold, 1e-9)
        confidence = min(1.0, ratio) if is_anom else max(0.0, 1.0 - ratio)

        return {
            "location_id": location_id,
            "anomaly_score": round(score, 6),
            "is_anomalous": bool(is_anom),
            "confidence": round(confidence, 4),
        }

    def get_window_fill(self, location_id: str) -> int:
        """Return how many readings are buffered for a location."""
        self._ensure_window(location_id)
        return len(self._windows[location_id])

    def reset(self, location_id: str | None = None):
        """Clear one or all sliding windows."""
        if location_id is None:
            self._windows.clear()
        else:
            self._windows.pop(location_id, None)
