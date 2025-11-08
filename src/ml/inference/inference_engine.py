"""Real-time inference engine for latency prediction."""

import torch
import numpy as np
from typing import Dict, Optional
from collections import deque
import time

from src.core.logging_config import get_logger

logger = get_logger()


class InferenceEngine:
    """Fast inference engine for real-time predictions."""

    def __init__(self, model: torch.nn.Module, device: torch.device, sequence_length: int = 50):
        self.model = model
        self.device = device
        self.sequence_length = sequence_length

        self.model.eval()
        self.model.to(device)

        self.feature_buffers: Dict[str, deque] = {}

        logger.verbose("Inference engine initialized", sequence_length=sequence_length)

    def predict(self, venue: str, features: np.ndarray) -> tuple[float, float, float]:
        """Make real-time prediction.

        Returns:
            (predicted_latency_us, confidence, inference_time_ms)
        """
        start_time = time.time()

        try:
            if venue not in self.feature_buffers:
                self.feature_buffers[venue] = deque(maxlen=self.sequence_length)

            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            self.feature_buffers[venue].append(features)

            if len(self.feature_buffers[venue]) < self.sequence_length:
                return 1000.0, 0.1, 0.0

            sequence = np.array(list(self.feature_buffers[venue]))
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

            with torch.no_grad():
                prediction, confidence = self.model(sequence_tensor)

            predicted_latency = float(prediction.cpu().numpy()[0][0])
            confidence_score = float(confidence.cpu().numpy()[0][0])

            predicted_latency = max(50.0, min(50000.0, predicted_latency))

            inference_time_ms = (time.time() - start_time) * 1000

            return predicted_latency, confidence_score, inference_time_ms

        except Exception as e:
            logger.debug(f"Inference error for {venue}: {e}")
            return 1000.0, 0.1, 0.0

    def clear_buffer(self, venue: str):
        """Clear feature buffer for venue."""
        if venue in self.feature_buffers:
            self.feature_buffers[venue].clear()

    def reset(self):
        """Reset all buffers."""
        self.feature_buffers.clear()
