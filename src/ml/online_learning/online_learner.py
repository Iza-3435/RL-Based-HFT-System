"""Online learning for continuous model improvement."""

import numpy as np
from typing import Dict, List
from collections import deque

from src.core.logging_config import get_logger

logger = get_logger()


class OnlineLatencyLearner:
    """Online learning for latency models."""

    def __init__(self, update_threshold: int = 100):
        self.update_threshold = update_threshold
        self.sample_buffers: Dict[str, List] = {}

        logger.verbose("Online learner initialized", threshold=update_threshold)

    def add_sample(self, venue: str, features: np.ndarray, actual_latency: float):
        """Add training sample from actual execution."""
        if venue not in self.sample_buffers:
            self.sample_buffers[venue] = []

        self.sample_buffers[venue].append((features, actual_latency))

        if len(self.sample_buffers[venue]) > self.update_threshold * 2:
            self.sample_buffers[venue] = self.sample_buffers[venue][-self.update_threshold :]

    def should_update(self, venue: str) -> bool:
        """Check if model should be updated."""
        return (
            venue in self.sample_buffers
            and len(self.sample_buffers[venue]) >= self.update_threshold
        )

    def get_training_data(self, venue: str) -> tuple:
        """Get accumulated training data."""
        if venue not in self.sample_buffers:
            return None, None

        data = self.sample_buffers[venue]
        features = np.array([x[0] for x in data])
        targets = np.array([x[1] for x in data])

        return features, targets

    def clear_buffer(self, venue: str):
        """Clear training buffer after update."""
        if venue in self.sample_buffers:
            self.sample_buffers[venue] = []


class PerformanceTracker:
    """Track prediction performance."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.predictions: Dict[str, deque] = {}
        self.errors: Dict[str, deque] = {}

    def record_prediction(self, venue: str, predicted: float, actual: float):
        """Record prediction vs actual."""
        if venue not in self.predictions:
            self.predictions[venue] = deque(maxlen=self.history_size)
            self.errors[venue] = deque(maxlen=self.history_size)

        error = abs(predicted - actual)
        self.predictions[venue].append((predicted, actual))
        self.errors[venue].append(error)

    def get_metrics(self, venue: str) -> Dict:
        """Get performance metrics for venue."""
        if venue not in self.errors or len(self.errors[venue]) == 0:
            return {"mae": 0, "accuracy": 0, "predictions": 0}

        errors = list(self.errors[venue])
        predictions = list(self.predictions[venue])

        mae = np.mean(errors)
        within_10pct = sum(1 for pred, actual in predictions if abs(pred - actual) / actual < 0.1)
        accuracy = within_10pct / len(predictions) if predictions else 0

        return {"mae": mae, "accuracy": accuracy * 100, "predictions": len(predictions)}
