"""Production latency predictor with legacy compatibility."""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from src.core.logging_config import get_logger
from data.latency_predictor import LatencyPredictor as LegacyLatencyPredictor

logger = get_logger()


@dataclass
class LatencyPrediction:
    """Structured latency prediction output."""

    venue: str
    predicted_latency_us: float
    confidence: float
    timestamp: float
    prediction_time_ms: float


class ProductionLatencyPredictor:
    """Production wrapper for latency prediction with clean interface."""

    def __init__(self, venues: List[str], feature_size: int = 45):
        self.venues = venues
        self.feature_size = feature_size

        self._predictor = LegacyLatencyPredictor(venues=venues, feature_size=feature_size)

        logger.verbose(
            "Latency predictor initialized", venues=len(venues), feature_size=feature_size
        )

    def predict(self, venue: str, features: np.ndarray) -> LatencyPrediction:
        """Predict venue latency from features."""
        try:
            result = self._predictor.predict(venue, features)

            return LatencyPrediction(
                venue=result.venue,
                predicted_latency_us=result.predicted_latency_us,
                confidence=result.confidence,
                timestamp=result.timestamp,
                prediction_time_ms=result.prediction_time_ms,
            )

        except Exception as e:
            logger.error(f"Prediction error for {venue}: {e}")

            return LatencyPrediction(
                venue=venue,
                predicted_latency_us=1000.0,
                confidence=0.1,
                timestamp=0.0,
                prediction_time_ms=0.0,
            )

    def train(self, venue: str, features: np.ndarray, targets: np.ndarray):
        """Train model for venue."""
        try:
            self._predictor.train_model(venue, features, targets)
            logger.verbose(f"Model trained for {venue}")

        except Exception as e:
            logger.error(f"Training error for {venue}: {e}")

    def update_online(self, venue: str, features: np.ndarray, actual_latency: float):
        """Update model with new observation."""
        try:
            self._predictor.update_online(venue, features, actual_latency)

        except Exception as e:
            logger.debug(f"Online update error for {venue}: {e}")

    def get_metrics(self, venue: str) -> Dict:
        """Get performance metrics for venue."""
        try:
            return self._predictor.get_model_performance(venue)
        except:
            return {"mae": 0, "accuracy": 0, "predictions": 0}

    def set_fast_mode(self, enabled: bool = True):
        """Configure for fast mode."""
        if hasattr(self._predictor, "set_fast_mode"):
            self._predictor.set_fast_mode(enabled)
