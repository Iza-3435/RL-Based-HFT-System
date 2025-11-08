"""Production latency predictor (native implementation)."""

from typing import Dict, List
import numpy as np
from dataclasses import dataclass
import time

from src.core.logging_config import get_logger
from src.ml.training.model_trainer import LatencyModelTrainer
from src.ml.inference.inference_engine import InferenceEngine
from src.ml.online_learning.online_learner import OnlineLatencyLearner, PerformanceTracker
from src.ml.features.feature_engineering import LatencyFeatureExtractor

logger = get_logger()


@dataclass
class LatencyPredictionResult:
    """Prediction result."""

    venue: str
    predicted_latency_us: float
    confidence: float
    timestamp: float
    prediction_time_ms: float


class LatencyPredictor:
    """Production latency predictor without legacy dependencies."""

    def __init__(self, venues: List[str], feature_size: int = 45):
        self.venues = venues
        self.feature_size = feature_size

        self.trainers: Dict[str, LatencyModelTrainer] = {}
        self.inference_engines: Dict[str, InferenceEngine] = {}
        self.online_learners: Dict[str, OnlineLatencyLearner] = {}
        self.performance_trackers: Dict[str, PerformanceTracker] = {}
        self.feature_extractor = LatencyFeatureExtractor(feature_size)

        for venue in venues:
            self.trainers[venue] = LatencyModelTrainer(feature_size=feature_size)
            self.inference_engines[venue] = InferenceEngine(
                self.trainers[venue].model, self.trainers[venue].device
            )
            self.online_learners[venue] = OnlineLatencyLearner()
            self.performance_trackers[venue] = PerformanceTracker()

        logger.info(f"Latency predictor initialized", venues=len(venues))

    def predict(self, venue: str, features: np.ndarray) -> LatencyPredictionResult:
        """Predict venue latency."""
        if venue not in self.inference_engines:
            logger.warning(f"Unknown venue: {venue}")
            return self._default_prediction(venue)

        try:
            latency, confidence, pred_time = self.inference_engines[venue].predict(venue, features)

            return LatencyPredictionResult(
                venue=venue,
                predicted_latency_us=latency,
                confidence=confidence,
                timestamp=time.time(),
                prediction_time_ms=pred_time,
            )

        except Exception as e:
            logger.error(f"Prediction error for {venue}: {e}")
            return self._default_prediction(venue)

    def train(self, venue: str, features: np.ndarray, targets: np.ndarray):
        """Train model for venue."""
        if venue not in self.trainers:
            logger.warning(f"Unknown venue: {venue}")
            return

        try:
            self.trainers[venue].train(features, targets, epochs=50)
            logger.info(f"Model trained for {venue}")

        except Exception as e:
            logger.error(f"Training error for {venue}: {e}")

    def update_online(self, venue: str, features: np.ndarray, actual_latency: float):
        """Update model with actual observation."""
        if venue not in self.online_learners:
            return

        self.online_learners[venue].add_sample(venue, features, actual_latency)

        if self.online_learners[venue].should_update(venue):
            train_features, train_targets = self.online_learners[venue].get_training_data(venue)
            if train_features is not None:
                self.train(venue, train_features, train_targets)
                self.online_learners[venue].clear_buffer(venue)

    def get_metrics(self, venue: str) -> Dict:
        """Get performance metrics."""
        if venue not in self.performance_trackers:
            return {"mae": 0, "accuracy": 0, "predictions": 0}

        return self.performance_trackers[venue].get_metrics(venue)

    def _default_prediction(self, venue: str) -> LatencyPredictionResult:
        """Default prediction when error occurs."""
        return LatencyPredictionResult(
            venue=venue,
            predicted_latency_us=1000.0,
            confidence=0.1,
            timestamp=time.time(),
            prediction_time_ms=0.0,
        )
