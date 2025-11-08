"""Helper functions for routing state and reward calculations."""

import time
import numpy as np
from datetime import datetime
from typing import Any, Dict

from .types import RoutingAction, RoutingDecision


def extract_state_features(
    venues: list,
    network_simulator: Any,
    latency_predictor: Any,
    feature_extractor: Any,
    order_book_manager: Any,
    symbol: str,
) -> np.ndarray:
    """Extract current state for routing decision."""
    state = []

    market_summary = order_book_manager.get_market_summary(symbol)

    for venue in venues:
        features = extract_venue_features(symbol, venue, feature_extractor, latency_predictor)
        pred = latency_predictor.predict(venue, features)
        state.append(pred.predicted_latency_us / 10000.0)

    state.extend(
        [
            market_summary.get("avg_mid_price", 100) / 1000.0,
            np.log1p(market_summary.get("total_volume", 1000)) / 10.0,
            market_summary.get("avg_spread", 0.01) / 0.1,
            market_summary.get("volatility", 0.01) * 100.0,
            market_summary.get("order_imbalance", 0.0),
            market_summary.get("trade_intensity", 0.5),
            datetime.now().hour / 24.0,
            datetime.now().minute / 60.0,
            float(9 <= datetime.now().hour <= 16),
            0.0,
        ]
    )

    state.extend([0.5, 0.1, 0.0, 0.0, 0.5])

    for venue in venues[:5]:
        latency = network_simulator.get_current_latency(venue)
        state.append(latency / 10000.0)

    return np.array(state, dtype=np.float32)


def extract_venue_features(
    symbol: str, venue: str, feature_extractor: Any, latency_predictor: Any
) -> np.ndarray:
    """Extract features for latency prediction."""
    feature_vector = feature_extractor.extract_features(symbol, venue, time.time())
    features = np.zeros(45)

    if hasattr(feature_vector, "features"):
        feature_dict = feature_vector.features
        features[0] = datetime.now().hour / 24.0
        features[5] = feature_dict.get("network_latency_mean", 1000) / 10000.0
        features[10] = feature_dict.get("mid_price", 100)
        features[11] = np.log1p(feature_dict.get("volume", 1000))
        features[14] = feature_dict.get("volatility_1min", 0.01)

    return features


def calculate_expected_reward(
    action: RoutingAction, expected_latency: float, urgency: float
) -> float:
    """Calculate expected reward for routing decision."""
    if action == RoutingAction.HOLD:
        return -0.1 * urgency
    elif action == RoutingAction.CANCEL:
        return -0.5
    else:
        latency_score = 1.0 / (1.0 + expected_latency / 1000.0)
        return latency_score * urgency * 10.0


def calculate_actual_reward(
    decision: RoutingDecision, actual_latency: float, fill_success: bool
) -> float:
    """Calculate actual reward based on execution results."""
    if not fill_success:
        return -5.0

    base_reward = 10.0

    if actual_latency < 500:
        latency_reward = 5.0
    elif actual_latency < 1000:
        latency_reward = 2.0
    elif actual_latency < 2000:
        latency_reward = 0.5
    else:
        latency_reward = -2.0

    if decision.expected_latency_us > 0:
        prediction_error = abs(actual_latency - decision.expected_latency_us)
        accuracy_bonus = max(0, 2.0 - prediction_error / 1000.0)
    else:
        accuracy_bonus = 0.0

    return base_reward + latency_reward + accuracy_bonus
