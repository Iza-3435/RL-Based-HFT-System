"""Feature engineering for latency prediction."""

import numpy as np
from datetime import datetime
from typing import Dict

from src.core.logging_config import get_logger

logger = get_logger()


class LatencyFeatureExtractor:
    """Extract features for latency prediction."""

    def __init__(self, feature_size: int = 45):
        self.feature_size = feature_size

    def extract(
        self, tick_data: Dict, network_data: Dict, order_book_data: Dict, market_features: Dict
    ) -> np.ndarray:
        """Extract comprehensive feature vector."""
        features = []

        features.extend(self._extract_temporal(tick_data))
        features.extend(self._extract_network(network_data))
        features.extend(self._extract_market(tick_data))
        features.extend(self._extract_order_book(order_book_data))
        features.extend(self._extract_market_features(market_features, tick_data))
        features.extend(self._extract_technical(market_features, tick_data))

        while len(features) < self.feature_size:
            features.append(0.0)

        features_array = np.array(features[: self.feature_size], dtype=np.float32)
        return np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)

    def _extract_temporal(self, tick_data: Dict) -> list:
        """Extract temporal features."""
        timestamp = tick_data.get("timestamp", datetime.now().timestamp())
        dt = datetime.fromtimestamp(timestamp)

        return [
            dt.hour + dt.minute / 60.0,
            dt.weekday() / 6.0,
            dt.microsecond / 1e6,
            np.sin(2 * np.pi * dt.hour / 24),
            np.cos(2 * np.pi * dt.hour / 24),
        ]

    def _extract_network(self, network_data: Dict) -> list:
        """Extract network features."""
        return [
            min(network_data.get("latency_us", 1000), 10000) / 10000.0,
            min(network_data.get("jitter_us", 100), 1000) / 1000.0,
            min(network_data.get("packet_loss_rate", 0), 1.0),
            min(network_data.get("congestion_score", 0.5), 1.0),
            min(network_data.get("bandwidth_utilization", 0.5), 1.0),
        ]

    def _extract_market(self, tick_data: Dict) -> list:
        """Extract market microstructure features."""
        mid_price = tick_data.get("mid_price", 100)
        volume = max(tick_data.get("volume", 1000), 1)
        bid_price = tick_data.get("bid_price", mid_price - 0.01)
        ask_price = tick_data.get("ask_price", mid_price + 0.01)

        return [
            mid_price / 1000.0,
            np.log(volume) / 10.0,
            max(ask_price - bid_price, 0.001) / mid_price,
            min(tick_data.get("trade_intensity", 0.5), 10.0) / 10.0,
            min(tick_data.get("volatility", 0.01), 0.1) / 0.1,
        ]

    def _extract_order_book(self, order_book_data: Dict) -> list:
        """Extract order book features."""
        return [
            min(order_book_data.get("bid_depth", 10000), 1000000) / 100000.0,
            min(order_book_data.get("ask_depth", 10000), 1000000) / 100000.0,
            max(-1, min(1, order_book_data.get("order_imbalance", 0))),
            max(-1, min(1, order_book_data.get("book_pressure", 0))),
            min(order_book_data.get("level2_spread", 0.01), 0.1) / 0.1,
        ]

    def _extract_market_features(self, market_features: Dict, tick_data: Dict) -> list:
        """Extract additional market features."""
        return [
            max(-0.1, min(0.1, market_features.get("vwap_deviation", 0))) / 0.1,
            max(-0.05, min(0.05, market_features.get("momentum_1min", 0))) / 0.05,
            max(-0.05, min(0.05, market_features.get("momentum_5min", 0))) / 0.05,
            market_features.get("rsi", 50) / 100.0,
            max(0, min(1, market_features.get("bollinger_position", 0.5))),
        ]

    def _extract_technical(self, market_features: Dict, tick_data: Dict) -> list:
        """Extract technical indicator features."""
        mid_price = tick_data.get("mid_price", 100)

        return [
            market_features.get("ema_20", mid_price) / 1000.0,
            market_features.get("ema_50", mid_price) / 1000.0,
            max(-0.1, min(0.1, market_features.get("macd_signal", 0))) / 0.1,
            market_features.get("stochastic_k", 50) / 100.0,
            min(market_features.get("atr", 1.0), 10.0) / 10.0,
        ]
