"""Integrated ML predictor for backtesting."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class IntegratedMLPredictor:
    """Integrated ML predictor for backtesting"""

    def __init__(self, latency_predictor, ensemble_model, routing_environment, regime_detector):
        self.latency_predictor = latency_predictor
        self.ensemble_model = ensemble_model
        self.routing_environment = routing_environment
        self.regime_detector = regime_detector

    def make_routing_decision(self, symbol):
        """Make routing decision for backtesting"""
        return self.routing_environment.make_routing_decision(symbol, urgency=0.5)

    def detect_market_regime(self, market_state):
        """Detect market regime for backtesting"""

        # Simplified regime detection
        class SimpleRegime:
            def __init__(self, regime):
                self.regime = regime
                self.value = regime

        volatility = market_state.get("volatility", 0.01)
        if volatility > 0.03:
            return SimpleRegime("volatile")
        elif volatility < 0.005:
            return SimpleRegime("quiet")
        else:
            return SimpleRegime("normal")
