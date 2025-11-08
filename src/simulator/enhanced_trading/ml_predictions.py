"""Enhanced ML predictions with latency forecasting."""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class MLPredictionEnhancer:
    """Enhanced ML predictions including latency forecasts."""

    def __init__(self, venues: list, execution_engine):
        self.venues = venues
        self.execution_engine = execution_engine

    async def get_enhanced_predictions(
        self, ml_predictor, tick, market_state: Dict
    ) -> Dict:
        """Get enhanced ML predictions including latency forecasts."""
        predictions = {}

        for venue in self.venues:
            routing_key = f"routing_{tick.symbol}_{venue}"

            routing_decision = ml_predictor.make_routing_decision(tick.symbol)

            if hasattr(self.execution_engine, "latency_simulator"):
                venue_stats = self.execution_engine.latency_simulator.get_venue_latency_stats(
                    venue, 5
                )
                current_latency_us = (
                    venue_stats.get("mean_us", 1000) if venue_stats else 1000
                )
            else:
                current_latency_us = 1000

            predictions[routing_key] = {
                "venue": routing_decision.venue,
                "predicted_latency_us": routing_decision.expected_latency_us,
                "confidence": routing_decision.confidence,
                "current_latency_us": current_latency_us,
                "latency_trend": self._calculate_latency_trend(venue),
                "congestion_probability": self._estimate_congestion_probability(
                    venue, market_state
                ),
            }

        regime_detection = ml_predictor.detect_market_regime(market_state)
        predictions["regime"] = regime_detection.regime.value
        predictions["regime_confidence"] = getattr(regime_detection, "confidence", 0.7)

        predictions["volatility_forecast"] = (
            market_state.get(f"{tick.symbol}_{tick.venue}", {}).get("volatility", 0.01)
        )

        base_momentum = np.random.randn() * 0.5
        latency_adjustment = self._calculate_latency_momentum_adjustment(tick.symbol)
        predictions[f"momentum_signal_{tick.symbol}"] = (
            base_momentum * latency_adjustment
        )

        predictions["market_impact_forecast"] = self._forecast_market_impact(
            tick, market_state
        )

        return predictions

    def _calculate_latency_trend(self, venue: str) -> str:
        """Calculate latency trend for venue."""
        if hasattr(self.execution_engine, "latency_simulator"):
            recent_stats = self.execution_engine.latency_simulator.get_venue_latency_stats(
                venue, 5
            )
            historical_stats = self.execution_engine.latency_simulator.get_venue_latency_stats(
                venue, 10
            )

            if recent_stats and historical_stats:
                recent_mean = recent_stats.get("mean_us", 1000)
                historical_mean = historical_stats.get("mean_us", 1000)

                if recent_mean > historical_mean * 1.1:
                    return "increasing"
                elif recent_mean < historical_mean * 0.9:
                    return "decreasing"
                else:
                    return "stable"

        return "unknown"

    def _estimate_congestion_probability(self, venue: str, market_state: Dict) -> float:
        """Estimate probability of congestion at venue."""
        base_probability = 0.1

        avg_volatility = np.mean(
            [
                state.get("volatility", 0.02)
                for key, state in market_state.items()
                if key.endswith(f"_{venue}")
            ]
        )
        volatility_factor = min(avg_volatility / 0.02, 3.0)

        avg_volume = np.mean(
            [
                state.get("volume", 1000)
                for key, state in market_state.items()
                if key.endswith(f"_{venue}")
            ]
        )
        volume_factor = min(avg_volume / 1000, 2.0)

        congestion_prob = base_probability * volatility_factor * volume_factor
        return min(congestion_prob, 0.8)

    def _calculate_latency_momentum_adjustment(self, symbol: str) -> float:
        """Calculate momentum signal adjustment based on latency expectations."""
        base_adjustment = 1.0

        if hasattr(self.execution_engine, "latency_simulator"):
            venue_latencies = []
            for venue in self.venues:
                stats = self.execution_engine.latency_simulator.get_venue_latency_stats(
                    venue, 5
                )
                if stats:
                    venue_latencies.append(stats["mean_us"])

            if venue_latencies:
                avg_latency = np.mean(venue_latencies)
                adjustment = max(0.7, 1.0 - (avg_latency - 1000) / 5000)
                return adjustment

        return base_adjustment

    def _forecast_market_impact(self, tick, market_state: Dict) -> float:
        """Forecast market impact considering latency."""
        base_impact = 0.1

        volatility = getattr(tick, "volatility", 0.02)
        volatility_multiplier = volatility / 0.02

        symbol_venue_key = f"{tick.symbol}_{tick.venue}"
        if symbol_venue_key in market_state:
            network_latency = market_state[symbol_venue_key].get(
                "current_network_latency_us", 1000
            )
            latency_multiplier = 1.0 + (network_latency - 500) / 2000
        else:
            latency_multiplier = 1.0

        return base_impact * volatility_multiplier * latency_multiplier
