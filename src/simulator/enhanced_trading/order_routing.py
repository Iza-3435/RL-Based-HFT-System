"""Enhanced order routing with latency optimization."""

import logging
from typing import Dict, List, Optional

from simulator.trading_simulator import Order, OrderSide, OrderType, TradingStrategyType

logger = logging.getLogger(__name__)


class OrderRoutingEnhancer:
    """Enhanced order routing with latency-aware venue selection."""

    def __init__(self, venues: list, latency_config: Dict):
        self.venues = venues
        self.latency_config = latency_config

    def enhance_orders_with_latency_routing(
        self, orders: List[Order], ml_predictions: Dict, market_state: Dict
    ) -> List[Order]:
        """Enhance orders with latency-aware routing decisions."""
        enhanced_orders = []

        for order in orders:
            venue_latencies = {}
            for venue in self.venues:
                routing_key = f"routing_{order.symbol}_{venue}"
                if routing_key in ml_predictions:
                    pred = ml_predictions[routing_key]
                    venue_latencies[venue] = {
                        "predicted_latency_us": pred["predicted_latency_us"],
                        "confidence": pred["confidence"],
                        "congestion_prob": pred["congestion_probability"],
                    }

            optimal_venue = self._select_optimal_venue(
                order, venue_latencies, market_state
            )

            order.venue = optimal_venue
            if optimal_venue in venue_latencies:
                order.predicted_latency_us = venue_latencies[optimal_venue][
                    "predicted_latency_us"
                ]
                order.routing_confidence = venue_latencies[optimal_venue]["confidence"]

            order.latency_tolerance_us = self._calculate_latency_tolerance(order)
            order.urgency_score = self._calculate_order_urgency(order, market_state)

            enhanced_orders.append(order)

        return enhanced_orders

    def _select_optimal_venue(
        self, order: Order, venue_latencies: Dict, market_state: Dict
    ) -> str:
        """Select optimal venue based on strategy-specific latency requirements."""
        if not venue_latencies:
            return order.venue

        strategy_requirements = self.latency_config["latency_thresholds"].get(
            order.strategy.value, {"acceptable_us": 2000, "optimal_us": 800}
        )

        venue_scores = {}

        for venue, latency_info in venue_latencies.items():
            predicted_latency = latency_info["predicted_latency_us"]
            confidence = latency_info["confidence"]
            congestion_prob = latency_info["congestion_prob"]

            latency_score = max(
                0, 1.0 - (predicted_latency / strategy_requirements["acceptable_us"])
            )

            confidence_bonus = confidence * 0.2

            congestion_penalty = congestion_prob * 0.3

            if order.strategy == TradingStrategyType.ARBITRAGE:
                if predicted_latency > 500:
                    latency_score *= 0.5
            elif order.strategy == TradingStrategyType.MARKET_MAKING:
                latency_score *= confidence

            venue_scores[venue] = (
                latency_score + confidence_bonus - congestion_penalty
            )

        best_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
        return best_venue

    def _calculate_latency_tolerance(self, order: Order) -> float:
        """Calculate acceptable latency tolerance for order."""
        base_tolerance = self.latency_config["latency_thresholds"].get(
            order.strategy.value, {"acceptable_us": 2000}
        )["acceptable_us"]

        if order.order_type == OrderType.MARKET:
            return base_tolerance * 0.7
        elif order.order_type == OrderType.IOC:
            return base_tolerance * 0.5
        else:
            return base_tolerance

    def _calculate_order_urgency(self, order: Order, market_state: Dict) -> float:
        """Calculate urgency score for order (0-1 scale)."""
        urgency = 0.5

        if order.strategy == TradingStrategyType.ARBITRAGE:
            urgency = 0.9
        elif order.strategy == TradingStrategyType.MARKET_MAKING:
            urgency = 0.3
        elif order.strategy == TradingStrategyType.MOMENTUM:
            urgency = 0.7

        symbol_venue_key = f"{order.symbol}_{order.venue}"
        if symbol_venue_key in market_state:
            volatility = market_state[symbol_venue_key].get("volatility", 0.02)
            if volatility > 0.04:
                urgency = min(1.0, urgency * 1.3)

        return urgency

    def find_alternative_venue(
        self, order: Order, market_state: Dict, execution_engine
    ) -> Optional[str]:
        """Find alternative venue during congestion."""
        if hasattr(execution_engine, "get_venue_latency_rankings"):
            rankings = execution_engine.get_venue_latency_rankings()

            for venue, latency in rankings:
                if venue != order.venue and latency < 2000:
                    state_key = f"{order.symbol}_{venue}"
                    if state_key in market_state:
                        return venue

        return None
