"""Execution analytics and quality validation."""

import logging
import time
from typing import Dict, List, Optional

from simulator.trading_simulator import Fill, Order

logger = logging.getLogger(__name__)


class ExecutionAnalytics:
    """Comprehensive execution analytics and quality validation."""

    def __init__(self, execution_engine, strategies: Dict):
        self.execution_engine = execution_engine
        self.strategies = strategies
        self.fill_history = []

    async def execute_orders_with_analytics(
        self, orders: List[Order], market_state: Dict, current_prices: Dict
    ) -> List[Dict]:
        """Execute orders with comprehensive latency analytics."""
        execution_results = []

        for order in orders:
            start_time = time.perf_counter()

            fill = await self._execute_order_enhanced(order, market_state)

            execution_time = (time.perf_counter() - start_time) * 1e6

            if fill:
                strategy = self.strategies[order.strategy.value]
                strategy.update_positions(fill, current_prices)

                result = {
                    "order": order,
                    "fill": fill,
                    "execution_time_us": execution_time,
                    "latency_breakdown": getattr(fill, "latency_breakdown", None),
                    "prediction_accuracy": self._calculate_prediction_accuracy(
                        order, fill
                    ),
                    "venue_performance_rank": self._get_venue_rank(order.venue),
                }

                execution_results.append(result)
                self.fill_history.append(fill)

        return execution_results

    async def _execute_order_enhanced(
        self, order: Order, market_state: Dict
    ) -> Optional[Fill]:
        """Execute single order with enhanced error handling and analytics."""
        state_key = f"{order.symbol}_{order.venue}"
        if state_key not in market_state:
            logger.warning(f"No market data for {state_key}")
            return None

        venue_market_state = market_state[state_key]

        if hasattr(self.execution_engine, "latency_simulator"):
            congestion = self.execution_engine.latency_simulator.get_congestion_analysis()
            if congestion["current_congestion_level"] == "critical":
                if getattr(order, "urgency_score", 0.5) > 0.8:
                    from .order_routing import OrderRoutingEnhancer

                    router = OrderRoutingEnhancer([], {})
                    alternative_venue = router.find_alternative_venue(
                        order, market_state, self.execution_engine
                    )
                    if alternative_venue:
                        order.venue = alternative_venue
                        state_key = f"{order.symbol}_{alternative_venue}"
                        venue_market_state = market_state.get(
                            state_key, venue_market_state
                        )

        try:
            fill = await self.execution_engine.execute_order(
                order, venue_market_state, getattr(order, "predicted_latency_us", None)
            )

            if fill and hasattr(fill, "latency_breakdown"):
                self._validate_execution_quality(order, fill)

            return fill

        except Exception as e:
            logger.error(f"Order execution failed for {order.order_id}: {e}")
            return None

    def _validate_execution_quality(self, order: Order, fill: Fill) -> None:
        """Validate execution met quality expectations."""
        if hasattr(fill, "latency_breakdown"):
            actual_latency = fill.latency_breakdown.total_latency_us
            tolerance = getattr(order, "latency_tolerance_us", 2000)

            if actual_latency > tolerance:
                logger.warning(
                    f"Order {order.order_id} exceeded latency tolerance: "
                    f"{actual_latency:.0f}μs > {tolerance:.0f}μs"
                )

            if hasattr(order, "predicted_latency_us") and order.predicted_latency_us:
                error_pct = (
                    abs(actual_latency - order.predicted_latency_us)
                    / order.predicted_latency_us
                    * 100
                )
                if error_pct > 25:
                    logger.warning(
                        f"Poor latency prediction for {order.order_id}: "
                        f"{error_pct:.1f}% error"
                    )

    def _calculate_prediction_accuracy(
        self, order: Order, fill: Fill
    ) -> Optional[float]:
        """Calculate prediction accuracy for executed order."""
        if not hasattr(order, "predicted_latency_us") or not order.predicted_latency_us:
            return None

        actual_latency = getattr(fill, "latency_us", 0)
        if actual_latency == 0:
            return None

        error_pct = (
            abs(actual_latency - order.predicted_latency_us)
            / order.predicted_latency_us
            * 100
        )
        accuracy = max(0, 100 - error_pct)

        return accuracy

    def _get_venue_rank(self, venue: str) -> int:
        """Get current performance rank of venue (1 = best)."""
        if hasattr(self.execution_engine, "get_venue_latency_rankings"):
            rankings = self.execution_engine.get_venue_latency_rankings()
            for rank, (v, _) in enumerate(rankings, 1):
                if v == venue:
                    return rank
        return len(self.execution_engine.fee_schedule)
