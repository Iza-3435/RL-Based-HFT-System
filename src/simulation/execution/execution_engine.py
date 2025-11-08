"""Enhanced order execution engine with realistic latency simulation."""

import logging
from typing import Dict, Optional, Any
from collections import defaultdict

from src.simulation.latency import LatencySimulator
from simulator.trading_simulator import MarketImpactModel, Fill, OrderSide
from .fees import DEFAULT_FEE_SCHEDULE
from .execution_stats import ExecutionStatistics
from . import price_calculations as price_calc
from . import cost_calculations as cost_calc

logger = logging.getLogger(__name__)


class EnhancedOrderExecutionEngine:
    """Enhanced execution engine with comprehensive latency simulation."""

    def __init__(self, venues: list[str], fee_schedule: Optional[Dict[str, Dict]] = None) -> None:
        self.market_impact_model = MarketImpactModel()
        self.execution_queue = []
        self.order_book = defaultdict(lambda: {"bids": [], "asks": []})

        self.latency_simulator = LatencySimulator(venues)
        self.fee_schedule = fee_schedule or DEFAULT_FEE_SCHEDULE
        self.stats = ExecutionStatistics()

        logger.info("EnhancedOrderExecutionEngine initialized")

    async def execute_order(
        self,
        order: Any,
        market_state: Dict,
        predicted_latency_us: Optional[float] = None,
    ) -> Optional[Fill]:
        """Execute order with comprehensive latency simulation."""
        self.latency_simulator.update_market_conditions(
            symbol=order.symbol,
            volatility=market_state.get("volatility", 0.02),
            volume_factor=market_state.get("volume", 1.0) / 1000000,
            timestamp=order.timestamp,
        )

        latency_breakdown = self.latency_simulator.simulate_latency(
            venue=order.venue,
            symbol=order.symbol,
            order_type=(
                order.order_type.value
                if hasattr(order.order_type, "value")
                else str(order.order_type)
            ),
            predicted_latency_us=predicted_latency_us
            or getattr(order, "predicted_latency_us", None),
            timestamp=order.timestamp,
        )

        actual_latency_us = latency_breakdown.total_latency_us
        arrival_price = market_state["mid_price"]

        price_drift = price_calc.simulate_price_drift(
            arrival_price, actual_latency_us, market_state, latency_breakdown
        )
        execution_price = arrival_price + price_drift

        permanent_impact, temporary_impact = self.market_impact_model.calculate_impact(
            order, market_state, actual_latency_us
        )

        latency_cost_bps = cost_calc.calculate_latency_cost(order, latency_breakdown, market_state)

        fill_price = price_calc.calculate_fill_price(
            order, market_state, temporary_impact, latency_cost_bps
        )

        is_maker = price_calc.determine_maker_status(order, market_state, actual_latency_us)
        fees, rebate = cost_calc.calculate_fees(order, fill_price, is_maker, self.fee_schedule)

        slippage_bps = price_calc.calculate_slippage(
            order, fill_price, arrival_price, latency_breakdown
        )

        fill = Fill(
            fill_id=f"F{self.stats.fill_count:08d}",
            order_id=order.order_id,
            symbol=order.symbol,
            venue=order.venue,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=order.timestamp + actual_latency_us / 1e6,
            fees=fees,
            rebate=rebate,
            latency_us=actual_latency_us,
            slippage_bps=slippage_bps,
            market_impact_bps=temporary_impact,
        )

        fill.latency_breakdown = latency_breakdown
        fill.latency_cost_bps = latency_cost_bps
        fill.prediction_error_us = latency_breakdown.prediction_error_us

        order.status = "FILLED"
        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.fill_timestamp = fill.timestamp
        order.latency_us = actual_latency_us
        order.latency_breakdown = latency_breakdown

        self.stats.update(fill, latency_breakdown)
        cost_calc.apply_permanent_impact(market_state, order.side, permanent_impact)

        return fill

    def get_enhanced_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        return self.stats.get_execution_stats(self.latency_simulator)

    def get_venue_latency_rankings(self) -> list[tuple[str, float]]:
        """Get venues ranked by average latency performance."""
        return self.stats.get_venue_rankings(self.latency_simulator)

    def get_latency_cost_analysis(self) -> Dict[str, Any]:
        """Analyze costs specifically attributed to latency."""
        return self.stats.get_cost_analysis()
