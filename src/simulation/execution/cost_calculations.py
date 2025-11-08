"""Cost calculations for latency and market impact."""

from typing import Dict, Any
from ..latency.types import LatencyBreakdown


def calculate_latency_cost(
    order: Any, latency_breakdown: LatencyBreakdown, market_state: Dict
) -> float:
    """Calculate opportunity cost due to latency in basis points."""
    base_cost_bps = latency_breakdown.total_latency_us / 100 * 0.1

    volatility_multiplier = market_state.get("volatility", 0.02) / 0.02
    vol_adjusted_cost = base_cost_bps * volatility_multiplier

    if (
        hasattr(order, "strategy")
        and order.strategy.value == "arbitrage"
        and latency_breakdown.queue_delay_us > 100
    ):
        queue_penalty = (latency_breakdown.queue_delay_us - 100) / 100 * 0.5
        vol_adjusted_cost += queue_penalty

    if latency_breakdown.network_latency_us > 2000:
        spike_penalty = (latency_breakdown.network_latency_us - 2000) / 1000 * 0.3
        vol_adjusted_cost += spike_penalty

    return vol_adjusted_cost


def apply_permanent_impact(market_state: Dict, side: Any, impact_bps: float) -> None:
    """Apply permanent market impact to market state."""
    impact_factor = impact_bps / 10000
    if side.value == "buy":
        market_state["mid_price"] *= 1 + impact_factor
        market_state["bid_price"] *= 1 + impact_factor
        market_state["ask_price"] *= 1 + impact_factor
    else:
        market_state["mid_price"] *= 1 - impact_factor
        market_state["bid_price"] *= 1 - impact_factor
        market_state["ask_price"] *= 1 - impact_factor


def calculate_fees(
    order: Any, fill_price: float, is_maker: bool, fee_schedule: Dict
) -> tuple[float, float]:
    """Calculate fees and rebates."""
    venue_fees = fee_schedule[order.venue]
    fee_bps = venue_fees["maker_fee"] if is_maker else venue_fees["taker_fee"]

    fees = abs(fee_bps) * fill_price * order.quantity / 10000 if fee_bps > 0 else 0
    rebate = abs(fee_bps) * fill_price * order.quantity / 10000 if fee_bps < 0 else 0

    return fees, rebate
