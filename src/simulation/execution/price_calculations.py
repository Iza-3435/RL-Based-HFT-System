"""Price calculations for order execution."""

import numpy as np
from typing import Dict, Any
from ..latency.types import LatencyBreakdown


def simulate_price_drift(
    price: float,
    latency_us: float,
    market_state: Dict,
    latency_breakdown: LatencyBreakdown,
) -> float:
    """Enhanced price movement simulation considering latency components."""
    latency_days = latency_us / (1e6 * 60 * 60 * 6.5)
    volatility = market_state.get("volatility", 0.02)

    network_drift = (
        (latency_breakdown.network_latency_us / 1000) * volatility * np.random.randn() * 0.1
    )
    queue_drift = (latency_breakdown.queue_delay_us / 1000) * volatility * np.random.randn() * 0.15
    exchange_drift = 0

    total_drift = network_drift + queue_drift + exchange_drift

    if hasattr(market_state, "recent_price_direction"):
        momentum_factor = market_state["recent_price_direction"] * 0.05
        total_drift += momentum_factor * (latency_us / 1000)

    return price * total_drift


def calculate_fill_price(
    order: Any, market_state: Dict, temporary_impact: float, latency_cost_bps: float
) -> float:
    """Calculate final fill price with all effects."""
    if order.order_type.value == "market":
        if order.side.value == "buy":
            base_price = market_state["ask_price"]
            price_impact = base_price * (temporary_impact + latency_cost_bps) / 10000
            fill_price = base_price + price_impact
        else:
            base_price = market_state["bid_price"]
            price_impact = base_price * (temporary_impact + latency_cost_bps) / 10000
            fill_price = base_price - price_impact

    elif order.order_type.value == "limit":
        if order.side.value == "buy":
            if order.price >= market_state["ask_price"]:
                base_price = market_state["ask_price"]
                price_impact = base_price * (temporary_impact + latency_cost_bps) / 10000
                fill_price = base_price + price_impact
            else:
                fill_price = order.price
        else:
            if order.price <= market_state["bid_price"]:
                base_price = market_state["bid_price"]
                price_impact = base_price * (temporary_impact + latency_cost_bps) / 10000
                fill_price = base_price - price_impact
            else:
                fill_price = order.price
    else:
        fill_price = market_state["mid_price"]

    return fill_price


def determine_maker_status(order: Any, market_state: Dict, latency_us: float) -> bool:
    """Determine if order is maker or taker considering latency."""
    if order.order_type.value != "limit":
        return False

    latency_factor = max(0.1, 1.0 - (latency_us - 500) / 2000)

    if order.side.value == "buy":
        is_marketable = order.price >= market_state["ask_price"]
        return not is_marketable and (np.random.random() < latency_factor)
    else:
        is_marketable = order.price <= market_state["bid_price"]
        return not is_marketable and (np.random.random() < latency_factor)


def calculate_slippage(
    order: Any, fill_price: float, arrival_price: float, latency_breakdown: LatencyBreakdown
) -> float:
    """Calculate enhanced slippage with latency considerations."""
    base_slippage_bps = abs(fill_price - arrival_price) / arrival_price * 10000

    if order.side.value == "buy" and fill_price > arrival_price:
        sign = 1.0
    elif order.side.value == "sell" and fill_price < arrival_price:
        sign = 1.0
    else:
        sign = -1.0

    latency_attribution = (latency_breakdown.total_latency_us / 1000) * 0.1
    slippage_bps = sign * (base_slippage_bps + latency_attribution)

    return slippage_bps
