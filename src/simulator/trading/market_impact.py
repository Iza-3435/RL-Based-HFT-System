"""Market impact modeling for HFT."""

from typing import Dict, Tuple

import numpy as np

from .dataclasses import Order


class MarketImpactModel:
    """Realistic market impact modeling for HFT."""

    def __init__(self):
        self.permanent_impact_factor = 0.1
        self.temporary_impact_factor = 0.2
        self.latency_impact_factor = 0.05

    def calculate_impact(
        self, order: Order, market_state: Dict, latency_us: float
    ) -> Tuple[float, float]:
        """
        Calculate market impact in basis points.

        Returns:
            (permanent_impact_bps, temporary_impact_bps)
        """
        adv = market_state.get("average_daily_volume", 1000000)
        volatility = market_state.get("volatility", 0.02)

        order_size_pct = (order.quantity / adv) * 100

        permanent_impact = (
            self.permanent_impact_factor * np.sqrt(order_size_pct) * volatility
        )

        temporary_impact = (
            self.temporary_impact_factor * order_size_pct * np.sqrt(volatility)
        )

        latency_impact = self.latency_impact_factor * (latency_us / 100)

        total_temporary = temporary_impact + latency_impact

        if market_state.get("regime") == "stressed":
            permanent_impact *= 2.0
            total_temporary *= 2.5
        elif market_state.get("regime") == "quiet":
            permanent_impact *= 0.5
            total_temporary *= 0.7

        return permanent_impact, total_temporary
