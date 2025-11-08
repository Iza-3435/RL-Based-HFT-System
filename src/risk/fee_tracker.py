"""Fee tracking and optimization."""

from typing import Dict, List, Tuple
from collections import defaultdict


class FeeTracker:
    """Track and optimize trading fees across venues."""

    def __init__(self):
        self.fee_schedule = {
            'NYSE': {'maker': -0.0020, 'taker': 0.0030, 'remove': 0.0030},
            'NASDAQ': {'maker': -0.0025, 'taker': 0.0030, 'remove': 0.0030},
            'CBOE': {'maker': -0.0023, 'taker': 0.0028, 'remove': 0.0028},
            'IEX': {'maker': 0.0000, 'taker': 0.0009, 'remove': 0.0009},
            'ARCA': {'maker': -0.0020, 'taker': 0.0030, 'remove': 0.0030}
        }

        self.monthly_volume = defaultdict(int)
        self.monthly_fees = defaultdict(float)
        self.tier_thresholds = {
            'tier1': 0,
            'tier2': 10_000_000,
            'tier3': 50_000_000,
            'tier4': 100_000_000
        }

    def calculate_fee(
        self,
        venue: str,
        order_type: str,
        volume: int,
        price: float,
        is_maker: bool
    ) -> Tuple[float, float]:
        """Calculate fee and rebate for order."""
        fee_structure = self.fee_schedule.get(venue, {})

        if is_maker:
            rate = fee_structure.get('maker', 0)
        else:
            rate = fee_structure.get('taker', 0)

        tier = self._get_volume_tier(venue)
        tier_discount = 0.0001 * (tier - 1)

        adjusted_rate = (
            rate + tier_discount if rate > 0 else rate - tier_discount
        )

        notional = volume * price

        if adjusted_rate > 0:
            fee = notional * adjusted_rate
            rebate = 0
        else:
            fee = 0
            rebate = -notional * adjusted_rate

        self.monthly_volume[venue] += volume
        self.monthly_fees[venue] += fee - rebate

        return fee, rebate

    def _get_volume_tier(self, venue: str) -> int:
        """Get current volume tier for venue."""
        volume = self.monthly_volume[venue]

        for tier in range(4, 0, -1):
            if volume >= self.tier_thresholds[f'tier{tier}']:
                return tier

        return 1

    def optimize_venue_selection(
        self,
        venues: List[str],
        order_size: int,
        price: float,
        can_be_maker: bool
    ) -> str:
        """Select optimal venue based on fees."""
        best_venue = venues[0]
        best_cost = float('inf')

        for venue in venues:
            fee, rebate = self.calculate_fee(
                venue, 'limit', order_size, price, can_be_maker
            )
            net_cost = fee - rebate

            if net_cost < best_cost:
                best_cost = net_cost
                best_venue = venue

        return best_venue
