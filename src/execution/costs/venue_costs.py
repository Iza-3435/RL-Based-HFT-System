"""Venue-specific cost calculation."""

from typing import Dict

from src.execution.costs.types import VenueCostProfile


class VenueCostCalculator:
    """Calculate venue-specific costs."""

    def __init__(self):
        self.venue_profiles = self._initialize_venue_profiles()

    def _initialize_venue_profiles(self) -> Dict[str, VenueCostProfile]:
        """Initialize realistic venue cost profiles."""
        return {
            'NYSE': VenueCostProfile(
                name='NYSE',
                maker_fee_bps=-0.20,
                taker_fee_bps=0.30,
                rebate_bps=0.20,
                impact_multiplier=0.95,
                liquidity_factor=1.2,
                latency_sensitivity=1.0,
                fill_probability=0.85,
                adverse_selection_factor=0.9
            ),
            'NASDAQ': VenueCostProfile(
                name='NASDAQ',
                maker_fee_bps=-0.25,
                taker_fee_bps=0.30,
                rebate_bps=0.25,
                impact_multiplier=1.0,
                liquidity_factor=1.1,
                latency_sensitivity=0.9,
                fill_probability=0.82,
                adverse_selection_factor=1.0
            ),
            'ARCA': VenueCostProfile(
                name='ARCA',
                maker_fee_bps=-0.20,
                taker_fee_bps=0.30,
                rebate_bps=0.20,
                impact_multiplier=1.1,
                liquidity_factor=0.9,
                latency_sensitivity=1.1,
                fill_probability=0.78,
                adverse_selection_factor=1.1
            ),
            'CBOE': VenueCostProfile(
                name='CBOE',
                maker_fee_bps=-0.23,
                taker_fee_bps=0.28,
                rebate_bps=0.23,
                impact_multiplier=1.25,
                liquidity_factor=0.7,
                latency_sensitivity=1.3,
                fill_probability=0.70,
                adverse_selection_factor=1.3
            ),
            'IEX': VenueCostProfile(
                name='IEX',
                maker_fee_bps=0.0,
                taker_fee_bps=0.09,
                rebate_bps=0.0,
                impact_multiplier=0.85,
                liquidity_factor=0.8,
                latency_sensitivity=0.7,
                fill_probability=0.75,
                adverse_selection_factor=0.7
            )
        }

    def calculate_fees(
        self,
        venue: str,
        quantity: int,
        price: float,
        is_maker: bool
    ) -> tuple:
        """Calculate fees and rebates."""
        profile = self.venue_profiles.get(venue)
        if not profile:
            return 0.0, 0.0

        notional = quantity * price

        if is_maker:
            fee_bps = profile.maker_fee_bps
            rebate_bps = profile.rebate_bps
        else:
            fee_bps = profile.taker_fee_bps
            rebate_bps = 0.0

        fees = max(0, notional * fee_bps / 10000)
        rebates = notional * rebate_bps / 10000

        return fees, rebates

    def get_venue_multipliers(
        self,
        venue: str,
        hour: int
    ) -> Dict[str, float]:
        """Get venue-specific cost multipliers."""
        profile = self.venue_profiles.get(venue)
        if not profile:
            return {
                'impact': 1.0,
                'liquidity': 1.0,
                'latency': 1.0,
                'adverse_selection': 1.0
            }

        is_peak = hour in profile.peak_hours
        peak_mult = profile.peak_cost_multiplier if is_peak else 1.0

        return {
            'impact': profile.impact_multiplier * peak_mult,
            'liquidity': profile.liquidity_factor,
            'latency': profile.latency_sensitivity,
            'adverse_selection': profile.adverse_selection_factor
        }

    def get_fill_probability(self, venue: str) -> float:
        """Get venue fill probability."""
        profile = self.venue_profiles.get(venue)
        return profile.fill_probability if profile else 0.75
