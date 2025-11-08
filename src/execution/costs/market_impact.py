"""Market impact model with non-linear dynamics."""

from typing import Dict, Tuple
from collections import defaultdict, deque
import numpy as np

from src.core.logging_config import get_logger
from src.execution.costs.types import (
    LiquidityTier, MarketImpactParameters, SlippageParameters
)

logger = get_logger()


class EnhancedMarketImpactModel:
    """Production-grade market impact model."""

    def __init__(self):
        self.liquidity_tiers = self._initialize_liquidity_tiers()
        self.impact_parameters = self._initialize_impact_parameters()

        self.recent_trades = defaultdict(deque)
        self.price_impact_decay = defaultdict(float)
        self.impact_history = defaultdict(list)

        logger.info("Market impact model initialized")

    def _initialize_liquidity_tiers(self) -> Dict[LiquidityTier, SlippageParameters]:
        """Initialize liquidity tier parameters."""
        return {
            LiquidityTier.HIGH: SlippageParameters(
                base_slippage_bps=0.5,
                size_impact_factor=0.2,
                volatility_multiplier=15.0,
                spread_sensitivity=0.8,
                time_of_day_factor={
                    9: 2.0, 10: 1.5, 11: 1.0, 12: 0.8,
                    13: 0.9, 14: 1.1, 15: 1.8, 16: 2.5
                }
            ),
            LiquidityTier.MEDIUM: SlippageParameters(
                base_slippage_bps=1.2,
                size_impact_factor=0.4,
                volatility_multiplier=25.0,
                spread_sensitivity=1.2,
                time_of_day_factor={
                    9: 2.5, 10: 1.8, 11: 1.2, 12: 1.0,
                    13: 1.1, 14: 1.3, 15: 2.2, 16: 3.0
                }
            ),
            LiquidityTier.LOW: SlippageParameters(
                base_slippage_bps=3.0,
                size_impact_factor=0.8,
                volatility_multiplier=40.0,
                spread_sensitivity=1.8,
                time_of_day_factor={
                    9: 3.5, 10: 2.5, 11: 1.5, 12: 1.2,
                    13: 1.3, 14: 1.6, 15: 3.0, 16: 4.0
                }
            )
        }

    def _initialize_impact_parameters(self) -> Dict[LiquidityTier, MarketImpactParameters]:
        """Initialize market impact parameters."""
        return {
            LiquidityTier.HIGH: MarketImpactParameters(
                temporary_impact_base=0.05,
                permanent_impact_base=0.02,
                volatility_scaling=20.0,
                adv_scaling=1.0,
                venue_multipliers={
                    'NYSE': 0.9, 'NASDAQ': 1.0, 'ARCA': 1.1,
                    'CBOE': 1.2, 'IEX': 0.95
                }
            ),
            LiquidityTier.MEDIUM: MarketImpactParameters(
                temporary_impact_base=0.12,
                permanent_impact_base=0.05,
                volatility_scaling=30.0,
                adv_scaling=1.2,
                venue_multipliers={
                    'NYSE': 1.0, 'NASDAQ': 1.1, 'ARCA': 1.3,
                    'CBOE': 1.4, 'IEX': 1.1
                }
            ),
            LiquidityTier.LOW: MarketImpactParameters(
                temporary_impact_base=0.25,
                permanent_impact_base=0.10,
                volatility_scaling=50.0,
                adv_scaling=1.5,
                venue_multipliers={
                    'NYSE': 1.1, 'NASDAQ': 1.2, 'ARCA': 1.5,
                    'CBOE': 1.8, 'IEX': 1.3
                }
            )
        }

    def calculate_impact(
        self,
        quantity: int,
        adv: float,
        volatility: float,
        venue: str,
        liquidity_tier: LiquidityTier,
        hour: int
    ) -> Tuple[float, float]:
        """Calculate temporary and permanent market impact."""
        params = self.impact_parameters[liquidity_tier]

        participation_rate = (quantity / adv) * 100

        if params.sqrt_scaling:
            size_factor = np.sqrt(participation_rate)
        else:
            size_factor = participation_rate

        vol_adjustment = volatility * params.volatility_scaling

        temporary_impact = (
            params.temporary_impact_base * size_factor * vol_adjustment
        )
        permanent_impact = (
            params.permanent_impact_base * size_factor * vol_adjustment
        )

        venue_mult = params.venue_multipliers.get(venue, 1.0)
        temporary_impact *= venue_mult
        permanent_impact *= venue_mult

        tod_params = self.liquidity_tiers[liquidity_tier]
        tod_mult = tod_params.time_of_day_factor.get(hour, 1.0)
        temporary_impact *= tod_mult

        return temporary_impact, permanent_impact

    def update_impact_decay(self, venue: str, dt: float):
        """Update decaying temporary impact."""
        decay_rate = 0.1
        self.price_impact_decay[venue] *= np.exp(-decay_rate * dt)
