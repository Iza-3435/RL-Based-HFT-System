"""Slippage modeling with non-linear dynamics."""

from typing import Dict
from datetime import datetime

from src.execution.costs.types import LiquidityTier, MarketRegime, SlippageParameters


class SlippageModel:
    """Non-linear slippage model with liquidity adjustments."""

    def __init__(self, liquidity_tiers: Dict[LiquidityTier, SlippageParameters]):
        self.liquidity_tiers = liquidity_tiers

    def calculate_slippage(
        self,
        quantity: int,
        adv: float,
        spread_bps: float,
        volatility: float,
        liquidity_tier: LiquidityTier,
        regime: MarketRegime,
        hour: int
    ) -> float:
        """Calculate slippage in basis points."""
        params = self.liquidity_tiers[liquidity_tier]

        base_slippage = params.base_slippage_bps

        participation_rate = (quantity / adv) * 100
        size_impact = params.size_impact_factor * (participation_rate ** 1.5)

        vol_impact = volatility * params.volatility_multiplier

        spread_impact = spread_bps * params.spread_sensitivity

        tod_multiplier = params.time_of_day_factor.get(hour, 1.0)

        slippage_bps = (
            (base_slippage + size_impact + vol_impact + spread_impact) *
            tod_multiplier
        )

        regime_multiplier = self._get_regime_multiplier(regime)
        slippage_bps *= regime_multiplier

        return max(0, slippage_bps)

    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get regime-based multiplier."""
        multipliers = {
            MarketRegime.QUIET: 0.7,
            MarketRegime.NORMAL: 1.0,
            MarketRegime.VOLATILE: 1.8,
            MarketRegime.STRESSED: 3.0
        }
        return multipliers.get(regime, 1.0)

    def calculate_implementation_shortfall(
        self,
        decision_price: float,
        execution_price: float,
        quantity: int,
        side: str
    ) -> float:
        """Calculate implementation shortfall in basis points."""
        if side.lower() == 'buy':
            shortfall = (execution_price - decision_price) / decision_price
        else:
            shortfall = (decision_price - execution_price) / decision_price

        return shortfall * 10000
