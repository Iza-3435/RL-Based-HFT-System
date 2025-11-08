"""Comprehensive execution cost analysis."""

from typing import Dict
from datetime import datetime

from src.core.logging_config import get_logger
from src.execution.costs.types import (
    ExecutionCostBreakdown, LiquidityTier, MarketRegime
)
from src.execution.costs.market_impact import EnhancedMarketImpactModel
from src.execution.costs.slippage_model import SlippageModel
from src.execution.costs.venue_costs import VenueCostCalculator

logger = get_logger()


class ExecutionCostAnalyzer:
    """Calculate comprehensive execution costs."""

    def __init__(self):
        self.impact_model = EnhancedMarketImpactModel()
        self.slippage_model = SlippageModel(
            self.impact_model.liquidity_tiers
        )
        self.venue_calculator = VenueCostCalculator()

        logger.info("Execution cost analyzer initialized")

    def calculate_execution_costs(
        self,
        order,
        market_state: Dict,
        actual_latency_us: float,
        execution_price: float,
        is_maker: bool = False
    ) -> ExecutionCostBreakdown:
        """Calculate comprehensive execution costs."""
        liquidity_tier = self._determine_liquidity_tier(
            market_state.get('average_daily_volume', 1000000)
        )

        regime = MarketRegime(market_state.get('regime', 'normal'))

        hour = datetime.fromtimestamp(order.timestamp).hour

        slippage_bps = self.slippage_model.calculate_slippage(
            quantity=order.quantity,
            adv=market_state.get('average_daily_volume', 1000000),
            spread_bps=market_state.get('spread_bps', 2.0),
            volatility=market_state.get('volatility', 0.02),
            liquidity_tier=liquidity_tier,
            regime=regime,
            hour=hour
        )

        temp_impact, perm_impact = self.impact_model.calculate_impact(
            quantity=order.quantity,
            adv=market_state.get('average_daily_volume', 1000000),
            volatility=market_state.get('volatility', 0.02),
            venue=order.venue,
            liquidity_tier=liquidity_tier,
            hour=hour
        )

        notional = order.quantity * execution_price

        slippage_cost = notional * slippage_bps / 10000
        temp_impact_cost = notional * temp_impact / 10000
        perm_impact_cost = notional * perm_impact / 10000
        market_impact_cost = temp_impact_cost + perm_impact_cost

        latency_cost = self._calculate_latency_cost(
            actual_latency_us,
            notional,
            market_state.get('volatility', 0.02)
        )

        fees, rebates = self.venue_calculator.calculate_fees(
            venue=order.venue,
            quantity=order.quantity,
            price=execution_price,
            is_maker=is_maker
        )

        impl_shortfall = self.slippage_model.calculate_implementation_shortfall(
            decision_price=order.price,
            execution_price=execution_price,
            quantity=order.quantity,
            side=order.side.value
        )

        gross_cost = slippage_cost + market_impact_cost + latency_cost + fees
        net_cost = gross_cost - rebates
        total_cost = net_cost

        cost_per_share = total_cost / order.quantity if order.quantity > 0 else 0
        cost_bps = (total_cost / notional) * 10000 if notional > 0 else 0

        return ExecutionCostBreakdown(
            order_id=order.order_id,
            symbol=order.symbol,
            venue=order.venue,
            timestamp=order.timestamp,
            side=order.side.value,
            quantity=order.quantity,
            order_price=order.price,
            execution_price=execution_price,
            slippage_cost=slippage_cost,
            temporary_impact_cost=temp_impact_cost,
            permanent_impact_cost=perm_impact_cost,
            market_impact_cost=market_impact_cost,
            latency_cost=latency_cost,
            fees_paid=fees,
            rebates_received=rebates,
            opportunity_cost=0.0,
            gross_execution_cost=gross_cost,
            net_execution_cost=net_cost,
            total_transaction_cost=total_cost,
            cost_per_share=cost_per_share,
            cost_bps=cost_bps,
            implementation_shortfall_bps=impl_shortfall
        )

    def _determine_liquidity_tier(self, adv: float) -> LiquidityTier:
        """Determine liquidity tier based on ADV."""
        if adv > 10_000_000:
            return LiquidityTier.HIGH
        elif adv > 1_000_000:
            return LiquidityTier.MEDIUM
        else:
            return LiquidityTier.LOW

    def _calculate_latency_cost(
        self,
        latency_us: float,
        notional: float,
        volatility: float
    ) -> float:
        """Calculate opportunity cost from latency."""
        latency_ms = latency_us / 1000
        drift_per_ms = volatility / (252 * 6.5 * 3600 * 1000)
        expected_drift = drift_per_ms * latency_ms
        return notional * expected_drift
