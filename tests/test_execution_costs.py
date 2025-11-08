"""Tests for execution cost models."""

import pytest
from datetime import datetime

from src.execution.costs.types import LiquidityTier, MarketRegime
from src.execution.costs.market_impact import EnhancedMarketImpactModel
from src.execution.costs.slippage_model import SlippageModel
from src.execution.costs.venue_costs import VenueCostCalculator
from src.execution.costs.cost_analyzer import ExecutionCostAnalyzer


class TestMarketImpact:
    """Test market impact model."""

    def test_impact_increases_with_size(self):
        """Test larger orders have more impact."""
        model = EnhancedMarketImpactModel()

        temp_small, perm_small = model.calculate_impact(
            quantity=100,
            adv=1_000_000,
            volatility=0.02,
            venue='NYSE',
            liquidity_tier=LiquidityTier.HIGH,
            hour=11
        )

        temp_large, perm_large = model.calculate_impact(
            quantity=10000,
            adv=1_000_000,
            volatility=0.02,
            venue='NYSE',
            liquidity_tier=LiquidityTier.HIGH,
            hour=11
        )

        assert temp_large > temp_small
        assert perm_large > perm_small

    def test_high_liquidity_lower_impact(self):
        """Test high liquidity tiers have lower impact."""
        model = EnhancedMarketImpactModel()

        temp_high, _ = model.calculate_impact(
            quantity=1000,
            adv=10_000_000,
            volatility=0.02,
            venue='NYSE',
            liquidity_tier=LiquidityTier.HIGH,
            hour=11
        )

        temp_low, _ = model.calculate_impact(
            quantity=1000,
            adv=100_000,
            volatility=0.02,
            venue='NYSE',
            liquidity_tier=LiquidityTier.LOW,
            hour=11
        )

        assert temp_low > temp_high

    def test_peak_hours_increase_impact(self):
        """Test peak hours have higher impact."""
        model = EnhancedMarketImpactModel()

        temp_normal, _ = model.calculate_impact(
            quantity=1000,
            adv=1_000_000,
            volatility=0.02,
            venue='NYSE',
            liquidity_tier=LiquidityTier.HIGH,
            hour=11  # Normal hour
        )

        temp_peak, _ = model.calculate_impact(
            quantity=1000,
            adv=1_000_000,
            volatility=0.02,
            venue='NYSE',
            liquidity_tier=LiquidityTier.HIGH,
            hour=9  # Market open
        )

        assert temp_peak > temp_normal


class TestSlippageModel:
    """Test slippage model."""

    def test_slippage_positive(self):
        """Test slippage is always positive."""
        model = EnhancedMarketImpactModel()
        slippage_model = SlippageModel(model.liquidity_tiers)

        slippage = slippage_model.calculate_slippage(
            quantity=1000,
            adv=1_000_000,
            spread_bps=2.0,
            volatility=0.02,
            liquidity_tier=LiquidityTier.HIGH,
            regime=MarketRegime.NORMAL,
            hour=11
        )

        assert slippage >= 0

    def test_stressed_regime_increases_slippage(self):
        """Test stressed market increases slippage."""
        model = EnhancedMarketImpactModel()
        slippage_model = SlippageModel(model.liquidity_tiers)

        normal_slippage = slippage_model.calculate_slippage(
            quantity=1000,
            adv=1_000_000,
            spread_bps=2.0,
            volatility=0.02,
            liquidity_tier=LiquidityTier.HIGH,
            regime=MarketRegime.NORMAL,
            hour=11
        )

        stressed_slippage = slippage_model.calculate_slippage(
            quantity=1000,
            adv=1_000_000,
            spread_bps=2.0,
            volatility=0.02,
            liquidity_tier=LiquidityTier.HIGH,
            regime=MarketRegime.STRESSED,
            hour=11
        )

        assert stressed_slippage > normal_slippage

    def test_implementation_shortfall(self):
        """Test implementation shortfall calculation."""
        model = EnhancedMarketImpactModel()
        slippage_model = SlippageModel(model.liquidity_tiers)

        shortfall = slippage_model.calculate_implementation_shortfall(
            decision_price=100.0,
            execution_price=100.10,
            quantity=1000,
            side='buy'
        )

        assert shortfall == 10.0  # 10 bps


class TestVenueCosts:
    """Test venue cost calculator."""

    def test_maker_receives_rebate(self):
        """Test maker orders receive rebates."""
        calculator = VenueCostCalculator()

        fees, rebates = calculator.calculate_fees(
            venue='NYSE',
            quantity=1000,
            price=100.0,
            is_maker=True
        )

        assert rebates > 0
        assert fees == 0

    def test_taker_pays_fees(self):
        """Test taker orders pay fees."""
        calculator = VenueCostCalculator()

        fees, rebates = calculator.calculate_fees(
            venue='NYSE',
            quantity=1000,
            price=100.0,
            is_maker=False
        )

        assert fees > 0
        assert rebates == 0

    def test_iex_no_rebates(self):
        """Test IEX has no rebates."""
        calculator = VenueCostCalculator()

        fees, rebates = calculator.calculate_fees(
            venue='IEX',
            quantity=1000,
            price=100.0,
            is_maker=True
        )

        assert rebates == 0

    def test_peak_hours_multiplier(self):
        """Test peak hours increase costs."""
        calculator = VenueCostCalculator()

        normal_mult = calculator.get_venue_multipliers('NYSE', 11)
        peak_mult = calculator.get_venue_multipliers('NYSE', 9)

        assert peak_mult['impact'] > normal_mult['impact']


class TestCostAnalyzer:
    """Test execution cost analyzer."""

    def test_cost_calculation_complete(self, sample_order):
        """Test comprehensive cost calculation."""
        analyzer = ExecutionCostAnalyzer()

        market_state = {
            'average_daily_volume': 5_000_000,
            'spread_bps': 2.0,
            'volatility': 0.02,
            'regime': 'normal'
        }

        breakdown = analyzer.calculate_execution_costs(
            order=sample_order,
            market_state=market_state,
            actual_latency_us=850.0,
            execution_price=150.05,
            is_maker=False
        )

        assert breakdown.total_transaction_cost > 0
        assert breakdown.cost_per_share > 0
        assert breakdown.cost_bps > 0
        assert breakdown.slippage_cost >= 0
        assert breakdown.market_impact_cost >= 0

    def test_maker_costs_lower(self, sample_order):
        """Test maker orders have lower costs."""
        analyzer = ExecutionCostAnalyzer()

        market_state = {
            'average_daily_volume': 5_000_000,
            'spread_bps': 2.0,
            'volatility': 0.02,
            'regime': 'normal'
        }

        maker_breakdown = analyzer.calculate_execution_costs(
            order=sample_order,
            market_state=market_state,
            actual_latency_us=850.0,
            execution_price=150.00,
            is_maker=True
        )

        taker_breakdown = analyzer.calculate_execution_costs(
            order=sample_order,
            market_state=market_state,
            actual_latency_us=850.0,
            execution_price=150.05,
            is_maker=False
        )

        assert maker_breakdown.total_transaction_cost < taker_breakdown.total_transaction_cost

    def test_liquidity_tier_determination(self):
        """Test liquidity tier classification."""
        analyzer = ExecutionCostAnalyzer()

        high_tier = analyzer._determine_liquidity_tier(20_000_000)
        medium_tier = analyzer._determine_liquidity_tier(5_000_000)
        low_tier = analyzer._determine_liquidity_tier(500_000)

        assert high_tier == LiquidityTier.HIGH
        assert medium_tier == LiquidityTier.MEDIUM
        assert low_tier == LiquidityTier.LOW
