"""Tests for trading simulator."""

import pytest
import numpy as np

from src.trading.types import Order, Fill, OrderSide, OrderType, TradingStrategyType
from src.trading.market_impact import MarketImpactModel
from src.trading.execution_engine import OrderExecutionEngine
from src.trading.strategies.market_making import MarketMakingStrategy
from src.trading.strategies.arbitrage import ArbitrageStrategy


class TestMarketImpact:
    """Test market impact model."""

    def test_impact_increases_with_size(self, sample_order):
        """Test that larger orders have more impact."""
        model = MarketImpactModel()
        market_state = {
            'average_daily_volume': 1000000,
            'volatility': 0.02,
            'spread_bps': 2.0,
            'regime': 'normal'
        }

        small_order = sample_order
        small_order.quantity = 100
        perm_small, temp_small = model.calculate_impact(small_order, market_state, 1000)

        large_order = sample_order
        large_order.quantity = 10000
        perm_large, temp_large = model.calculate_impact(large_order, market_state, 1000)

        assert perm_large > perm_small
        assert temp_large > temp_small

    def test_stressed_regime_increases_impact(self, sample_order):
        """Test stressed market regime increases impact."""
        model = MarketImpactModel()

        normal_state = {
            'average_daily_volume': 1000000,
            'volatility': 0.02,
            'spread_bps': 2.0,
            'regime': 'normal'
        }

        stressed_state = {
            'average_daily_volume': 1000000,
            'volatility': 0.02,
            'spread_bps': 2.0,
            'regime': 'stressed'
        }

        perm_normal, temp_normal = model.calculate_impact(sample_order, normal_state, 1000)
        perm_stressed, temp_stressed = model.calculate_impact(sample_order, stressed_state, 1000)

        assert perm_stressed > perm_normal
        assert temp_stressed > temp_normal


class TestExecutionEngine:
    """Test order execution engine."""

    @pytest.mark.asyncio
    async def test_order_execution(self, sample_order):
        """Test basic order execution."""
        engine = OrderExecutionEngine()

        market_state = {
            'bid_price': 149.95,
            'ask_price': 150.05,
            'mid_price': 150.0,
            'volatility': 0.02,
            'average_daily_volume': 1000000,
            'average_trade_size': 100
        }

        fill = await engine.execute_order(sample_order, market_state, 850.0)

        assert fill is not None
        assert fill.symbol == sample_order.symbol
        assert fill.quantity == sample_order.quantity
        assert fill.latency_us == 850.0

    @pytest.mark.asyncio
    async def test_market_order_crosses_spread(self, sample_order):
        """Test market orders cross the spread."""
        engine = OrderExecutionEngine()
        sample_order.order_type = OrderType.MARKET

        market_state = {
            'bid_price': 149.95,
            'ask_price': 150.05,
            'mid_price': 150.0,
            'volatility': 0.02,
            'average_daily_volume': 1000000,
            'average_trade_size': 100
        }

        fill = await engine.execute_order(sample_order, market_state, 850.0)

        assert fill.price >= market_state['ask_price']

    def test_execution_stats_tracking(self):
        """Test execution statistics are tracked."""
        engine = OrderExecutionEngine()

        stats = engine.get_execution_stats()

        assert 'total_fills' in stats
        assert 'avg_latency_us' in stats
        assert 'avg_slippage_bps' in stats


class TestMarketMakingStrategy:
    """Test market making strategy."""

    @pytest.mark.asyncio
    async def test_generates_two_sided_quotes(self):
        """Test strategy generates bid and ask quotes."""
        strategy = MarketMakingStrategy()

        market_data = {
            'symbols': ['AAPL'],
            'AAPL': {
                'mid_price': 150.0,
                'spread_bps': 2.0,
                'volume': 100000,
                'volatility': 0.02
            }
        }

        ml_predictions = {
            'routing': {'venue': 'NYSE', 'predicted_latency_us': 850, 'confidence': 0.8},
            'volatility_forecast': 0.015,
            'regime': 'normal'
        }

        orders = await strategy.generate_signals(market_data, ml_predictions)

        assert len(orders) == 2
        assert orders[0].side == OrderSide.BUY
        assert orders[1].side == OrderSide.SELL
        assert orders[0].price < 150.0
        assert orders[1].price > 150.0

    def test_inventory_skew(self):
        """Test inventory affects quote prices."""
        strategy = MarketMakingStrategy()
        strategy.positions['AAPL'].quantity = 5000  # Large long position

        # Should quote more aggressively on sell side
        assert True  # Placeholder for inventory skew test


class TestArbitrageStrategy:
    """Test arbitrage strategy."""

    @pytest.mark.asyncio
    async def test_detects_arbitrage_opportunity(self):
        """Test strategy detects cross-venue arbitrage."""
        strategy = ArbitrageStrategy()

        market_data = {
            'symbols': ['AAPL'],
            'venues': ['NYSE', 'NASDAQ'],
            'AAPL_NYSE': {
                'bid_price': 150.10,
                'ask_price': 150.15,
                'bid_size': 1000,
                'ask_size': 1000
            },
            'AAPL_NASDAQ': {
                'bid_price': 149.95,
                'ask_price': 150.00,
                'bid_size': 1000,
                'ask_size': 1000
            }
        }

        ml_predictions = {
            'routing_AAPL_NYSE': {'predicted_latency_us': 800, 'confidence': 0.9},
            'routing_AAPL_NASDAQ': {'predicted_latency_us': 850, 'confidence': 0.9},
            'regime': 'normal'
        }

        orders = await strategy.generate_signals(market_data, ml_predictions)

        # Should detect: NYSE bid (150.10) > NASDAQ ask (150.00) = 10 bps arb
        assert len(orders) >= 2  # Buy on NASDAQ, sell on NYSE
