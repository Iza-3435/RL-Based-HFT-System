"""Tests for trading simulator modules."""

import pytest
import numpy as np

from src.simulator.trading import (
    Order,
    Fill,
    Position,
    OrderType,
    OrderSide,
    OrderStatus,
    TradingStrategyType,
    MarketImpactModel,
    OrderExecutionEngine,
)


class TestOrder:
    """Test Order dataclass."""

    def test_order_creation(self):
        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            venue="NYSE",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0,
            timestamp=1234567890.0,
            strategy=TradingStrategyType.MARKET_MAKING,
        )
        assert order.order_id == "TEST_001"
        assert order.symbol == "AAPL"
        assert order.status == OrderStatus.PENDING


class TestPosition:
    """Test Position dataclass."""

    def test_position_update_buy(self):
        position = Position(symbol="AAPL")
        fill = Fill(
            fill_id="F001",
            order_id="O001",
            symbol="AAPL",
            venue="NYSE",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            timestamp=1234567890.0,
            fees=1.5,
            rebate=0.5,
            latency_us=1000.0,
            slippage_bps=2.0,
            market_impact_bps=1.0,
        )

        position.update_position(fill, 150.0)
        assert position.quantity == 100
        assert position.average_cost == 150.0

    def test_position_update_sell(self):
        position = Position(symbol="AAPL", quantity=100, average_cost=150.0)
        fill = Fill(
            fill_id="F002",
            order_id="O002",
            symbol="AAPL",
            venue="NYSE",
            side=OrderSide.SELL,
            quantity=50,
            price=155.0,
            timestamp=1234567890.0,
            fees=1.5,
            rebate=0.5,
            latency_us=1000.0,
            slippage_bps=2.0,
            market_impact_bps=1.0,
        )

        position.update_position(fill, 155.0)
        assert position.quantity == 50
        assert position.realized_pnl == 250.0


class TestMarketImpactModel:
    """Test MarketImpactModel."""

    def test_calculate_impact(self):
        model = MarketImpactModel()
        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            venue="NYSE",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
            price=150.0,
            timestamp=1234567890.0,
            strategy=TradingStrategyType.ARBITRAGE,
        )

        market_state = {
            "average_daily_volume": 1000000,
            "volatility": 0.02,
            "spread_bps": 2.0,
        }

        permanent, temporary = model.calculate_impact(order, market_state, 1000.0)

        assert permanent > 0
        assert temporary > 0
        assert temporary > permanent


class TestOrderExecutionEngine:
    """Test OrderExecutionEngine."""

    @pytest.mark.asyncio
    async def test_execute_market_order(self):
        engine = OrderExecutionEngine()

        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            venue="NYSE",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            price=150.0,
            timestamp=1234567890.0,
            strategy=TradingStrategyType.MARKET_MAKING,
        )

        market_state = {
            "bid_price": 149.95,
            "ask_price": 150.05,
            "mid_price": 150.0,
            "spread_bps": 6.67,
            "volume": 1000,
            "volatility": 0.02,
            "average_daily_volume": 1000000,
            "average_trade_size": 100,
        }

        fill = await engine.execute_order(order, market_state, 1000.0)

        assert fill is not None
        assert fill.symbol == "AAPL"
        assert fill.quantity == 100
        assert fill.venue == "NYSE"
        assert order.status == OrderStatus.FILLED
