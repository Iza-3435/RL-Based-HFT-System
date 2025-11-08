"""Tests for core type definitions."""

import pytest
from datetime import datetime

from src.core.types import (
    OrderSide, OrderType, TradingMode, MarketRegime,
    VenueConfig, OrderRequest, Trade, Position, MarketTick
)


class TestEnums:
    """Test enum definitions."""

    def test_order_side_values(self):
        """Test OrderSide enum."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_type_values(self):
        """Test OrderType enum."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"

    def test_trading_mode_values(self):
        """Test TradingMode enum."""
        assert TradingMode.FAST.value == "fast"
        assert TradingMode.BALANCED.value == "balanced"
        assert TradingMode.PRODUCTION.value == "production"

    def test_market_regime_values(self):
        """Test MarketRegime enum."""
        assert MarketRegime.LOW_VOLATILITY.value == "low_volatility"
        assert MarketRegime.HIGH_VOLATILITY.value == "high_volatility"


class TestDataClasses:
    """Test dataclass definitions."""

    def test_venue_config_creation(self):
        """Test VenueConfig dataclass."""
        venue = VenueConfig(
            name='NYSE',
            base_latency_us=850,
            latency_range_us=(50, 200),
            fee_bps=0.001,
            slippage_bps=1.5
        )

        assert venue.name == 'NYSE'
        assert venue.base_latency_us == 850
        assert venue.latency_range_us == (50, 200)
        assert venue.fee_bps == 0.001
        assert venue.slippage_bps == 1.5

    def test_order_request_creation(self):
        """Test OrderRequest dataclass."""
        order = OrderRequest(
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0,
            venue='NYSE'
        )

        assert order.symbol == 'AAPL'
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 100
        assert order.price == 150.0
        assert order.venue == 'NYSE'
        assert isinstance(order.timestamp, datetime)

    def test_trade_creation(self):
        """Test Trade dataclass."""
        now = datetime.now()
        trade = Trade(
            id='T123',
            symbol='MSFT',
            side=OrderSide.SELL,
            quantity=50,
            price=380.0,
            venue='NASDAQ',
            timestamp=now,
            latency_us=1250,
            fees=1.90
        )

        assert trade.id == 'T123'
        assert trade.symbol == 'MSFT'
        assert trade.quantity == 50
        assert trade.latency_us == 1250

    def test_position_creation(self):
        """Test Position dataclass."""
        position = Position(
            symbol='GOOGL',
            quantity=200,
            avg_price=140.5,
            unrealized_pnl=2500.0,
            realized_pnl=1200.0
        )

        assert position.symbol == 'GOOGL'
        assert position.quantity == 200
        assert position.avg_price == 140.5
        assert position.unrealized_pnl == 2500.0
        assert position.realized_pnl == 1200.0

    def test_market_tick_creation(self):
        """Test MarketTick dataclass."""
        now = datetime.now()
        tick = MarketTick(
            symbol='TSLA',
            timestamp=now,
            bid=250.0,
            ask=250.5,
            bid_size=100,
            ask_size=150,
            last_price=250.25,
            volume=10000,
            venue='NASDAQ'
        )

        assert tick.symbol == 'TSLA'
        assert tick.bid == 250.0
        assert tick.ask == 250.5
        assert tick.bid_size == 100
        assert tick.venue == 'NASDAQ'
