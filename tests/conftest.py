"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

from src.trading.types import Order, Fill, OrderSide, OrderType, TradingStrategyType
from src.risk.types import RiskMetric, RiskLevel


@pytest.fixture
def sample_order():
    """Sample trading order."""
    return Order(
        order_id="TEST_001",
        symbol="AAPL",
        venue="NYSE",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=150.0,
        timestamp=datetime.now().timestamp(),
        strategy=TradingStrategyType.MARKET_MAKING
    )


@pytest.fixture
def sample_fill():
    """Sample trade fill."""
    return Fill(
        fill_id="FILL_001",
        order_id="TEST_001",
        symbol="AAPL",
        venue="NYSE",
        side=OrderSide.BUY,
        quantity=100,
        price=150.0,
        timestamp=datetime.now().timestamp(),
        fees=0.30,
        rebate=0.0,
        latency_us=850.0,
        slippage_bps=1.5,
        market_impact_bps=0.8
    )


@pytest.fixture
def market_prices():
    """Sample market prices."""
    return {
        'AAPL': 150.0,
        'MSFT': 300.0,
        'GOOGL': 2800.0,
        'TSLA': 250.0
    }


@pytest.fixture
def sample_features():
    """Sample feature vector for ML models."""
    return np.random.randn(45).astype(np.float32)


@pytest.fixture
def sample_targets():
    """Sample target values for ML models."""
    return np.random.uniform(500, 1500, 100).astype(np.float32)


@pytest.fixture
def venues():
    """Sample venue list."""
    return ['NYSE', 'NASDAQ', 'CBOE', 'IEX', 'ARCA']


@pytest.fixture
def symbols():
    """Sample symbol list."""
    return ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
