"""Tests for enhanced trading simulator modules."""

import pytest

from src.simulator.enhanced_trading import (
    create_enhanced_trading_simulator,
    quick_latency_test,
)


class TestEnhancedTradingSimulator:
    """Test EnhancedTradingSimulator creation and configuration."""

    def test_create_simulator(self):
        symbols = ["AAPL", "MSFT"]
        venues = ["NYSE", "NASDAQ"]

        simulator = create_enhanced_trading_simulator(symbols, venues)

        assert simulator is not None
        assert simulator.symbols == symbols
        assert simulator.venues == venues
        assert hasattr(simulator, "latency_analytics")
        assert hasattr(simulator, "network_simulator")

    def test_create_simulator_with_config(self):
        symbols = ["AAPL"]
        venues = ["NYSE"]
        config = {"enable_latency_simulation": False}

        simulator = create_enhanced_trading_simulator(symbols, venues, config)

        assert simulator is not None


class TestLatencyAnalytics:
    """Test latency analytics functionality."""

    def test_quick_latency_test_no_simulator(self):
        from src.simulator.trading import TradingSimulator

        basic_simulator = TradingSimulator(["NYSE"], ["AAPL"])
        result = quick_latency_test(basic_simulator, "NYSE", 10)

        assert "error" in result
        assert result["error"] == "Latency simulator not available"
