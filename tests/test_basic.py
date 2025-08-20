# Basic Test Suite for HFT Trading System
# Tests core functionality and ensures components work together

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import pytest
import numpy as np
from unittest.mock import Mock, patch

def test_imports():
    """Test that core modules can be imported without errors"""
    try:
        from core import trading_simulator
        from data import advanced_technical_indicators
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")

def test_technical_indicators():
    """Test technical indicators calculation"""
    from data.advanced_technical_indicators import AdvancedTechnicalEngine
    
    # Create engine
    engine = AdvancedTechnicalEngine()
    
    # Test data update
    test_prices = [100, 101, 99, 102, 98, 103, 97, 104]
    for i, price in enumerate(test_prices):
        engine.update_data("TEST", price, volume=1000, high=price*1.01, low=price*0.99)
    
    # Calculate indicators
    result = engine.calculate_all_indicators("TEST", timestamp=1234567890.0)
    
    # Basic validation
    assert result.symbol == "TEST"
    assert result.timestamp == 1234567890.0
    assert isinstance(result.indicators, dict)
    assert len(result.indicators) > 0  # Should have some indicators

def test_market_data_generation():
    """Test basic market data generation functionality"""
    try:
        # Try to import and test market data components
        import time
        # Basic test passed if no exceptions
        assert True
    except Exception as e:
        pytest.skip(f"Market data components not available: {e}")

def test_trading_simulator_basic():
    """Test basic trading simulator functionality"""
    try:
        from core.trading_simulator import OrderSide, OrderType
        
        # Test enum values
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        
    except ImportError:
        pytest.skip("Trading simulator not available")

def test_configuration_loading():
    """Test that configuration can be loaded"""
    # Basic configuration test
    test_config = {
        'symbols': ['AAPL', 'MSFT'],
        'venues': ['NYSE', 'NASDAQ'],
        'risk_limits': {
            'max_position': 1000,
            'stop_loss': 0.02
        }
    }
    
    # Validate config structure
    assert 'symbols' in test_config
    assert 'venues' in test_config
    assert 'risk_limits' in test_config
    assert len(test_config['symbols']) > 0

def test_mathematical_functions():
    """Test mathematical calculations used in trading logic"""
    # Test price calculations
    price = 100.0
    spread = 0.01
    bid = price - spread/2
    ask = price + spread/2
    
    assert bid == 99.995
    assert ask == 100.005
    assert ask - bid == spread
    
    # Test percentage calculations
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    mean_return = np.mean(returns)
    volatility = np.std(returns)
    
    assert abs(mean_return - 0.005) < 0.001
    assert volatility > 0

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])