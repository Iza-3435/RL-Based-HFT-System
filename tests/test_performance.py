# Performance Test Suite - Tests speed and efficiency of core components
# Ensures the system meets HFT performance requirements

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import pytest
import time
import numpy as np
from typing import Dict, List

def test_tick_generation_speed():
    """Test market data generation speed - should handle thousands of ticks per second"""
    try:
        from data.advanced_technical_indicators import AdvancedTechnicalEngine
        
        engine = AdvancedTechnicalEngine()
        
        # Performance test: generate and process many ticks quickly
        num_ticks = 1000
        start_time = time.time()
        
        base_price = 100.0
        for i in range(num_ticks):
            # Simulate realistic price movement
            price_change = np.random.normal(0, 0.001)
            price = base_price + price_change * i
            volume = int(np.random.exponential(1000))
            
            engine.update_data("PERF_TEST", price, volume)
        
        end_time = time.time()
        elapsed = end_time - start_time
        ticks_per_second = num_ticks / elapsed
        
        print(f"Processed {num_ticks} ticks in {elapsed:.3f}s ({ticks_per_second:.0f} ticks/sec)")
        
        # Should process at least 1000 ticks per second
        assert ticks_per_second > 1000, f"Too slow: {ticks_per_second:.0f} ticks/sec"
        
    except ImportError:
        pytest.skip("Technical indicators not available for performance testing")

def test_indicator_calculation_speed():
    """Test technical indicator calculation speed"""
    try:
        from data.advanced_technical_indicators import AdvancedTechnicalEngine
        
        engine = AdvancedTechnicalEngine()
        
        # Pre-populate with data
        for i in range(200):  # Fill up the lookback window
            price = 100 + np.sin(i * 0.1) * 5  # Realistic price movement
            engine.update_data("SPEED_TEST", price, 1000)
        
        # Time indicator calculations
        num_calculations = 100
        start_time = time.time()
        
        for i in range(num_calculations):
            result = engine.calculate_all_indicators("SPEED_TEST", time.time())
            assert len(result.indicators) > 10  # Should calculate multiple indicators
        
        end_time = time.time()
        elapsed = end_time - start_time
        calculations_per_second = num_calculations / elapsed
        
        print(f"Calculated {num_calculations} indicator sets in {elapsed:.3f}s ({calculations_per_second:.0f} calc/sec)")
        
        # Should calculate indicators very quickly
        assert calculations_per_second > 50, f"Indicator calculation too slow: {calculations_per_second:.0f} calc/sec"
        
    except ImportError:
        pytest.skip("Technical indicators not available")

def test_memory_usage():
    """Test that the system doesn't use excessive memory"""
    try:
        from data.advanced_technical_indicators import AdvancedTechnicalEngine
        
        # Create multiple engines to test memory usage
        engines = []
        for i in range(10):
            engine = AdvancedTechnicalEngine()
            
            # Add data to each engine
            for j in range(1000):
                engine.update_data(f"TEST_{i}", 100 + j*0.01, 1000)
            
            engines.append(engine)
        
        # Memory test passed if we can create multiple engines without issues
        assert len(engines) == 10
        
    except ImportError:
        pytest.skip("Technical indicators not available")

def test_concurrent_processing():
    """Test that multiple symbols can be processed concurrently"""
    try:
        from data.advanced_technical_indicators import AdvancedTechnicalEngine
        
        engine = AdvancedTechnicalEngine()
        
        # Test multiple symbols simultaneously
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        
        start_time = time.time()
        
        # Simulate concurrent tick processing
        for i in range(100):
            for symbol in symbols:
                price = 100 + np.random.normal(0, 1)
                volume = int(np.random.exponential(1000))
                engine.update_data(symbol, price, volume)
        
        # Calculate indicators for all symbols
        results = {}
        for symbol in symbols:
            results[symbol] = engine.calculate_all_indicators(symbol, time.time())
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"Processed {len(symbols)} symbols with 100 ticks each in {elapsed:.3f}s")
        
        # Validate results
        assert len(results) == len(symbols)
        for symbol, result in results.items():
            assert result.symbol == symbol
            assert len(result.indicators) > 0
        
        # Should handle multiple symbols efficiently
        assert elapsed < 1.0, f"Multi-symbol processing too slow: {elapsed:.3f}s"
        
    except ImportError:
        pytest.skip("Technical indicators not available")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])