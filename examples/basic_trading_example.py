import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import asyncio
import time
from typing import Dict, List

def basic_market_simulation():
    """Run a basic market data simulation"""
    
    print("🚀 Starting Basic HFT Trading Example")
    print("=" * 50)
    
    try:
        # Import technical indicators
        from data.advanced_technical_indicators import AdvancedTechnicalEngine
        
        # Create technical analysis engine
        engine = AdvancedTechnicalEngine()
        print("✅ Technical indicators engine initialized")
        
        # Simulate market data for a few symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        base_prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0}
        
        print(f"\n📊 Simulating market data for {symbols}")
        
        # Generate and process market ticks
        for tick_num in range(50):
            for symbol in symbols:
                # Generate realistic price movement
                import numpy as np
                base_price = base_prices[symbol]
                price_change = np.random.normal(0, base_price * 0.001)  # 0.1% volatility
                current_price = base_price + price_change
                
                # Generate volume with exponential distribution
                volume = int(np.random.exponential(1000))
                
                # Update the technical engine
                engine.update_data(
                    symbol=symbol,
                    price=current_price,
                    volume=volume,
                    high=current_price * 1.002,  # Slightly higher high
                    low=current_price * 0.998,   # Slightly lower low
                    bid=current_price * 0.9995,  # Bid slightly below mid
                    ask=current_price * 1.0005   # Ask slightly above mid
                )
                
                # Update base price for next iteration
                base_prices[symbol] = current_price
        
        print("✅ Generated 50 ticks per symbol")
        
        # Calculate technical indicators for each symbol
        print("\n🧮 Calculating Technical Indicators:")
        print("-" * 40)
        
        for symbol in symbols:
            result = engine.calculate_all_indicators(symbol, time.time())
            
            print(f"\n{symbol}:")
            print(f"  Market Regime: {result.market_regime}")
            print(f"  Volatility Regime: {result.volatility_regime}")
            print(f"  Liquidity Score: {result.liquidity_score:.1f}/100")
            print(f"  Indicators Count: {len(result.indicators)}")
            print(f"  Microstructure Features: {len(result.microstructure_features)}")
            
            # Show a few key indicators
            if result.indicators:
                key_indicators = ['rsi_14', 'bb_position', 'volume_ratio_short']
                for indicator in key_indicators:
                    if indicator in result.indicators:
                        print(f"  {indicator}: {result.indicators[indicator]:.3f}")
        
        print("\n" + "=" * 50)
        print("✅ Basic trading simulation completed successfully!")
        print("💡 This example showed:")
        print("   • Market data simulation")
        print("   • Technical indicator calculation") 
        print("   • Multi-symbol processing")
        print("   • Regime detection")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        print("💡 Check that the system is properly set up")

def performance_test():
    """Test the performance of core components"""
    
    print("\n🏃 Running Performance Test")
    print("-" * 30)
    
    try:
        from data.advanced_technical_indicators import AdvancedTechnicalEngine
        import numpy as np
        
        engine = AdvancedTechnicalEngine()
        
        # Performance test
        num_ticks = 1000
        start_time = time.time()
        
        for i in range(num_ticks):
            price = 100 + np.random.normal(0, 1)
            volume = int(np.random.exponential(1000))
            engine.update_data("PERF", price, volume)
        
        # Calculate indicators
        result = engine.calculate_all_indicators("PERF", time.time())
        
        end_time = time.time()
        elapsed = end_time - start_time
        ticks_per_second = num_ticks / elapsed
        
        print(f"⚡ Performance Results:")
        print(f"   Processed {num_ticks} ticks in {elapsed:.3f} seconds")
        print(f"   Speed: {ticks_per_second:.0f} ticks/second")
        print(f"   Generated {len(result.indicators)} indicators")
        
        if ticks_per_second > 5000:
            print("🚀 Excellent performance!")
        elif ticks_per_second > 1000:
            print("✅ Good performance")
        else:
            print("⚠️  Consider optimizing for better performance")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

async def async_example():
    """Example of asynchronous trading operations"""
    
    print("\n🔄 Async Trading Example")
    print("-" * 25)
    
    try:
        # Simulate async operations
        print("📡 Connecting to market data...")
        await asyncio.sleep(0.1)  # Simulate connection time
        print("✅ Connected to market data")
        
        print("🤖 Starting trading algorithms...")
        await asyncio.sleep(0.1)  # Simulate startup time
        print("✅ Trading algorithms active")
        
        print("📊 Processing market data stream...")
        # Simulate processing multiple ticks asynchronously
        tasks = []
        for i in range(5):
            task = asyncio.create_task(process_tick_async(f"tick_{i}"))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        print(f"✅ Processed {len(results)} ticks asynchronously")
        
    except Exception as e:
        print(f"❌ Async example failed: {e}")

async def process_tick_async(tick_id: str):
    """Simulate asynchronous tick processing"""
    await asyncio.sleep(0.01)  # Simulate processing time
    return f"Processed {tick_id}"

def main():
    """Main entry point for the example"""
    
    print("🏦 HFT Trading System - Basic Example")
    print("=====================================")
    
    # Run basic simulation
    basic_market_simulation()
    
    # Run performance test
    performance_test()
    
    # Run async example
    try:
        asyncio.run(async_example())
    except Exception as e:
        print(f"❌ Async example failed: {e}")
    
    print("\n🎯 Next Steps:")
    print("   1. Run: python run.py (for full system)")
    print("   2. Check: examples/quickstart/ (for more examples)")
    print("   3. Test: pytest tests/ (to run test suite)")
    print("   4. Configure: configs/ (to customize settings)")
    
    print("\n🎓 Learning Resources:")
    print("   • README.md - Full documentation")
    print("   • examples/ - More code examples") 
    print("   • tests/ - Test cases for reference")

if __name__ == "__main__":
    main()