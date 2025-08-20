import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'python'))

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class BacktestResult:
    """Simple backtest results"""
    total_return: float
    num_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    final_portfolio_value: float

class SimpleBacktester:
    """Basic backtesting engine for testing trading strategies"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.portfolio_values = []
        
    def buy(self, symbol: str, quantity: int, price: float) -> bool:
        """Execute a buy order"""
        cost = quantity * price
        
        if cost <= self.current_capital:
            self.current_capital -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            self.trades.append(('BUY', symbol, quantity, price, time.time()))
            return True
        return False
    
    def sell(self, symbol: str, quantity: int, price: float) -> bool:
        """Execute a sell order"""
        current_position = self.positions.get(symbol, 0)
        
        if quantity <= current_position:
            proceeds = quantity * price
            self.current_capital += proceeds
            self.positions[symbol] = current_position - quantity
            self.trades.append(('SELL', symbol, quantity, price, time.time()))
            return True
        return False
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        total_value = self.current_capital
        
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                total_value += quantity * current_prices[symbol]
        
        return total_value
    
    def calculate_results(self) -> BacktestResult:
        """Calculate backtest performance metrics"""
        if not self.portfolio_values:
            return BacktestResult(0, 0, 0, 0, 0, self.initial_capital)
        
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate win rate
        winning_trades = 0
        total_trades = len(self.trades) // 2  # Buy/sell pairs
        
        # Simple win rate calculation (every sell after buy)
        buy_prices = {}
        for trade in self.trades:
            action, symbol, quantity, price, timestamp = trade
            if action == 'BUY':
                buy_prices[symbol] = price
            elif action == 'SELL' and symbol in buy_prices:
                if price > buy_prices[symbol]:
                    winning_trades += 1
        
        win_rate = winning_trades / max(total_trades, 1)
        
        # Calculate max drawdown
        max_value = max(self.portfolio_values)
        max_drawdown = 0
        for value in self.portfolio_values:
            drawdown = (max_value - value) / max_value
            max_drawdown = max(max_drawdown, drawdown)
        
        # Simple Sharpe ratio calculation
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        return BacktestResult(
            total_return=total_return,
            num_trades=len(self.trades),
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            final_portfolio_value=final_value
        )

class SimpleMovingAverageStrategy:
    """Simple moving average crossover strategy"""
    
    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.price_history = {}
        
    def update(self, symbol: str, price: float):
        """Update price history"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Keep only necessary history
        max_period = max(self.fast_period, self.slow_period)
        if len(self.price_history[symbol]) > max_period * 2:
            self.price_history[symbol] = self.price_history[symbol][-max_period * 2:]
    
    def get_signal(self, symbol: str) -> str:
        """Get trading signal: BUY, SELL, or HOLD"""
        if symbol not in self.price_history:
            return "HOLD"
        
        prices = self.price_history[symbol]
        
        if len(prices) < self.slow_period:
            return "HOLD"
        
        # Calculate moving averages
        fast_ma = np.mean(prices[-self.fast_period:])
        slow_ma = np.mean(prices[-self.slow_period:])
        
        # Previous moving averages for crossover detection
        if len(prices) < self.slow_period + 1:
            return "HOLD"
            
        prev_fast_ma = np.mean(prices[-(self.fast_period + 1):-1])
        prev_slow_ma = np.mean(prices[-(self.slow_period + 1):-1])
        
        # Detect crossovers
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            return "BUY"  # Golden cross
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            return "SELL"  # Death cross
        else:
            return "HOLD"

def generate_sample_data(symbols: List[str], days: int = 30) -> Dict[str, List[Tuple[float, int]]]:
    """Generate sample price data for backtesting"""
    
    print(f"📊 Generating {days} days of sample data for {symbols}")
    
    data = {}
    base_prices = {"AAPL": 150, "MSFT": 300, "GOOGL": 2500}
    
    for symbol in symbols:
        prices = []
        current_price = base_prices.get(symbol, 100)
        
        # Generate realistic price movement
        for day in range(days):
            for tick in range(100):  # 100 ticks per day
                # Random walk with slight upward bias
                change = np.random.normal(0.0001, 0.01)  # 1% volatility
                current_price *= (1 + change)
                
                # Volume (not used in this simple example)
                volume = int(np.random.exponential(1000))
                
                prices.append((current_price, volume))
        
        data[symbol] = prices
    
    return data

def run_simple_backtest():
    """Run a simple backtesting example"""
    
    print("🏦 Simple Backtesting Example")
    print("=" * 40)
    
    # Configuration
    symbols = ["AAPL", "MSFT"]
    initial_capital = 100000
    position_size = 100  # shares per trade
    
    # Initialize components
    backtester = SimpleBacktester(initial_capital)
    strategy = SimpleMovingAverageStrategy(fast_period=5, slow_period=20)
    
    # Generate sample data
    market_data = generate_sample_data(symbols, days=30)
    
    print(f"💰 Starting backtest with ${initial_capital:,} capital")
    print(f"📈 Strategy: {strategy.fast_period}/{strategy.slow_period} Moving Average Crossover")
    print(f"🎯 Testing on symbols: {symbols}")
    
    # Run backtest
    print("\n🚀 Running backtest...")
    
    current_prices = {}
    step = 0
    
    # Process each tick
    max_steps = len(market_data[symbols[0]])
    
    for i in range(max_steps):
        # Update current prices and strategy
        for symbol in symbols:
            if i < len(market_data[symbol]):
                price, volume = market_data[symbol][i]
                current_prices[symbol] = price
                strategy.update(symbol, price)
        
        # Check for trading signals every 10 steps
        if i % 10 == 0:
            for symbol in symbols:
                if symbol in current_prices:
                    signal = strategy.get_signal(symbol)
                    price = current_prices[symbol]
                    
                    current_position = backtester.positions.get(symbol, 0)
                    
                    if signal == "BUY" and current_position == 0:
                        backtester.buy(symbol, position_size, price)
                        print(f"📈 BUY {position_size} {symbol} at ${price:.2f}")
                        
                    elif signal == "SELL" and current_position > 0:
                        backtester.sell(symbol, current_position, price)
                        print(f"📉 SELL {current_position} {symbol} at ${price:.2f}")
        
        # Record portfolio value
        portfolio_value = backtester.get_portfolio_value(current_prices)
        backtester.portfolio_values.append(portfolio_value)
        
        step += 1
        
        # Progress indicator
        if step % 500 == 0:
            progress = (step / max_steps) * 100
            print(f"⏳ Progress: {progress:.1f}% - Portfolio: ${portfolio_value:,.2f}")
    
    # Calculate results
    results = backtester.calculate_results()
    
    # Display results
    print("\n" + "=" * 40)
    print("📊 BACKTEST RESULTS")
    print("=" * 40)
    print(f"Initial Capital:    ${backtester.initial_capital:,.2f}")
    print(f"Final Value:        ${results.final_portfolio_value:,.2f}")
    print(f"Total Return:       {results.total_return:.2%}")
    print(f"Number of Trades:   {results.num_trades}")
    print(f"Win Rate:           {results.win_rate:.2%}")
    print(f"Max Drawdown:       {results.max_drawdown:.2%}")
    print(f"Sharpe Ratio:       {results.sharpe_ratio:.2f}")
    
    # Performance assessment
    if results.total_return > 0.1:
        print("🚀 Excellent performance!")
    elif results.total_return > 0:
        print("✅ Positive returns")
    else:
        print("📉 Strategy needs improvement")
    
    # Final positions
    if backtester.positions:
        print(f"\nFinal Positions:")
        for symbol, quantity in backtester.positions.items():
            if quantity > 0:
                value = quantity * current_prices.get(symbol, 0)
                print(f"  {symbol}: {quantity} shares (${value:,.2f})")

def main():
    """Main entry point"""
    
    try:
        run_simple_backtest()
        
        print(f"\n💡 Next Steps:")
        print("   • Modify the strategy parameters")
        print("   • Add more sophisticated indicators")
        print("   • Test with different symbols")
        print("   • Implement risk management")
        print("   • Add transaction costs")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()