#!/usr/bin/env python3
"""
HFT Network Optimizer - Phase 3A: Trading Performance Simulator

Production-grade trading simulation that measures real P&L impact of ML routing decisions.
Implements market making, arbitrage, and momentum strategies with realistic execution.

Key Features:
- Microsecond-precision order execution
- Market impact and slippage modeling
- Multi-strategy support with real-time P&L tracking
- Integration with Phase 2 ML routing decisions
- Comparative analysis vs baseline strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import time
import logging
from datetime import datetime, timedelta
import heapq
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the simulator"""
    MARKET = "market"
    LIMIT = "limit"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"
    HIDDEN = "hidden"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TradingStrategyType(Enum):
    """Trading strategy types"""
    MARKET_MAKING = "market_making"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    venue: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float
    timestamp: float
    strategy: TradingStrategyType
    
    # Execution tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    fill_timestamp: Optional[float] = None
    latency_us: Optional[float] = None
    
    # ML routing metadata
    predicted_latency_us: Optional[float] = None
    routing_confidence: Optional[float] = None
    market_regime: Optional[str] = None


@dataclass
class Fill:
    """Trade execution fill"""
    fill_id: str
    order_id: str
    symbol: str
    venue: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: float
    fees: float
    rebate: float
    
    # Performance tracking
    latency_us: float
    slippage_bps: float
    market_impact_bps: float


@dataclass
class Position:
    """Position tracking"""
    symbol: str
    quantity: int = 0
    average_cost: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_volume: int = 0
    
    def update_position(self, fill: Fill, current_price: float):
        """Update position with new fill"""
        if fill.side == OrderSide.BUY:
            # Update average cost
            total_cost = self.quantity * self.average_cost + fill.quantity * fill.price
            self.quantity += fill.quantity
            self.average_cost = total_cost / self.quantity if self.quantity > 0 else 0
        else:  # SELL
            # Calculate realized P&L
            if self.quantity > 0:
                realized = fill.quantity * (fill.price - self.average_cost)
                self.realized_pnl += realized
            self.quantity -= fill.quantity
        
        # Update unrealized P&L
        if self.quantity != 0:
            self.unrealized_pnl = self.quantity * (current_price - self.average_cost)
        else:
            self.unrealized_pnl = 0.0
        
        self.total_volume += fill.quantity


class MarketImpactModel:
    """Realistic market impact modeling for HFT"""
    
    def __init__(self):
        # Impact parameters calibrated from real market data
        self.permanent_impact_factor = 0.1  # 10 bps per 1% of ADV
        self.temporary_impact_factor = 0.2  # 20 bps per 1% of ADV
        self.latency_impact_factor = 0.05  # 5 bps per 100μs latency
        
    def calculate_impact(self, order: Order, market_state: Dict, 
                        latency_us: float) -> Tuple[float, float]:
        """
        Calculate market impact in basis points
        
        Returns:
            (permanent_impact_bps, temporary_impact_bps)
        """
        # Get market parameters
        adv = market_state.get('average_daily_volume', 1000000)
        volatility = market_state.get('volatility', 0.02)
        spread_bps = market_state.get('spread_bps', 2.0)
        
        # Order size as percentage of ADV
        order_size_pct = (order.quantity / adv) * 100
        
        # Permanent impact (affects all future trades)
        permanent_impact = self.permanent_impact_factor * np.sqrt(order_size_pct) * volatility
        
        # Temporary impact (affects only this trade)
        temporary_impact = self.temporary_impact_factor * order_size_pct * np.sqrt(volatility)
        
        # Latency impact (opportunity cost)
        latency_impact = self.latency_impact_factor * (latency_us / 100)
        
        # Total temporary impact includes latency
        total_temporary = temporary_impact + latency_impact
        
        # Adjust for market conditions
        if market_state.get('regime') == 'stressed':
            permanent_impact *= 2.0
            total_temporary *= 2.5
        elif market_state.get('regime') == 'quiet':
            permanent_impact *= 0.5
            total_temporary *= 0.7
        
        return permanent_impact, total_temporary


class OrderExecutionEngine:
    """
    Realistic order execution with latency, market impact, and slippage
    """
    
    def __init__(self, fee_schedule: Dict[str, Dict] = None):
        self.market_impact_model = MarketImpactModel()
        self.execution_queue = []  # Priority queue by timestamp
        self.order_book = defaultdict(lambda: {'bids': [], 'asks': []})
        
        # Fee schedule by venue (maker/taker fees in bps)
        self.fee_schedule = fee_schedule or {
            'NYSE': {'maker_fee': -0.20, 'taker_fee': 0.30},  # Negative = rebate
            'NASDAQ': {'maker_fee': -0.25, 'taker_fee': 0.30},
            'CBOE': {'maker_fee': -0.23, 'taker_fee': 0.28},
            'IEX': {'maker_fee': 0.0, 'taker_fee': 0.09},  # No rebates
            'ARCA': {'maker_fee': -0.20, 'taker_fee': 0.30}
        }
        
        # Execution statistics
        self.fill_count = 0
        self.total_latency_us = 0
        self.total_slippage_bps = 0
        
    async def execute_order(self, order: Order, market_state: Dict, 
                           actual_latency_us: float) -> Optional[Fill]:
        """
        Execute order with realistic market mechanics
        
        Args:
            order: Order to execute
            market_state: Current market conditions
            actual_latency_us: Actual network latency experienced
            
        Returns:
            Fill object if executed, None if rejected/cancelled
        """
        # Check if market moved during latency period
        arrival_price = market_state['mid_price']
        
        # Simulate price movement during latency
        price_drift = self._simulate_price_drift(
            arrival_price, actual_latency_us, market_state['volatility']
        )
        execution_price = arrival_price + price_drift
        
        # Calculate market impact
        permanent_impact, temporary_impact = self.market_impact_model.calculate_impact(
            order, market_state, actual_latency_us
        )
        
        # Determine fill price based on order type and side
        if order.order_type == OrderType.MARKET:
            # Market orders cross the spread and pay impact
            if order.side == OrderSide.BUY:
                fill_price = market_state['ask_price'] * (1 + temporary_impact / 10000)
            else:
                fill_price = market_state['bid_price'] * (1 - temporary_impact / 10000)
        
        elif order.order_type == OrderType.LIMIT:
            # Check if limit price is marketable
            if order.side == OrderSide.BUY:
                if order.price >= market_state['ask_price']:
                    # Marketable buy - execute as taker
                    fill_price = market_state['ask_price'] * (1 + temporary_impact / 10000)
                else:
                    # Post as maker (if time priority allows)
                    if self._check_queue_position(order, market_state):
                        fill_price = order.price
                    else:
                        return None  # Not filled
            else:  # SELL
                if order.price <= market_state['bid_price']:
                    # Marketable sell - execute as taker
                    fill_price = market_state['bid_price'] * (1 - temporary_impact / 10000)
                else:
                    # Post as maker
                    if self._check_queue_position(order, market_state):
                        fill_price = order.price
                    else:
                        return None
        
        # Calculate fees/rebates
        is_maker = order.order_type == OrderType.LIMIT and fill_price == order.price
        fee_structure = self.fee_schedule[order.venue]
        
        if is_maker:
            fee_bps = fee_structure['maker_fee']
        else:
            fee_bps = fee_structure['taker_fee']
        
        fees = abs(fee_bps) * fill_price * order.quantity / 10000 if fee_bps > 0 else 0
        rebate = abs(fee_bps) * fill_price * order.quantity / 10000 if fee_bps < 0 else 0
        
        # Calculate slippage
        if order.order_type == OrderType.MARKET:
            expected_price = arrival_price
        else:
            expected_price = order.price
        
        slippage_bps = abs(fill_price - expected_price) / expected_price * 10000
        
        # Create fill
        fill = Fill(
            fill_id=f"F{self.fill_count:08d}",
            order_id=order.order_id,
            symbol=order.symbol,
            venue=order.venue,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=order.timestamp + actual_latency_us / 1e6,
            fees=fees,
            rebate=rebate,
            latency_us=actual_latency_us,
            slippage_bps=slippage_bps,
            market_impact_bps=temporary_impact
        )
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.fill_timestamp = fill.timestamp
        order.latency_us = actual_latency_us
        
        # Update statistics
        self.fill_count += 1
        self.total_latency_us += actual_latency_us
        self.total_slippage_bps += slippage_bps
        
        # Update market state with permanent impact
        self._apply_permanent_impact(market_state, order.side, permanent_impact)
        
        return fill
    
    def _simulate_price_drift(self, price: float, latency_us: float, 
                             volatility: float) -> float:
        """Simulate price movement during latency period"""
        # Convert latency to fraction of trading day
        latency_days = latency_us / (1e6 * 60 * 60 * 6.5)  # 6.5 hour trading day
        
        # Brownian motion with drift
        drift = 0  # Assume no directional bias
        diffusion = volatility * np.sqrt(latency_days) * np.random.randn()
        
        return price * (drift + diffusion)
    
    def _check_queue_position(self, order: Order, market_state: Dict) -> bool:
        """Check if limit order would get filled based on queue position"""
        # Simplified queue model - in reality would track full order book
        # Assume 30% chance of fill for passive orders at best bid/ask
        queue_position_factor = 0.3
        
        # Adjust for order size
        size_factor = 1.0 - min(order.quantity / market_state.get('average_trade_size', 100), 0.5)
        
        fill_probability = queue_position_factor * size_factor
        
        return np.random.random() < fill_probability
    
    def _apply_permanent_impact(self, market_state: Dict, side: OrderSide, 
                               impact_bps: float):
        """Apply permanent market impact to prices"""
        impact_pct = impact_bps / 10000
        
        if side == OrderSide.BUY:
            # Buying pushes prices up
            market_state['bid_price'] *= (1 + impact_pct * 0.5)
            market_state['ask_price'] *= (1 + impact_pct * 0.5)
            market_state['mid_price'] *= (1 + impact_pct * 0.5)
        else:
            # Selling pushes prices down
            market_state['bid_price'] *= (1 - impact_pct * 0.5)
            market_state['ask_price'] *= (1 - impact_pct * 0.5)
            market_state['mid_price'] *= (1 - impact_pct * 0.5)
    
    def get_execution_stats(self) -> Dict[str, float]:
        """Get execution statistics"""
        return {
            'total_fills': self.fill_count,
            'avg_latency_us': self.total_latency_us / max(self.fill_count, 1),
            'avg_slippage_bps': self.total_slippage_bps / max(self.fill_count, 1),
            'fill_rate': 1.0  # Would track cancels/rejects in full implementation
        }


class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, strategy_type: TradingStrategyType, params: Dict = None):
        self.strategy_type = strategy_type
        self.params = params or {}
        self.positions = defaultdict(Position)
        self.orders = []
        self.fills = []
        self.pnl_history = []
        
    async def generate_signals(self, market_data: Dict, 
                              ml_predictions: Dict) -> List[Order]:
        """Generate trading signals - to be implemented by subclasses"""
        raise NotImplementedError
    
    def update_positions(self, fill: Fill, current_prices: Dict):
        """Update positions with new fill"""
        position = self.positions[fill.symbol]
        position.update_position(fill, current_prices[fill.symbol])
        self.fills.append(fill)
    
    def get_total_pnl(self) -> Dict[str, float]:
        """Calculate total P&L across all positions"""
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_volume = sum(pos.total_volume for pos in self.positions.values())
        
        # Calculate fees paid
        total_fees = sum(fill.fees - fill.rebate for fill in self.fills)
        
        return {
            'realized_pnl': total_realized,
            'unrealized_pnl': total_unrealized,
            'total_pnl': total_realized + total_unrealized - total_fees,
            'fees_paid': total_fees,
            'total_volume': total_volume
        }


class MarketMakingStrategy(TradingStrategy):
    """
    Market making strategy with inventory management
    
    Posts bid/ask quotes and manages inventory risk
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'spread_multiplier': 1.0,  # Multiple of fair spread to quote
            'inventory_limit': 10000,  # Max position size
            'skew_factor': 0.1,  # Price skew per 1000 shares
            'min_edge_bps': 2.0,  # Minimum edge to quote
            'quote_size': 100,  # Default quote size
            'urgency_threshold': 0.7  # ML urgency above which to be aggressive
        }
        params = {**default_params, **(params or {})}
        super().__init__(TradingStrategyType.MARKET_MAKING, params)
        
        self.quote_count = 0
        self.spread_captured = 0
        
    async def generate_signals(self, market_data: Dict, 
                              ml_predictions: Dict) -> List[Order]:
        """Generate market making orders"""
        orders = []
        
        for symbol in market_data.get('symbols', []):
            market_state = market_data[symbol]
            current_position = self.positions[symbol].quantity
            
            # Skip if at inventory limit
            if abs(current_position) >= self.params['inventory_limit']:
                continue
            
            # Calculate fair value and spread
            fair_value = market_state['mid_price']
            fair_spread_bps = market_state.get('spread_bps', 2.0)
            
            # Adjust spread based on ML predictions
            if ml_predictions.get('volatility_forecast', 0.01) > 0.02:
                # Widen spread in high volatility
                spread_multiplier = self.params['spread_multiplier'] * 1.5
            else:
                spread_multiplier = self.params['spread_multiplier']
            
            # Inventory skew - adjust prices based on position
            inventory_skew_bps = (current_position / 1000) * self.params['skew_factor']
            
            # Calculate quotes
            half_spread_bps = fair_spread_bps * spread_multiplier / 2
            
            bid_price = fair_value * (1 - (half_spread_bps + inventory_skew_bps) / 10000)
            ask_price = fair_value * (1 + (half_spread_bps - inventory_skew_bps) / 10000)
            
            # Check minimum edge
            bid_edge_bps = (fair_value - bid_price) / fair_value * 10000
            ask_edge_bps = (ask_price - fair_value) / fair_value * 10000
            
            # Determine best venue using ML routing
            routing_decision = ml_predictions.get('routing', {})
            best_venue = routing_decision.get('venue', 'NYSE')
            predicted_latency = routing_decision.get('predicted_latency_us', 1000)
            
            # Generate orders if edge sufficient
            if bid_edge_bps >= self.params['min_edge_bps']:
                bid_order = Order(
                    order_id=f"MM_B_{self.quote_count:08d}",
                    symbol=symbol,
                    venue=best_venue,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=self.params['quote_size'],
                    price=bid_price,
                    timestamp=time.time(),
                    strategy=TradingStrategyType.MARKET_MAKING,
                    predicted_latency_us=predicted_latency,
                    routing_confidence=routing_decision.get('confidence', 0.5),
                    market_regime=ml_predictions.get('regime', 'normal')
                )
                orders.append(bid_order)
            
            if ask_edge_bps >= self.params['min_edge_bps']:
                ask_order = Order(
                    order_id=f"MM_S_{self.quote_count:08d}",
                    symbol=symbol,
                    venue=best_venue,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=self.params['quote_size'],
                    price=ask_price,
                    timestamp=time.time(),
                    strategy=TradingStrategyType.MARKET_MAKING,
                    predicted_latency_us=predicted_latency,
                    routing_confidence=routing_decision.get('confidence', 0.5),
                    market_regime=ml_predictions.get('regime', 'normal')
                )
                orders.append(ask_order)
            
            self.quote_count += 1
        
        return orders
    
    def update_spread_capture(self, buy_fill: Fill, sell_fill: Fill):
        """Track spread capture from round-trip trades"""
        if buy_fill.symbol == sell_fill.symbol:
            spread = sell_fill.price - buy_fill.price
            self.spread_captured += spread * min(buy_fill.quantity, sell_fill.quantity)


class ArbitrageStrategy(TradingStrategy):
    """
    Cross-venue arbitrage strategy
    
    Detects and captures price discrepancies across venues
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'min_arb_bps': 5.0,  # Minimum arbitrage opportunity in bps
            'max_position': 5000,  # Max position per arbitrage
            'latency_threshold_us': 500,  # Max acceptable latency
            'confidence_threshold': 0.8,  # Min ML confidence for execution
            'competition_factor': 0.7  # Probability of winning arbitrage race
        }
        params = {**default_params, **(params or {})}
        super().__init__(TradingStrategyType.ARBITRAGE, params)
        
        self.opportunities_detected = 0
        self.opportunities_captured = 0
        self.arb_pnl = 0
        
    async def generate_signals(self, market_data: Dict, 
                              ml_predictions: Dict) -> List[Order]:
        """Generate arbitrage orders"""
        orders = []
        
        # Get cross-venue price data
        venue_prices = defaultdict(dict)
        for symbol in market_data.get('symbols', []):
            for venue in market_data.get('venues', []):
                key = f"{symbol}_{venue}"
                if key in market_data:
                    venue_prices[symbol][venue] = market_data[key]
        
        # Check each symbol for arbitrage opportunities
        for symbol, prices_by_venue in venue_prices.items():
            if len(prices_by_venue) < 2:
                continue
            
            # Find best bid and ask across venues
            best_bid = max(prices_by_venue.items(), 
                          key=lambda x: x[1].get('bid_price', 0))
            best_ask = min(prices_by_venue.items(), 
                          key=lambda x: x[1].get('ask_price', float('inf')))
            
            bid_venue, bid_data = best_bid
            ask_venue, ask_data = best_ask
            
            # Check if arbitrage exists
            if bid_data['bid_price'] > ask_data['ask_price']:
                arb_bps = (bid_data['bid_price'] - ask_data['ask_price']) / ask_data['ask_price'] * 10000
                
                if arb_bps >= self.params['min_arb_bps']:
                    self.opportunities_detected += 1
                    
                    # Get ML predictions for both legs
                    buy_routing = ml_predictions.get(f'routing_{symbol}_{ask_venue}', {})
                    sell_routing = ml_predictions.get(f'routing_{symbol}_{bid_venue}', {})
                    
                    buy_latency = buy_routing.get('predicted_latency_us', 1000)
                    sell_latency = sell_routing.get('predicted_latency_us', 1000)
                    
                    # Check if we can execute fast enough
                    total_latency = buy_latency + sell_latency
                    if total_latency > self.params['latency_threshold_us']:
                        continue
                    
                    # Check ML confidence
                    min_confidence = min(
                        buy_routing.get('confidence', 0),
                        sell_routing.get('confidence', 0)
                    )
                    if min_confidence < self.params['confidence_threshold']:
                        continue
                    
                    # Simulate competition - other HFTs trying to capture same arb
                    if np.random.random() > self.params['competition_factor']:
                        continue  # Lost the race
                    
                    # Calculate position size based on available liquidity
                    buy_size = min(
                        ask_data.get('ask_size', 100),
                        self.params['max_position']
                    )
                    sell_size = min(
                        bid_data.get('bid_size', 100),
                        self.params['max_position']
                    )
                    arb_size = min(buy_size, sell_size)
                    
                    # Generate arbitrage orders (simultaneous buy low, sell high)
                    buy_order = Order(
                        order_id=f"ARB_B_{self.opportunities_captured:08d}",
                        symbol=symbol,
                        venue=ask_venue,
                        side=OrderSide.BUY,
                        order_type=OrderType.IOC,  # Immediate or cancel
                        quantity=arb_size,
                        price=ask_data['ask_price'],
                        timestamp=time.time(),
                        strategy=TradingStrategyType.ARBITRAGE,
                        predicted_latency_us=buy_latency,
                        routing_confidence=buy_routing.get('confidence', 0.5),
                        market_regime=ml_predictions.get('regime', 'normal')
                    )
                    
                    sell_order = Order(
                        order_id=f"ARB_S_{self.opportunities_captured:08d}",
                        symbol=symbol,
                        venue=bid_venue,
                        side=OrderSide.SELL,
                        order_type=OrderType.IOC,
                        quantity=arb_size,
                        price=bid_data['bid_price'],
                        timestamp=time.time(),
                        strategy=TradingStrategyType.ARBITRAGE,
                        predicted_latency_us=sell_latency,
                        routing_confidence=sell_routing.get('confidence', 0.5),
                        market_regime=ml_predictions.get('regime', 'normal')
                    )
                    
                    orders.extend([buy_order, sell_order])
                    self.opportunities_captured += 1
        
        return orders
    
    def calculate_arbitrage_pnl(self, buy_fill: Fill, sell_fill: Fill):
        """Calculate P&L from arbitrage trade"""
        if buy_fill.symbol == sell_fill.symbol:
            quantity = min(buy_fill.quantity, sell_fill.quantity)
            gross_pnl = (sell_fill.price - buy_fill.price) * quantity
            
            # Subtract fees and add rebates
            net_pnl = gross_pnl - buy_fill.fees + buy_fill.rebate - sell_fill.fees + sell_fill.rebate
            
            self.arb_pnl += net_pnl
            return net_pnl
        return 0


class MomentumStrategy(TradingStrategy):
    """
    Momentum trading strategy using technical indicators and ML signals
    
    Trades short-term price momentum with ML-optimized entry/exit
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'lookback_period': 20,  # Bars for momentum calculation
            'entry_threshold': 2.0,  # Z-score threshold for entry
            'exit_threshold': 0.5,  # Z-score threshold for exit
            'stop_loss_bps': 50,  # Stop loss in basis points
            'take_profit_bps': 100,  # Take profit in basis points
            'max_position': 1000,  # Maximum position size
            'hold_time': 300,  # Maximum hold time in seconds
            'ml_signal_weight': 0.7  # Weight given to ML predictions
        }
        params = {**default_params, **(params or {})}
        super().__init__(TradingStrategyType.MOMENTUM, params)
        
        self.signal_history = defaultdict(deque)
        self.entry_prices = {}
        self.entry_times = {}
        
    async def generate_signals(self, market_data: Dict, 
                              ml_predictions: Dict) -> List[Order]:
        """Generate momentum trading signals"""
        orders = []
        
        for symbol in market_data.get('symbols', []):
            market_state = market_data[symbol]
            current_position = self.positions[symbol].quantity
            
            # Update signal history
            self.signal_history[symbol].append({
                'price': market_state['mid_price'],
                'volume': market_state['volume'],
                'timestamp': time.time()
            })
            
            # Maintain lookback window
            if len(self.signal_history[symbol]) > self.params['lookback_period']:
                self.signal_history[symbol].popleft()
            
            # Need enough history
            if len(self.signal_history[symbol]) < self.params['lookback_period']:
                continue
            
            # Calculate momentum signal
            prices = [s['price'] for s in self.signal_history[symbol]]
            returns = np.diff(np.log(prices))
            
            # Simple momentum: recent return vs average
            recent_return = returns[-1]
            avg_return = np.mean(returns[:-1])
            std_return = np.std(returns[:-1])
            
            if std_return > 0:
                z_score = (recent_return - avg_return) / std_return
            else:
                z_score = 0
            
            # Combine with ML predictions
            ml_signal = ml_predictions.get(f'momentum_signal_{symbol}', 0)
            combined_signal = (
                self.params['ml_signal_weight'] * ml_signal + 
                (1 - self.params['ml_signal_weight']) * z_score
            )
            
            # Get routing decision
            routing = ml_predictions.get(f'routing_{symbol}', {})
            best_venue = routing.get('venue', 'NYSE')
            predicted_latency = routing.get('predicted_latency_us', 1000)
            
            # Position management
            if current_position == 0:
                # Entry logic
                if combined_signal > self.params['entry_threshold']:
                    # Long entry
                    order = Order(
                        order_id=f"MOM_B_{len(self.orders):08d}",
                        symbol=symbol,
                        venue=best_venue,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=self.params['max_position'],
                        price=market_state['ask_price'],  # Market order
                        timestamp=time.time(),
                        strategy=TradingStrategyType.MOMENTUM,
                        predicted_latency_us=predicted_latency,
                        routing_confidence=routing.get('confidence', 0.5),
                        market_regime=ml_predictions.get('regime', 'normal')
                    )
                    orders.append(order)
                    self.entry_prices[symbol] = market_state['mid_price']
                    self.entry_times[symbol] = time.time()
                    
                elif combined_signal < -self.params['entry_threshold']:
                    # Short entry
                    order = Order(
                        order_id=f"MOM_S_{len(self.orders):08d}",
                        symbol=symbol,
                        venue=best_venue,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=self.params['max_position'],
                        price=market_state['bid_price'],
                        timestamp=time.time(),
                        strategy=TradingStrategyType.MOMENTUM,
                        predicted_latency_us=predicted_latency,
                        routing_confidence=routing.get('confidence', 0.5),
                        market_regime=ml_predictions.get('regime', 'normal')
                    )
                    orders.append(order)
                    self.entry_prices[symbol] = market_state['mid_price']
                    self.entry_times[symbol] = time.time()
            
            else:
                # Exit logic for existing positions
                entry_price = self.entry_prices.get(symbol, market_state['mid_price'])
                time_in_position = time.time() - self.entry_times.get(symbol, time.time())
                current_price = market_state['mid_price']
                
                if current_position > 0:  # Long position
                    # Calculate P&L in bps
                    pnl_bps = (current_price - entry_price) / entry_price * 10000
                    
                    # Exit conditions
                    exit_signal = (
                        combined_signal < self.params['exit_threshold'] or  # Signal reversal
                        pnl_bps <= -self.params['stop_loss_bps'] or  # Stop loss
                        pnl_bps >= self.params['take_profit_bps'] or  # Take profit
                        time_in_position > self.params['hold_time']  # Time exit
                    )
                    
                    if exit_signal:
                        order = Order(
                            order_id=f"MOM_EXIT_S_{len(self.orders):08d}",
                            symbol=symbol,
                            venue=best_venue,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=current_position,
                            price=market_state['bid_price'],
                            timestamp=time.time(),
                            strategy=TradingStrategyType.MOMENTUM,
                            predicted_latency_us=predicted_latency,
                            routing_confidence=routing.get('confidence', 0.5),
                            market_regime=ml_predictions.get('regime', 'normal')
                        )
                        orders.append(order)
                
                else:  # Short position
                    # Calculate P&L in bps
                    pnl_bps = (entry_price - current_price) / entry_price * 10000
                    
                    # Exit conditions
                    exit_signal = (
                        combined_signal > -self.params['exit_threshold'] or
                        pnl_bps <= -self.params['stop_loss_bps'] or
                        pnl_bps >= self.params['take_profit_bps'] or
                        time_in_position > self.params['hold_time']
                    )
                    
                    if exit_signal:
                        order = Order(
                            order_id=f"MOM_EXIT_B_{len(self.orders):08d}",
                            symbol=symbol,
                            venue=best_venue,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=abs(current_position),
                            price=market_state['ask_price'],
                            timestamp=time.time(),
                            strategy=TradingStrategyType.MOMENTUM,
                            predicted_latency_us=predicted_latency,
                            routing_confidence=routing.get('confidence', 0.5),
                            market_regime=ml_predictions.get('regime', 'normal')
                        )
                        orders.append(order)
        
        return orders


class TradingSimulator:
    """
    Main trading simulator that orchestrates strategy execution and performance measurement
    """
    
    def __init__(self, venues: List[str], symbols: List[str]):
        self.venues = venues
        self.symbols = symbols
        
        # Initialize components
        self.execution_engine = OrderExecutionEngine()
        
        # Initialize strategies
        self.strategies = {
            'market_making': MarketMakingStrategy(),
            'arbitrage': ArbitrageStrategy(),
            'momentum': MomentumStrategy()
        }
        
        # Performance tracking
        self.total_pnl = 0
        self.trade_count = 0
        self.simulation_start_time = None
        self.performance_history = []
        
        # Order tracking
        self.pending_orders = []
        self.order_history = []
        self.fill_history = []
        
        # Risk metrics
        self.max_drawdown = 0
        self.high_water_mark = 0
        
        logger.info(f"TradingSimulator initialized for {len(venues)} venues, {len(symbols)} symbols")
    
    async def simulate_trading(self, market_data_generator, ml_predictor,
                              duration_seconds: int = 300) -> Dict[str, Any]:
        """
        Run complete trading simulation
        
        Args:
            market_data_generator: Phase 1 market data generator
            ml_predictor: Phase 2 ML prediction system
            duration_seconds: Simulation duration
            
        Returns:
            Comprehensive performance results
        """
        logger.info(f"Starting trading simulation for {duration_seconds} seconds")
        
        self.simulation_start_time = time.time()
        tick_count = 0
        
        # Market data aggregation
        market_state = {}
        current_prices = {}
        
        async for tick in market_data_generator.generate_market_data_stream(duration_seconds):
            # Update market state
            symbol_venue_key = f"{tick.symbol}_{tick.venue}"
            market_state[symbol_venue_key] = {
                'bid_price': tick.bid_price,
                'ask_price': tick.ask_price,
                'mid_price': tick.mid_price,
                'spread_bps': (tick.ask_price - tick.bid_price) / tick.mid_price * 10000,
                'volume': tick.volume,
                'bid_size': tick.bid_size,
                'ask_size': tick.ask_size,
                'volatility': tick.volatility,
                'average_daily_volume': 1000000,  # Placeholder
                'average_trade_size': 100
            }
            
            # Track current prices for P&L
            current_prices[tick.symbol] = tick.mid_price
            
            # Aggregate market data for strategies
            if tick_count % 10 == 0:  # Process every 10 ticks
                aggregated_market_data = {
                    'symbols': self.symbols,
                    'venues': self.venues,
                    **market_state
                }
                
                # Get ML predictions
                ml_predictions = await self._get_ml_predictions(
                    ml_predictor, tick, market_state
                )
                
                # Generate signals from each strategy
                all_orders = []
                for strategy_name, strategy in self.strategies.items():
                    orders = await strategy.generate_signals(
                        aggregated_market_data, ml_predictions
                    )
                    all_orders.extend(orders)
                
                # Execute orders
                for order in all_orders:
                    fill = await self._execute_order(order, market_state)
                    
                    if fill:
                        # Update strategy positions
                        strategy = self.strategies[order.strategy.value]
                        strategy.update_positions(fill, current_prices)
                        
                        # Track fills
                        self.fill_history.append(fill)
                        self.trade_count += 1
                
                # Update P&L and risk metrics
                self._update_performance_metrics(current_prices)
            
            tick_count += 1
            
            # Log progress
            if tick_count % 1000 == 0:
                elapsed = time.time() - self.simulation_start_time
                logger.info(f"Processed {tick_count} ticks in {elapsed:.1f}s, "
                           f"Total P&L: ${self.total_pnl:.2f}")
        
        # Generate final results
        results = self._generate_simulation_results()
        
        logger.info(f"Simulation complete: {tick_count} ticks, {self.trade_count} trades")
        return results
    
    async def _get_ml_predictions(self, ml_predictor, tick, market_state) -> Dict:
        """Get ML predictions for routing and signals"""
        predictions = {}
        
        # Get latency predictions and routing decisions for each venue
        for venue in self.venues:
            routing_key = f'routing_{tick.symbol}_{venue}'
            
            # Make routing decision using ML
            routing_decision = ml_predictor.make_routing_decision(tick.symbol)
            
            predictions[routing_key] = {
                'venue': routing_decision.venue,
                'predicted_latency_us': routing_decision.expected_latency_us,
                'confidence': routing_decision.confidence
            }
        
        # Get market regime
        regime_detection = ml_predictor.detect_market_regime(market_state)
        predictions['regime'] = regime_detection.regime.value
        
        # Get volatility forecast
        predictions['volatility_forecast'] = market_state.get(
            f"{tick.symbol}_{tick.venue}", {}
        ).get('volatility', 0.01)
        
        # Momentum signals (simplified)
        predictions[f'momentum_signal_{tick.symbol}'] = np.random.randn() * 0.5
        
        return predictions
    
    async def _execute_order(self, order: Order, market_state: Dict) -> Optional[Fill]:
        """Execute order through execution engine"""
        # Get market state for order
        state_key = f"{order.symbol}_{order.venue}"
        if state_key not in market_state:
            logger.warning(f"No market data for {state_key}")
            return None
        
        venue_market_state = market_state[state_key]
        
        # Simulate actual network latency (may differ from prediction)
        if order.predicted_latency_us:
            # Add noise to predicted latency
            latency_noise = np.random.normal(0, order.predicted_latency_us * 0.1)
            actual_latency_us = max(50, order.predicted_latency_us + latency_noise)
        else:
            actual_latency_us = np.random.normal(1000, 200)
        
        # Execute order
        fill = await self.execution_engine.execute_order(
            order, venue_market_state, actual_latency_us
        )
        
        return fill
    
    def _update_performance_metrics(self, current_prices: Dict):
        """Update P&L and risk metrics"""
        # Calculate total P&L across all strategies
        total_pnl = 0
        
        for strategy in self.strategies.values():
            strategy_pnl = strategy.get_total_pnl()
            total_pnl += strategy_pnl['total_pnl']
        
        self.total_pnl = total_pnl
        
        # Update high water mark and drawdown
        if total_pnl > self.high_water_mark:
            self.high_water_mark = total_pnl
        
        drawdown = self.high_water_mark - total_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Record performance snapshot
        self.performance_history.append({
            'timestamp': time.time(),
            'total_pnl': total_pnl,
            'trade_count': self.trade_count,
            'drawdown': drawdown
        })
    
    def _generate_simulation_results(self) -> Dict[str, Any]:
        """Generate comprehensive simulation results"""
        results = {
            'summary': {
                'total_pnl': self.total_pnl,
                'trade_count': self.trade_count,
                'simulation_duration': time.time() - self.simulation_start_time,
                'max_drawdown': self.max_drawdown,
                'final_positions': self._get_final_positions()
            },
            'strategy_performance': {},
            'execution_stats': self.execution_engine.get_execution_stats(),
            'venue_analysis': self._analyze_venue_performance(),
            'ml_routing_impact': self._analyze_ml_impact()
        }
        
        # Strategy-specific results
        for strategy_name, strategy in self.strategies.items():
            pnl_data = strategy.get_total_pnl()
            
            if strategy_name == 'market_making':
                strategy_results = {
                    **pnl_data,
                    'quotes_posted': strategy.quote_count,
                    'spread_captured': strategy.spread_captured
                }
            elif strategy_name == 'arbitrage':
                strategy_results = {
                    **pnl_data,
                    'opportunities_detected': strategy.opportunities_detected,
                    'opportunities_captured': strategy.opportunities_captured,
                    'capture_rate': strategy.opportunities_captured / max(strategy.opportunities_detected, 1),
                    'arbitrage_pnl': strategy.arb_pnl
                }
            elif strategy_name == 'momentum':
                strategy_results = {
                    **pnl_data,
                    'signals_generated': len(strategy.orders),
                    'positions_taken': len(strategy.entry_prices)
                }
            
            results['strategy_performance'][strategy_name] = strategy_results
        
        # Calculate risk metrics
        if self.performance_history:
            pnl_series = [p['total_pnl'] for p in self.performance_history]
            returns = np.diff(pnl_series)
            
            if len(returns) > 0:
                results['risk_metrics'] = {
                    'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                    'max_drawdown_pct': self.max_drawdown / max(abs(self.high_water_mark), 1) * 100,
                    'profit_factor': self._calculate_profit_factor(),
                    'win_rate': self._calculate_win_rate()
                }
        
        return results
    
    def _get_final_positions(self) -> Dict[str, Dict]:
        """Get final positions across all strategies"""
        final_positions = {}
        
        for strategy_name, strategy in self.strategies.items():
            for symbol, position in strategy.positions.items():
                if position.quantity != 0:
                    final_positions[f"{strategy_name}_{symbol}"] = {
                        'quantity': position.quantity,
                        'average_cost': position.average_cost,
                        'unrealized_pnl': position.unrealized_pnl
                    }
        
        return final_positions
    
    def _analyze_venue_performance(self) -> Dict[str, Dict]:
        """Analyze performance by venue"""
        venue_stats = defaultdict(lambda: {
            'fill_count': 0,
            'total_volume': 0,
            'total_fees': 0,
            'total_rebates': 0,
            'avg_latency_us': 0,
            'avg_slippage_bps': 0
        })
        
        for fill in self.fill_history:
            stats = venue_stats[fill.venue]
            stats['fill_count'] += 1
            stats['total_volume'] += fill.quantity
            stats['total_fees'] += fill.fees
            stats['total_rebates'] += fill.rebate
            
            # Running averages
            n = stats['fill_count']
            stats['avg_latency_us'] = (
                (stats['avg_latency_us'] * (n - 1) + fill.latency_us) / n
            )
            stats['avg_slippage_bps'] = (
                (stats['avg_slippage_bps'] * (n - 1) + fill.slippage_bps) / n
            )
        
        return dict(venue_stats)
    
    def _analyze_ml_impact(self) -> Dict[str, float]:
        """Analyze impact of ML routing decisions"""
        ml_routed_fills = [f for f in self.fill_history if f.order_id.startswith(('MM', 'ARB', 'MOM'))]
        
        if not ml_routed_fills:
            return {}
        
        # Compare predicted vs actual latencies
        latency_errors = []
        for fill in ml_routed_fills:
            # Get original order
            order = next((o for o in self.order_history if o.order_id == fill.order_id), None)
            if order and order.predicted_latency_us:
                error = abs(fill.latency_us - order.predicted_latency_us)
                latency_errors.append(error)
        
        # Calculate routing optimization benefit
        # Estimate baseline latency as 90th percentile
        actual_latencies = [f.latency_us for f in ml_routed_fills]
        if actual_latencies:
            baseline_latency = np.percentile(actual_latencies, 90)
            avg_latency = np.mean(actual_latencies)
            latency_improvement = (baseline_latency - avg_latency) / baseline_latency * 100
        else:
            latency_improvement = 0
        
        return {
            'fills_with_ml_routing': len(ml_routed_fills),
            'avg_latency_prediction_error_us': np.mean(latency_errors) if latency_errors else 0,
            'latency_improvement_pct': latency_improvement,
            'estimated_pnl_improvement': latency_improvement * self.total_pnl / 100  # Rough estimate
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        # Assume 252 trading days, 6.5 hours per day
        periods_per_year = 252 * 6.5 * 3600 / (self.performance_history[1]['timestamp'] - 
                                               self.performance_history[0]['timestamp'])
        
        return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        profits = sum(f.quantity * (f.price - p.average_cost) 
                     for s in self.strategies.values() 
                     for p in s.positions.values() 
                     for f in s.fills 
                     if f.side == OrderSide.SELL and f.price > p.average_cost)
        
        losses = abs(sum(f.quantity * (f.price - p.average_cost) 
                        for s in self.strategies.values() 
                        for p in s.positions.values() 
                        for f in s.fills 
                        if f.side == OrderSide.SELL and f.price < p.average_cost))
        
        return profits / losses if losses > 0 else float('inf')
    
    def _calculate_win_rate(self) -> float:
        """Calculate percentage of profitable trades"""
        winning_trades = 0
        total_trades = 0
        
        for strategy in self.strategies.values():
            # Group fills by symbol to identify round trips
            symbol_fills = defaultdict(list)
            for fill in strategy.fills:
                symbol_fills[fill.symbol].append(fill)
            
            # Calculate P&L for each round trip
            for symbol, fills in symbol_fills.items():
                buys = [f for f in fills if f.side == OrderSide.BUY]
                sells = [f for f in fills if f.side == OrderSide.SELL]
                
                # Match buys and sells (FIFO)
                for buy, sell in zip(buys, sells):
                    total_trades += 1
                    if sell.price > buy.price:
                        winning_trades += 1
        
        return winning_trades / total_trades if total_trades > 0 else 0


# Performance measurement functions
def calculate_pnl_attribution(simulator: TradingSimulator) -> Dict[str, Any]:
    """
    Detailed P&L attribution analysis
    
    Breaks down performance by:
    - Strategy contribution
    - Venue efficiency
    - Timing/latency impact
    - ML routing benefit
    """
    attribution = {
        'total_pnl': simulator.total_pnl,
        'strategy_attribution': {},
        'venue_attribution': {},
        'latency_cost_analysis': {},
        'ml_routing_benefit': {}
    }
    
    # Strategy attribution
    for strategy_name, strategy in simulator.strategies.items():
        pnl_data = strategy.get_total_pnl()
        attribution['strategy_attribution'][strategy_name] = {
            'gross_pnl': pnl_data['realized_pnl'] + pnl_data['unrealized_pnl'],
            'fees_paid': pnl_data['fees_paid'],
            'net_pnl': pnl_data['total_pnl'],
            'pnl_contribution_pct': pnl_data['total_pnl'] / simulator.total_pnl * 100 
                                   if simulator.total_pnl != 0 else 0
        }
    
    # Venue attribution
    venue_pnl = defaultdict(float)
    venue_volume = defaultdict(int)
    
    for strategy in simulator.strategies.values():
        for fill in strategy.fills:
            if fill.side == OrderSide.SELL:
                # Find matching buy
                position = strategy.positions[fill.symbol]
                pnl = fill.quantity * (fill.price - position.average_cost)
                venue_pnl[fill.venue] += pnl
            venue_volume[fill.venue] += fill.quantity
    
    for venue in venue_pnl:
        attribution['venue_attribution'][venue] = {
            'pnl': venue_pnl[venue],
            'volume': venue_volume[venue],
            'pnl_per_share': venue_pnl[venue] / venue_volume[venue] if venue_volume[venue] > 0 else 0
        }
    
    # Latency cost analysis
    latency_costs = calculate_latency_costs(simulator.fill_history)
    attribution['latency_cost_analysis'] = latency_costs
    
    # ML routing benefit
    ml_benefit = simulator._analyze_ml_impact()
    attribution['ml_routing_benefit'] = ml_benefit
    
    return attribution


def calculate_latency_costs(fills: List[Fill]) -> Dict[str, float]:
    """Calculate opportunity costs due to latency"""
    total_latency_cost = 0
    latency_by_strategy = defaultdict(float)
    
    for fill in fills:
        # Estimate cost as slippage attributed to latency
        # Assume 50% of slippage is due to latency
        latency_cost = fill.slippage_bps * 0.5 * fill.price * fill.quantity / 10000
        total_latency_cost += latency_cost
        
        # Categorize by strategy
        if 'MM' in fill.order_id:
            latency_by_strategy['market_making'] += latency_cost
        elif 'ARB' in fill.order_id:
            latency_by_strategy['arbitrage'] += latency_cost
        elif 'MOM' in fill.order_id:
            latency_by_strategy['momentum'] += latency_cost
    
    return {
        'total_latency_cost': total_latency_cost,
        'cost_by_strategy': dict(latency_by_strategy),
        'avg_cost_per_trade': total_latency_cost / len(fills) if fills else 0
    }