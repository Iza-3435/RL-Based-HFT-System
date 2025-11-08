import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from collections import defaultdict, deque
from enum import Enum
import bisect
import json

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides"""
    BID = "bid"
    ASK = "ask"

@dataclass
class Order:
    """Individual order in the book"""
    order_id: str
    symbol: str
    venue: str
    side: OrderSide
    price: float
    size: int
    timestamp: float
    order_type: OrderType = OrderType.LIMIT
    remaining_size: int = field(init=False)
    
    def __post_init__(self):
        self.remaining_size = self.size
    
    @property
    def is_filled(self) -> bool:
        return self.remaining_size <= 0

@dataclass
class PriceLevel:
    """Price level in order book"""
    price: float
    total_size: int
    order_count: int
    orders: List[Order] = field(default_factory=list)
    
    def add_order(self, order: Order):
        """Add order to this price level"""
        self.orders.append(order)
        self.total_size += order.remaining_size
        self.order_count += 1
    
    def remove_order(self, order_id: str) -> bool:
        """Remove order from this price level"""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                self.total_size -= order.remaining_size
                self.order_count -= 1
                del self.orders[i]
                return True
        return False
    
    def modify_order(self, order_id: str, new_size: int) -> bool:
        """Modify order size at this price level"""
        for order in self.orders:
            if order.order_id == order_id:
                size_diff = new_size - order.remaining_size
                order.remaining_size = new_size
                self.total_size += size_diff
                return True
        return False

@dataclass
class BookSnapshot:
    """Complete order book snapshot"""
    symbol: str
    venue: str
    timestamp: float
    sequence_number: int
    bids: List[Tuple[float, int, int]]  # (price, size, order_count)
    asks: List[Tuple[float, int, int]]  # (price, size, order_count)
    
    @property
    def best_bid(self) -> Optional[Tuple[float, int]]:
        return (self.bids[0][0], self.bids[0][1]) if self.bids else None
    
    @property
    def best_ask(self) -> Optional[Tuple[float, int]]:
        return (self.asks[0][0], self.asks[0][1]) if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask[0] - self.best_bid[0]
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid[0] + self.best_ask[0]) / 2
        return None

@dataclass
class BookUpdate:
    """Order book update event"""
    symbol: str
    venue: str
    timestamp: float
    sequence_number: int
    update_type: str  # "add", "modify", "delete", "trade"
    side: OrderSide
    price: float
    size: int
    order_id: Optional[str] = None

class OrderBook:
    """
    High-performance order book implementation with microsecond precision.
    
    Features:
    - Efficient price-time priority matching
    - Real-time book depth calculation
    - Order tracking and modification
    - Market impact modeling
    - Book imbalance metrics
    """
    
    def __init__(self, symbol: str, venue: str):
        self.symbol = symbol
        self.venue = venue
        
        # Price levels stored as sorted lists for efficient access
        self.bids: List[PriceLevel] = []  # Descending price order
        self.asks: List[PriceLevel] = []  # Ascending price order
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.price_to_level: Dict[Tuple[OrderSide, float], PriceLevel] = {}
        
        # Book state
        self.sequence_number = 0
        self.last_update_time = 0.0
        self.total_volume = 0
        
        # Performance metrics
        self.update_count = 0
        self.trade_count = 0
        
        # Book depth tracking
        self.max_depth_levels = 50  # Track top 50 levels each side
        
        logger.debug(f"OrderBook initialized for {symbol}@{venue}")
    
    def _find_price_level_index(self, side: OrderSide, price: float) -> int:
        """Find insertion index for price level using binary search"""
        levels = self.bids if side == OrderSide.BID else self.asks
        
        if side == OrderSide.BID:
            # Bids: descending price order
            return bisect.bisect_left([level.price for level in levels], price, 
                                    key=lambda x: -x)
        else:
            # Asks: ascending price order
            return bisect.bisect_left([level.price for level in levels], price)
    
    def _get_or_create_price_level(self, side: OrderSide, price: float) -> PriceLevel:
        """Get existing price level or create new one"""
        key = (side, price)
        
        if key in self.price_to_level:
            return self.price_to_level[key]
        
        # Create new price level
        level = PriceLevel(price=price, total_size=0, order_count=0)
        self.price_to_level[key] = level
        
        # Insert at correct position
        levels = self.bids if side == OrderSide.BID else self.asks
        index = self._find_price_level_index(side, price)
        levels.insert(index, level)
        
        return level
    
    def _remove_price_level_if_empty(self, side: OrderSide, price: float):
        """Remove price level if it has no orders"""
        key = (side, price)
        
        if key not in self.price_to_level:
            return
        
        level = self.price_to_level[key]
        if level.total_size <= 0:
            levels = self.bids if side == OrderSide.BID else self.asks
            levels.remove(level)
            del self.price_to_level[key]
    
    def add_order(self, order: Order) -> bool:
        """Add new order to the book"""
        if order.order_id in self.orders:
            logger.warning(f"Order {order.order_id} already exists")
            return False
        
        # Add to price level
        level = self._get_or_create_price_level(order.side, order.price)
        level.add_order(order)
        
        # Track order
        self.orders[order.order_id] = order
        
        # Update book state
        self.sequence_number += 1
        self.last_update_time = order.timestamp
        self.update_count += 1
        
        return True
    
    def modify_order(self, order_id: str, new_size: int, timestamp: float) -> bool:
        """Modify existing order size"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found for modification")
            return False
        
        order = self.orders[order_id]
        old_size = order.remaining_size
        
        # Update at price level
        key = (order.side, order.price)
        if key in self.price_to_level:
            level = self.price_to_level[key]
            level.modify_order(order_id, new_size)
            
            # Remove level if empty
            if new_size <= 0:
                self._remove_price_level_if_empty(order.side, order.price)
        
        # Update book state
        self.sequence_number += 1
        self.last_update_time = timestamp
        self.update_count += 1
        
        return True
    
    def cancel_order(self, order_id: str, timestamp: float) -> bool:
        """Cancel and remove order from book"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found for cancellation")
            return False
        
        order = self.orders[order_id]
        
        # Remove from price level
        key = (order.side, order.price)
        if key in self.price_to_level:
            level = self.price_to_level[key]
            level.remove_order(order_id)
            self._remove_price_level_if_empty(order.side, order.price)
        
        # Remove from tracking
        del self.orders[order_id]
        
        # Update book state
        self.sequence_number += 1
        self.last_update_time = timestamp
        self.update_count += 1
        
        return True
    
    def execute_trade(self, price: float, size: int, timestamp: float) -> List[str]:
        """Execute trade and update affected orders"""
        executed_orders = []
        remaining_size = size
        
        # Determine which side is being hit
        if self.bids and price <= self.bids[0].price:
            # Sell order hitting bids
            side_to_hit = OrderSide.BID
            levels = self.bids
        elif self.asks and price >= self.asks[0].price:
            # Buy order hitting asks
            side_to_hit = OrderSide.ASK
            levels = self.asks
        else:
            logger.warning(f"Trade price {price} doesn't match any book levels")
            return executed_orders
        
        # Execute against orders at price levels
        levels_to_remove = []
        
        for level in levels:
            if remaining_size <= 0:
                break
            
            if ((side_to_hit == OrderSide.BID and level.price < price) or
                (side_to_hit == OrderSide.ASK and level.price > price)):
                break
            
            # Execute against orders at this level
            orders_to_remove = []
            for order in level.orders:
                if remaining_size <= 0:
                    break
                
                execution_size = min(remaining_size, order.remaining_size)
                order.remaining_size -= execution_size
                remaining_size -= execution_size
                level.total_size -= execution_size
                
                executed_orders.append(order.order_id)
                
                if order.is_filled:
                    orders_to_remove.append(order.order_id)
            
            # Remove filled orders
            for order_id in orders_to_remove:
                level.remove_order(order_id)
                if order_id in self.orders:
                    del self.orders[order_id]
            
            # Mark empty levels for removal
            if level.total_size <= 0:
                levels_to_remove.append(level)
        
        # Remove empty price levels
        for level in levels_to_remove:
            key = (side_to_hit, level.price)
            if key in self.price_to_level:
                del self.price_to_level[key]
            levels.remove(level)
        
        # Update book state
        self.sequence_number += 1
        self.last_update_time = timestamp
        self.trade_count += 1
        self.total_volume += (size - remaining_size)
        
        return executed_orders
    
    def get_best_bid_ask(self) -> Tuple[Optional[Tuple[float, int]], Optional[Tuple[float, int]]]:
        """Get best bid and ask prices with sizes"""
        best_bid = None
        best_ask = None
        
        if self.bids:
            level = self.bids[0]
            best_bid = (level.price, level.total_size)
        
        if self.asks:
            level = self.asks[0]
            best_ask = (level.price, level.total_size)
        
        return best_bid, best_ask
    
    def get_book_depth(self, levels: int = 10) -> Tuple[List[Tuple[float, int, int]], List[Tuple[float, int, int]]]:
        """Get order book depth for specified number of levels"""
        bid_depth = []
        ask_depth = []
        
        # Get bid depth (descending price)
        for i, level in enumerate(self.bids[:levels]):
            bid_depth.append((level.price, level.total_size, level.order_count))
        
        # Get ask depth (ascending price)
        for i, level in enumerate(self.asks[:levels]):
            ask_depth.append((level.price, level.total_size, level.order_count))
        
        return bid_depth, ask_depth
    
    def calculate_book_imbalance(self, depth_levels: int = 5) -> float:
        """Calculate order book imbalance ratio"""
        bid_depth, ask_depth = self.get_book_depth(depth_levels)
        
        total_bid_size = sum(size for _, size, _ in bid_depth)
        total_ask_size = sum(size for _, size, _ in ask_depth)
        
        if total_bid_size + total_ask_size == 0:
            return 0.0
        
        return (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
    
    def estimate_market_impact(self, side: OrderSide, size: int) -> Dict[str, float]:
        """Estimate market impact of a market order"""
        if side == OrderSide.BID:
            # Buying - walk up the ask side
            levels = self.asks
        else:
            # Selling - walk down the bid side
            levels = self.bids
        
        if not levels:
            return {'average_price': 0.0, 'worst_price': 0.0, 'levels_consumed': 0}
        
        remaining_size = size
        total_cost = 0.0
        levels_consumed = 0
        worst_price = levels[0].price
        
        for level in levels:
            if remaining_size <= 0:
                break
            
            consumption = min(remaining_size, level.total_size)
            total_cost += consumption * level.price
            remaining_size -= consumption
            levels_consumed += 1
            worst_price = level.price
            
            if remaining_size <= 0:
                break
        
        executed_size = size - remaining_size
        average_price = total_cost / executed_size if executed_size > 0 else 0.0
        
        return {
            'average_price': average_price,
            'worst_price': worst_price,
            'levels_consumed': levels_consumed,
            'executed_size': executed_size,
            'slippage': worst_price - levels[0].price if levels else 0.0
        }
    
    def get_snapshot(self) -> BookSnapshot:
        """Get complete book snapshot"""
        bid_depth, ask_depth = self.get_book_depth(self.max_depth_levels)
        
        return BookSnapshot(
            symbol=self.symbol,
            venue=self.venue,
            timestamp=self.last_update_time,
            sequence_number=self.sequence_number,
            bids=bid_depth,
            asks=ask_depth
        )


class OrderBookManager:
    """
    Manages multiple order books across venues with advanced analytics.
    
    Features:
    - Multi-venue order book synchronization
    - Cross-venue arbitrage detection
    - Liquidity aggregation
    - Real-time book analytics
    - Performance monitoring
    """
    
    def __init__(self, symbols: List[str], venues: List[str]):
        self.symbols = symbols
        self.venues = venues
        
        # Order books indexed by (symbol, venue)
        self.order_books: Dict[Tuple[str, str], OrderBook] = {}
        
        # Initialize all order books
        for symbol in symbols:
            for venue in venues:
                key = (symbol, venue)
                self.order_books[key] = OrderBook(symbol, venue)
        
        # Cross-venue analytics
        self.arbitrage_opportunities = deque(maxlen=10000)
        self.liquidity_metrics = defaultdict(dict)
        
        # Performance tracking
        self.total_updates = 0
        self.total_trades = 0
        self.start_time = time.time()
        
        logger.info(f"OrderBookManager initialized: {len(symbols)} symbols × {len(venues)} venues = {len(self.order_books)} books")
    
    def get_order_book(self, symbol: str, venue: str) -> Optional[OrderBook]:
        """Get order book for symbol-venue pair"""
        key = (symbol, venue)
        return self.order_books.get(key)
    
    def add_order(self, symbol: str, venue: str, order: Order) -> bool:
        """Add order to specific order book"""
        book = self.get_order_book(symbol, venue)
        if not book:
            logger.error(f"No order book found for {symbol}@{venue}")
            return False
        
        success = book.add_order(order)
        if success:
            self.total_updates += 1
            self._update_cross_venue_analytics(symbol)
        
        return success
    
    def process_book_update(self, update: BookUpdate) -> bool:
        """Process order book update"""
        book = self.get_order_book(update.symbol, update.venue)
        if not book:
            logger.error(f"No order book found for {update.symbol}@{update.venue}")
            return False
        
        success = False
        
        if update.update_type == "add":
            order = Order(
                order_id=update.order_id or f"order_{update.sequence_number}",
                symbol=update.symbol,
                venue=update.venue,
                side=update.side,
                price=update.price,
                size=update.size,
                timestamp=update.timestamp
            )
            success = book.add_order(order)
        
        elif update.update_type == "modify":
            success = book.modify_order(update.order_id, update.size, update.timestamp)
        
        elif update.update_type == "delete":
            success = book.cancel_order(update.order_id, update.timestamp)
        
        elif update.update_type == "trade":
            executed_orders = book.execute_trade(update.price, update.size, update.timestamp)
            success = len(executed_orders) > 0
            if success:
                self.total_trades += 1
        
        if success:
            self.total_updates += 1
            self._update_cross_venue_analytics(update.symbol)
        
        return success
    
    def _update_cross_venue_analytics(self, symbol: str):
        """Update cross-venue analytics for symbol"""
        # Detect arbitrage opportunities
        self._detect_arbitrage_opportunities(symbol)
        
        # Update liquidity metrics
        self._calculate_liquidity_metrics(symbol)
    
    def _detect_arbitrage_opportunities(self, symbol: str):
        """Detect cross-venue arbitrage opportunities"""
        venue_books = []
        current_time = time.time()
        
        # Collect current best bid/ask from all venues
        for venue in self.venues:
            book = self.get_order_book(symbol, venue)
            if book:
                best_bid, best_ask = book.get_best_bid_ask()
                if best_bid and best_ask:
                    venue_books.append({
                        'venue': venue,
                        'bid_price': best_bid[0],
                        'bid_size': best_bid[1],
                        'ask_price': best_ask[0],
                        'ask_size': best_ask[1],
                        'timestamp': book.last_update_time
                    })
        
        if len(venue_books) < 2:
            return
        
        # Find arbitrage opportunities
        for i, buy_venue in enumerate(venue_books):
            for j, sell_venue in enumerate(venue_books):
                if i != j:
                    # Check if we can buy on one venue and sell on another
                    buy_price = buy_venue['ask_price']
                    sell_price = sell_venue['bid_price']
                    
                    gross_profit = sell_price - buy_price
                    
                    if gross_profit > 0.01:  # Minimum 1 cent profit
                        max_size = min(buy_venue['ask_size'], sell_venue['bid_size'])
                        
                        opportunity = {
                            'symbol': symbol,
                            'timestamp': current_time,
                            'buy_venue': buy_venue['venue'],
                            'sell_venue': sell_venue['venue'],
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'profit_per_share': gross_profit,
                            'max_size': max_size,
                            'total_profit': gross_profit * max_size
                        }
                        
                        self.arbitrage_opportunities.append(opportunity)
    
    def process_tick(self, tick):
    
        book = self.get_order_book(tick.symbol, tick.venue)
        if not book:
         logger.warning(f"No order book found for {tick.symbol}@{tick.venue}")
         return
    
    # Create synthetic book updates from tick data
        timestamp = getattr(tick, 'timestamp', time.time())
        sequence_num = int(timestamp * 1000000)  # Use timestamp as sequence number
    
    # Clear existing synthetic orders to avoid stale data
        orders_to_cancel = [order_id for order_id in book.orders.keys() 
                       if order_id.startswith('synthetic_')]
    
        for order_id in orders_to_cancel:
            book.cancel_order(order_id, timestamp)
    
    # Update best bid if available
        if hasattr(tick, 'bid_price') and hasattr(tick, 'bid_size') and tick.bid_price > 0:
            bid_update = BookUpdate(
            symbol=tick.symbol,
            venue=tick.venue,
            timestamp=timestamp,
            sequence_number=sequence_num,
            update_type="add",
            side=OrderSide.BID,
            price=tick.bid_price,
            size=getattr(tick, 'bid_size', 100),
            order_id=f"synthetic_bid_{sequence_num}"
            )
            self.process_book_update(bid_update)
    
    # Update best ask if available
        if hasattr(tick, 'ask_price') and hasattr(tick, 'ask_size') and tick.ask_price > 0:
            ask_update = BookUpdate(
            symbol=tick.symbol,
            venue=tick.venue,
            timestamp=timestamp,
            sequence_number=sequence_num + 1,
            update_type="add",
            side=OrderSide.ASK,
            price=tick.ask_price,
            size=getattr(tick, 'ask_size', 100),
            order_id=f"synthetic_ask_{sequence_num}"
        )
            self.process_book_update(ask_update)
    
        # if hasattr(tick, 'last_price') and hasattr(tick, 'volume') and tick.volume > 0:
        #     trade_update = BookUpdate(
        #     symbol=tick.symbol,
        #     venue=tick.venue,
        #     timestamp=timestamp,
        #     sequence_number=sequence_num + 2,
        #     update_type="trade",
        #     side=OrderSide.BID,  # Assume buyer initiated
        #     price=tick.last_price,
        #     size=tick.volume
        # )
        #     self.process_book_update(trade_update)

    def get_book_state(self, symbol: str, venue: str) -> Dict:
    
        book = self.get_order_book(symbol, venue)
        if not book:
            return {
                'symbol': symbol,
                'venue': venue,
                'best_bid': None,
                'best_ask': None,
                'spread': None,
                'mid_price': None,
                'book_imbalance': 0.0,
                'sequence_number': 0
            }
        best_bid, best_ask = book.get_best_bid_ask()
        snapshot = book.get_snapshot()
    
        return {
            'symbol': symbol,
            'venue': venue,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': snapshot.spread,
            'mid_price': snapshot.mid_price,
            'book_imbalance': book.calculate_book_imbalance(),
            'sequence_number': book.sequence_number,
            'order_count': len(book.orders),
            'last_update': book.last_update_time
        }

    def _calculate_liquidity_metrics(self, symbol: str):
        metrics = {
            'total_bid_liquidity': 0,
            'total_ask_liquidity': 0,
            'weighted_bid_price': 0,
            'weighted_ask_price': 0,
            'venue_count': 0,
            'best_bid': 0,
            'best_ask': float('inf')
        }
        
        total_bid_volume = 0
        total_ask_volume = 0
        
        for venue in self.venues:
            book = self.get_order_book(symbol, venue)
            if not book:
                continue
            
            best_bid, best_ask = book.get_best_bid_ask()
            
            if best_bid:
                metrics['total_bid_liquidity'] += best_bid[1]
                metrics['weighted_bid_price'] += best_bid[0] * best_bid[1]
                total_bid_volume += best_bid[1]
                metrics['best_bid'] = max(metrics['best_bid'], best_bid[0])
            
            if best_ask:
                metrics['total_ask_liquidity'] += best_ask[1]
                metrics['weighted_ask_price'] += best_ask[0] * best_ask[1]
                total_ask_volume += best_ask[1]
                metrics['best_ask'] = min(metrics['best_ask'], best_ask[0])
            
            metrics['venue_count'] += 1
        
        # Calculate weighted averages
        if total_bid_volume > 0:
            metrics['weighted_bid_price'] /= total_bid_volume
        
        if total_ask_volume > 0:
            metrics['weighted_ask_price'] /= total_ask_volume
        
        # Calculate spread and mid-price
        if metrics['best_bid'] > 0 and metrics['best_ask'] < float('inf'):
            metrics['best_spread'] = metrics['best_ask'] - metrics['best_bid']
            metrics['best_mid_price'] = (metrics['best_bid'] + metrics['best_ask']) / 2
        
        self.liquidity_metrics[symbol] = metrics
    
    def get_consolidated_book_depth(self, symbol: str, levels: int = 10) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
        """Get consolidated book depth across all venues"""
        all_bids = []
        all_asks = []
        
        # Collect all price levels from all venues
        for venue in self.venues:
            book = self.get_order_book(symbol, venue)
            if not book:
                continue
            
            bid_depth, ask_depth = book.get_book_depth(levels)
            
            for price, size, _ in bid_depth:
                all_bids.append((price, size))
            
            for price, size, _ in ask_depth:
                all_asks.append((price, size))
        
        # Aggregate by price level
        bid_aggregated = defaultdict(int)
        ask_aggregated = defaultdict(int)
        
        for price, size in all_bids:
            bid_aggregated[price] += size
        
        for price, size in all_asks:
            ask_aggregated[price] += size
        
        # Sort and limit to requested levels
        consolidated_bids = sorted(bid_aggregated.items(), key=lambda x: x[0], reverse=True)[:levels]
        consolidated_asks = sorted(ask_aggregated.items(), key=lambda x: x[0])[:levels]
        
        return consolidated_bids, consolidated_asks
    
    def get_best_execution_plan(self, symbol: str, side: OrderSide, size: int) -> List[Dict]:
        """Get optimal execution plan across venues"""
        execution_plan = []
        remaining_size = size
        
        # Get all available liquidity
        venue_liquidity = []
        
        for venue in self.venues:
            book = self.get_order_book(symbol, venue)
            if not book:
                continue
            
            best_bid, best_ask = book.get_best_bid_ask()
            
            if side == OrderSide.BID and best_ask:
                # Buying - need ask liquidity
                venue_liquidity.append({
                    'venue': venue,
                    'price': best_ask[0],
                    'size': best_ask[1],
                    'book': book
                })
            elif side == OrderSide.ASK and best_bid:
                # Selling - need bid liquidity
                venue_liquidity.append({
                    'venue': venue,
                    'price': best_bid[0],
                    'size': best_bid[1],
                    'book': book
                })
        
        # Sort by best price
        if side == OrderSide.BID:
            venue_liquidity.sort(key=lambda x: x['price'])  # Buy at lowest price
        else:
            venue_liquidity.sort(key=lambda x: x['price'], reverse=True)  # Sell at highest price
        
        # Create execution plan
        for venue_info in venue_liquidity:
            if remaining_size <= 0:
                break
            
            execution_size = min(remaining_size, venue_info['size'])
            
            execution_plan.append({
                'venue': venue_info['venue'],
                'price': venue_info['price'],
                'size': execution_size,
                'estimated_cost': execution_size * venue_info['price']
            })
            
            remaining_size -= execution_size
        
        return execution_plan
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        runtime = time.time() - self.start_time
        
        metrics = {
            'runtime_seconds': runtime,
            'total_updates': self.total_updates,
            'total_trades': self.total_trades,
            'update_rate': self.total_updates / runtime if runtime > 0 else 0,
            'arbitrage_opportunities': len(self.arbitrage_opportunities),
            'symbols_tracked': len(self.symbols),
            'venues_tracked': len(self.venues),
            'total_order_books': len(self.order_books),
            'liquidity_metrics': dict(self.liquidity_metrics)
        }
        
        # Add per-book metrics
        book_metrics = {}
        for (symbol, venue), book in self.order_books.items():
            key = f"{symbol}@{venue}"
            best_bid, best_ask = book.get_best_bid_ask()
            
            book_metrics[key] = {
                'sequence_number': book.sequence_number,
                'update_count': book.update_count,
                'trade_count': book.trade_count,
                'total_volume': book.total_volume,
                'order_count': len(book.orders),
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': (best_ask[0] - best_bid[0]) if (best_bid and best_ask) else None
            }
        
        metrics['book_metrics'] = book_metrics
        
        return metrics
    
    def export_arbitrage_data(self) -> List[Dict]:
        """Export arbitrage opportunity data for analysis"""
        return list(self.arbitrage_opportunities)
    
    def get_market_summary(self, symbol: str) -> Dict:
        """Get comprehensive market summary for a symbol"""
        summary = {
            'symbol': symbol,
            'timestamp': time.time(),
            'venues': {},
            'consolidated': {},
            'arbitrage_count': 0
        }
        
        # Per-venue data
        for venue in self.venues:
            book = self.get_order_book(symbol, venue)
            if book:
                best_bid, best_ask = book.get_best_bid_ask()
                snapshot = book.get_snapshot()
                
                summary['venues'][venue] = {
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'spread': snapshot.spread,
                    'mid_price': snapshot.mid_price,
                    'book_imbalance': book.calculate_book_imbalance(),
                    'order_count': len(book.orders),
                    'update_count': book.update_count
                }
        
        # Consolidated data
        consolidated_bids, consolidated_asks = self.get_consolidated_book_depth(symbol, 5)
        if consolidated_bids and consolidated_asks:
            summary['consolidated'] = {
                'best_bid': consolidated_bids[0],
                'best_ask': consolidated_asks[0],
                'spread': consolidated_asks[0][0] - consolidated_bids[0][0],
                'mid_price': (consolidated_bids[0][0] + consolidated_asks[0][0]) / 2,
                'total_bid_liquidity': sum(size for _, size in consolidated_bids),
                'total_ask_liquidity': sum(size for _, size in consolidated_asks)
            }
        
        # Count recent arbitrage opportunities
        current_time = time.time()
        recent_arbitrage = [
            opp for opp in self.arbitrage_opportunities
            if opp['symbol'] == symbol and (current_time - opp['timestamp']) < 60
        ]
        summary['arbitrage_count'] = len(recent_arbitrage)
        
        return summary


# Example usage and testing
async def test_order_book_manager():
    """Test the OrderBookManager"""
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    venues = ['NYSE', 'NASDAQ', 'ARCA']
    
    manager = OrderBookManager(symbols, venues)
    
    print("Testing OrderBookManager...")
    
    # Add some test orders
    test_orders = [
        # AAPL orders
        Order("order_1", "AAPL", "NYSE", OrderSide.BID, 150.00, 1000, time.time()),
        Order("order_2", "AAPL", "NYSE", OrderSide.BID, 149.99, 500, time.time()),
        Order("order_3", "AAPL", "NYSE", OrderSide.ASK, 150.01, 800, time.time()),
        Order("order_4", "AAPL", "NASDAQ", OrderSide.BID, 150.01, 1200, time.time()),  # Cross-venue arb
        Order("order_5", "AAPL", "NASDAQ", OrderSide.ASK, 150.02, 600, time.time()),
        
        # MSFT orders
        Order("order_6", "MSFT", "NYSE", OrderSide.BID, 300.00, 2000, time.time()),
        Order("order_7", "MSFT", "NASDAQ", OrderSide.ASK, 300.05, 1500, time.time()),
    ]
    
    # Add orders to books
    for order in test_orders:
        success = manager.add_order(order.symbol, order.venue, order)
        print(f"Added order {order.order_id}: {success}")
    
    # Test book updates
    updates = [
        BookUpdate("AAPL", "NYSE", time.time(), 1, "trade", OrderSide.BID, 150.00, 300),
        BookUpdate("MSFT", "NASDAQ", time.time(), 2, "modify", OrderSide.ASK, 300.05, 1000, "order_7"),
    ]
    
    for update in updates:
        success = manager.process_book_update(update)
        print(f"Processed update: {success}")
    
    # Test analytics
    print("\n=== Market Summary ===")
    for symbol in symbols:
        summary = manager.get_market_summary(symbol)
        print(f"\n{symbol}:")
        
        for venue, data in summary['venues'].items():
            if data['best_bid'] and data['best_ask']:
                print(f"  {venue}: {data['best_bid'][0]:.2f} x {data['best_ask'][0]:.2f} "
                      f"(spread: {data['spread']:.3f})")
        
        if summary['consolidated']:
            cons = summary['consolidated']
            print(f"  Consolidated: {cons['best_bid'][0]:.2f} x {cons['best_ask'][0]:.2f}")
        
        print(f"  Arbitrage opportunities: {summary['arbitrage_count']}")
    
    # Performance metrics
    metrics = manager.get_performance_metrics()
    print(f"\n=== Performance Metrics ===")
    print(f"Total updates: {metrics['total_updates']}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Update rate: {metrics['update_rate']:.1f}/sec")
    print(f"Arbitrage opportunities: {metrics['arbitrage_opportunities']}")
    
    return manager

if __name__ == "__main__":
    asyncio.run(test_order_book_manager())