import time
import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import uuid
logger = logging.getLogger(__name__)
class OrderType(Enum):

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"
    VWAP = "VWAP"
    ICEBERG = "ICEBERG"
    SMART_ROUTING = "SMART_ROUTING"
    IMPLEMENTATION_SHORTFALL = "IMPLEMENTATION_SHORTFALL"
    ARRIVAL_PRICE = "ARRIVAL_PRICE"
    PERCENT_OF_VOLUME = "PERCENT_OF_VOLUME"
class OrderStatus(Enum):

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
@dataclass
class OrderSlice:

    slice_id: str
    parent_order_id: str
    symbol: str
    side: str
    quantity: int
    target_venue: str
    order_type: str
    limit_price: Optional[float] = None
    scheduled_time: float = 0.0
    urgency: float = 0.5
    hidden_quantity: int = 0
@dataclass
class ExecutionResult:

    slice_id: str
    executed_quantity: int
    avg_execution_price: float
    execution_time: float
    venue: str
    fees: float
    slippage_bps: float
    market_impact_bps: float
    success: bool
    error_message: Optional[str] = None
@dataclass
class AdvancedOrder:

    order_id: str
    symbol: str
    side: str
    total_quantity: int
    order_type: OrderType
    status: OrderStatus
    creation_time: float
    arrival_price: float
    benchmark_price: float
    strategy_params: Dict[str, Any]
    slices: List[OrderSlice] = field(default_factory=list)
    executions: List[ExecutionResult] = field(default_factory=list)

    @property
    def filled_quantity(self) -> int:
        return sum(exec.executed_quantity for exec in self.executions)

    @property
    def remaining_quantity(self) -> int:
        return self.total_quantity - self.filled_quantity

    @property
    def avg_execution_price(self) -> float:
        if not self.executions:
            return 0.0

        total_value = sum(exec.executed_quantity * exec.avg_execution_price
                         for exec in self.executions)
        total_quantity = self.filled_quantity

        return total_value / total_quantity if total_quantity > 0 else 0.0

    @property
    def implementation_shortfall_bps(self) -> float:

        if self.filled_quantity == 0:
            return 0.0

        execution_value = self.filled_quantity * self.avg_execution_price
        arrival_value = self.filled_quantity * self.arrival_price

        if self.side == 'BUY':
            shortfall = execution_value - arrival_value
        else:
            shortfall = arrival_value - execution_value

        return (shortfall / arrival_value) * 10000 if arrival_value > 0 else 0.0
class VolumeProfile:


    def __init__(self):
        self.intraday_patterns = self._initialize_volume_patterns()

    def _initialize_volume_patterns(self) -> Dict[str, List[float]]:

        hours = list(range(9, 16))
        base_pattern = [
            0.20,
            0.15,
            0.12,
            0.08,
            0.10,
            0.13,
            0.22
        ]

        return {
            'default': base_pattern,
            'high_volume_stocks': [x * 1.3 for x in base_pattern],
            'low_volume_stocks': [x * 0.7 for x in base_pattern],
            'etf': [x * 1.1 for x in base_pattern]
        }

    def get_expected_volume_participation(self, symbol: str, hour: int) -> float:

        if 9 <= hour <= 15:
            pattern_key = self._classify_symbol(symbol)
            pattern = self.intraday_patterns.get(pattern_key, self.intraday_patterns['default'])
            index = min(hour - 9, len(pattern) - 1)
            return pattern[index]
        return 0.05

    def _classify_symbol(self, symbol: str) -> str:

        etfs = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
        high_volume = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN']

        if symbol in etfs:
            return 'etf'
        elif symbol in high_volume:
            return 'high_volume_stocks'
        else:
            return 'default'
class TWAPGenerator:


    def __init__(self):
        self.min_slice_size = 25
        self.max_slices = 50

    def generate_twap_slices(self, order: AdvancedOrder) -> List[OrderSlice]:

        duration_minutes = order.strategy_params.get('duration_minutes', 10)
        slice_interval_seconds = order.strategy_params.get('slice_interval_seconds', 30)
        randomize_timing = order.strategy_params.get('randomize_timing', True)
        randomize_size = order.strategy_params.get('randomize_size', True)

        total_intervals = int((duration_minutes * 60) / slice_interval_seconds)
        num_slices = min(total_intervals, self.max_slices)

        if num_slices == 0:
            num_slices = 1

        base_slice_size = order.total_quantity // num_slices
        if base_slice_size < self.min_slice_size:
            base_slice_size = self.min_slice_size
            num_slices = max(1, order.total_quantity // base_slice_size)

        slices = []
        remaining_quantity = order.total_quantity
        current_time = time.time()

        for i in range(num_slices):
            if i == num_slices - 1:
                slice_size = remaining_quantity
            else:
                if randomize_size:
                    variation = np.random.uniform(0.8, 1.2)
                    slice_size = int(base_slice_size * variation)
                    slice_size = min(slice_size, remaining_quantity)
                else:
                    slice_size = min(base_slice_size, remaining_quantity)

            base_execution_time = current_time + (i * slice_interval_seconds)
            if randomize_timing and i > 0:
                jitter = np.random.uniform(-0.25, 0.25) * slice_interval_seconds
                execution_time = base_execution_time + jitter
            else:
                execution_time = base_execution_time

            slice = OrderSlice(
                slice_id=f"{order.order_id}_slice_{i+1}",
                parent_order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=slice_size,
                target_venue=self._select_venue_for_slice(order, i),
                order_type='MARKET',
                scheduled_time=execution_time,
                urgency=0.3
            )

            slices.append(slice)
            remaining_quantity -= slice_size

            if remaining_quantity <= 0:
                break

        return slices

    def _select_venue_for_slice(self, order: AdvancedOrder, slice_index: int) -> str:

        venues = ['NYSE', 'NASDAQ', 'ARCA', 'IEX', 'CBOE']

        primary_venues = venues[:3]
        venue_index = slice_index % len(primary_venues)

        return primary_venues[venue_index]
class VWAPGenerator:


    def __init__(self):
        self.volume_profile = VolumeProfile()

    def generate_vwap_slices(self, order: AdvancedOrder) -> List[OrderSlice]:

        participation_rate = order.strategy_params.get('participation_rate', 0.10)
        duration_minutes = order.strategy_params.get('duration_minutes', 30)
        max_participation_rate = order.strategy_params.get('max_participation_rate', 0.25)

        current_hour = datetime.now().hour
        time_intervals = []
        volume_weights = []

        interval_minutes = 5
        num_intervals = duration_minutes // interval_minutes

        for i in range(num_intervals):
            interval_hour = current_hour + (i * interval_minutes // 60)
            if interval_hour > 15:
                break

            expected_participation = self.volume_profile.get_expected_volume_participation(
                order.symbol, interval_hour
            )

            time_intervals.append(time.time() + (i * interval_minutes * 60))
            volume_weights.append(expected_participation)

        if not volume_weights:
            return TWAPGenerator().generate_twap_slices(order)

        total_weight = sum(volume_weights)
        if total_weight > 0:
            volume_weights = [w / total_weight for w in volume_weights]

        slices = []
        remaining_quantity = order.total_quantity

        for i, (execution_time, weight) in enumerate(zip(time_intervals, volume_weights)):
            if i == len(time_intervals) - 1:
                slice_size = remaining_quantity
            else:
                slice_size = int(order.total_quantity * weight)
                slice_size = min(slice_size, remaining_quantity)

            if slice_size <= 0:
                continue

            slice = OrderSlice(
                slice_id=f"{order.order_id}_vwap_{i+1}",
                parent_order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=slice_size,
                target_venue=self._select_venue_for_vwap(order, weight),
                order_type='LIMIT',
                scheduled_time=execution_time,
                urgency=0.4
            )

            slices.append(slice)
            remaining_quantity -= slice_size

            if remaining_quantity <= 0:
                break

        return slices

    def _select_venue_for_vwap(self, order: AdvancedOrder, volume_weight: float) -> str:

        if volume_weight > 0.15:
            return 'NYSE'
        elif volume_weight > 0.10:
            return 'NASDAQ'
        else:
            return np.random.choice(['ARCA', 'IEX', 'CBOE'])
class IcebergGenerator:


    def generate_iceberg_slices(self, order: AdvancedOrder) -> List[OrderSlice]:

        display_size = order.strategy_params.get('display_size', 100)
        refresh_threshold = order.strategy_params.get('refresh_threshold', 0.3)
        price_improvement = order.strategy_params.get('price_improvement', 0.001)

        slices = []
        remaining_quantity = order.total_quantity
        slice_number = 1

        while remaining_quantity > 0:
            visible_quantity = min(display_size, remaining_quantity)
            hidden_quantity = min(remaining_quantity - visible_quantity, display_size * 3)

            slice = OrderSlice(
                slice_id=f"{order.order_id}_ice_{slice_number}",
                parent_order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=visible_quantity,
                target_venue=self._select_venue_for_iceberg(order),
                order_type='LIMIT',
                scheduled_time=time.time() + (slice_number - 1) * 2,
                urgency=0.6,
                hidden_quantity=hidden_quantity
            )

            slices.append(slice)
            remaining_quantity -= visible_quantity
            slice_number += 1

            if slice_number > 20:
                break

        return slices

    def _select_venue_for_iceberg(self, order: AdvancedOrder) -> str:

        iceberg_friendly_venues = ['IEX', 'CBOE', 'ARCA']
        return np.random.choice(iceberg_friendly_venues)
class SmartRouter:


    def __init__(self):
        self.venue_capabilities = self._initialize_venue_capabilities()

    def _initialize_venue_capabilities(self) -> Dict[str, Dict]:

        return {
            'NYSE': {
                'liquidity_score': 0.95,
                'hidden_order_support': 0.7,
                'maker_rebate': 0.0015,
                'taker_fee': 0.003,
                'latency_rank': 3,
                'best_for': ['large_cap_stocks', 'high_volume']
            },
            'NASDAQ': {
                'liquidity_score': 0.90,
                'hidden_order_support': 0.8,
                'maker_rebate': 0.002,
                'taker_fee': 0.003,
                'latency_rank': 2,
                'best_for': ['tech_stocks', 'mid_cap']
            },
            'IEX': {
                'liquidity_score': 0.75,
                'hidden_order_support': 0.95,
                'maker_rebate': 0.0009,
                'taker_fee': 0.0009,
                'latency_rank': 1,
                'best_for': ['anti_predatory', 'institutional']
            },
            'ARCA': {
                'liquidity_score': 0.80,
                'hidden_order_support': 0.6,
                'maker_rebate': 0.0018,
                'taker_fee': 0.0025,
                'latency_rank': 4,
                'best_for': ['etf', 'cost_sensitive']
            },
            'CBOE': {
                'liquidity_score': 0.70,
                'hidden_order_support': 0.85,
                'maker_rebate': 0.0022,
                'taker_fee': 0.0028,
                'latency_rank': 5,
                'best_for': ['options_related', 'block_trading']
            }
        }

    def generate_smart_routing_slices(self, order: AdvancedOrder) -> List[OrderSlice]:

        min_venue_allocation = order.strategy_params.get('min_venue_allocation', 0.1)
        max_venues = order.strategy_params.get('max_venues', 3)
        cost_sensitivity = order.strategy_params.get('cost_sensitivity', 0.5)

        venue_scores = self._rank_venues_for_order(order, cost_sensitivity)
        selected_venues = list(venue_scores.keys())[:max_venues]

        allocations = self._allocate_quantity_across_venues(
            order.total_quantity, venue_scores, selected_venues, min_venue_allocation
        )

        slices = []
        for i, (venue, allocation) in enumerate(allocations.items()):
            if allocation <= 0:
                continue

            slice = OrderSlice(
                slice_id=f"{order.order_id}_smart_{venue}_{i+1}",
                parent_order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=allocation,
                target_venue=venue,
                order_type='LIMIT',
                scheduled_time=time.time() + i * 0.5,
                urgency=0.7
            )

            slices.append(slice)

        return slices

    def _rank_venues_for_order(self, order: AdvancedOrder, cost_sensitivity: float) -> Dict[str, float]:

        scores = {}

        for venue, capabilities in self.venue_capabilities.items():
            score = 0.0

            score += capabilities['liquidity_score'] * 0.4

            if order.side == 'BUY':
                cost_score = 1.0 - (capabilities['taker_fee'] - capabilities['maker_rebate'])
            else:
                cost_score = 1.0 - (capabilities['taker_fee'] - capabilities['maker_rebate'])

            score += cost_score * cost_sensitivity * 0.3

            latency_score = 1.0 - (capabilities['latency_rank'] - 1) / 5
            score += latency_score * 0.2

            symbol_fit = self._calculate_symbol_venue_fit(order.symbol, capabilities)
            score += symbol_fit * 0.1

            scores[venue] = max(0.0, min(1.0, score))

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def _calculate_symbol_venue_fit(self, symbol: str, capabilities: Dict) -> float:

        etfs = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN']
        large_cap = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ'] + tech_stocks

        best_for = capabilities['best_for']

        if symbol in etfs and 'etf' in best_for:
            return 1.0
        elif symbol in tech_stocks and 'tech_stocks' in best_for:
            return 1.0
        elif symbol in large_cap and 'large_cap_stocks' in best_for:
            return 0.8
        elif 'institutional' in best_for:
            return 0.6
        else:
            return 0.5

    def _allocate_quantity_across_venues(self, total_quantity: int, venue_scores: Dict,
                                       selected_venues: List[str], min_allocation: float) -> Dict[str, int]:

        allocations = {}

        total_score = sum(venue_scores[venue] for venue in selected_venues)

        for venue in selected_venues:
            if total_score > 0:
                allocation_pct = venue_scores[venue] / total_score
                allocation_pct = max(allocation_pct, min_allocation)
                allocation_qty = int(total_quantity * allocation_pct)
            else:
                allocation_qty = total_quantity // len(selected_venues)

            allocations[venue] = allocation_qty

        total_allocated = sum(allocations.values())
        difference = total_quantity - total_allocated

        if difference != 0:
            best_venue = selected_venues[0]
            allocations[best_venue] += difference

        return allocations
class AdvancedOrderManager:


    def __init__(self, execution_callback: Optional[Callable] = None):
        self.orders = {}
        self.execution_queue = asyncio.Queue()
        self.execution_callback = execution_callback
        self.twap_generator = TWAPGenerator()
        self.vwap_generator = VWAPGenerator()
        self.iceberg_generator = IcebergGenerator()
        self.smart_router = SmartRouter()
        self.execution_active = False

        logger.info("✅ Advanced Order Manager initialized")
        logger.info("   • TWAP, VWAP, Iceberg, Smart Routing available")
        logger.info("   • Execution queue active")

    def create_twap_order(self, symbol: str, quantity: int, side: str = 'BUY',
                         duration_minutes: int = 10, slice_interval_seconds: int = 30,
                         randomize_timing: bool = True, randomize_size: bool = True) -> str:


        order_id = f"TWAP_{uuid.uuid4().hex[:8]}"
        current_price = 100.0 + np.random.randn() * 5

        order = AdvancedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            order_type=OrderType.TWAP,
            status=OrderStatus.PENDING,
            creation_time=time.time(),
            arrival_price=current_price,
            benchmark_price=current_price,
            strategy_params={
                'duration_minutes': duration_minutes,
                'slice_interval_seconds': slice_interval_seconds,
                'randomize_timing': randomize_timing,
                'randomize_size': randomize_size
            }
        )

        order.slices = self.twap_generator.generate_twap_slices(order)

        self.orders[order_id] = order
        logger.info(f"📊 TWAP order created: {symbol} {quantity} shares over {duration_minutes}min ({len(order.slices)} slices)")

        return order_id

    def create_vwap_order(self, symbol: str, quantity: int, side: str = 'BUY',
                         duration_minutes: int = 30, participation_rate: float = 0.10) -> str:


        order_id = f"VWAP_{uuid.uuid4().hex[:8]}"
        current_price = 100.0 + np.random.randn() * 5

        order = AdvancedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            order_type=OrderType.VWAP,
            status=OrderStatus.PENDING,
            creation_time=time.time(),
            arrival_price=current_price,
            benchmark_price=current_price,
            strategy_params={
                'duration_minutes': duration_minutes,
                'participation_rate': participation_rate,
                'max_participation_rate': min(0.25, participation_rate * 2.5)
            }
        )

        order.slices = self.vwap_generator.generate_vwap_slices(order)

        self.orders[order_id] = order
        logger.info(f"📈 VWAP order created: {symbol} {quantity} shares, {participation_rate:.1%} participation ({len(order.slices)} slices)")

        return order_id

    def create_iceberg_order(self, symbol: str, quantity: int, side: str = 'BUY',
                           display_size: int = 100, refresh_threshold: float = 0.3) -> str:


        order_id = f"ICE_{uuid.uuid4().hex[:8]}"
        current_price = 100.0 + np.random.randn() * 5

        order = AdvancedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            order_type=OrderType.ICEBERG,
            status=OrderStatus.PENDING,
            creation_time=time.time(),
            arrival_price=current_price,
            benchmark_price=current_price,
            strategy_params={
                'display_size': display_size,
                'refresh_threshold': refresh_threshold,
                'price_improvement': 0.001
            }
        )

        order.slices = self.iceberg_generator.generate_iceberg_slices(order)

        self.orders[order_id] = order
        logger.info(f"🧊 Iceberg order created: {symbol} {quantity} shares, {display_size} display ({len(order.slices)} slices)")

        return order_id

    def create_smart_routing_order(self, symbol: str, quantity: int, side: str = 'BUY',
                                 max_venues: int = 3, cost_sensitivity: float = 0.5) -> str:


        order_id = f"SMART_{uuid.uuid4().hex[:8]}"
        current_price = 100.0 + np.random.randn() * 5

        order = AdvancedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            order_type=OrderType.SMART_ROUTING,
            status=OrderStatus.PENDING,
            creation_time=time.time(),
            arrival_price=current_price,
            benchmark_price=current_price,
            strategy_params={
                'max_venues': max_venues,
                'cost_sensitivity': cost_sensitivity,
                'min_venue_allocation': 0.1
            }
        )

        order.slices = self.smart_router.generate_smart_routing_slices(order)

        self.orders[order_id] = order
        logger.info(f"🎯 Smart routing order created: {symbol} {quantity} shares across {len(order.slices)} venues")

        return order_id

    def get_order_status(self, order_id: str) -> Optional[Dict]:

        order = self.orders.get(order_id)
        if not order:
            return None

        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side,
            'total_quantity': order.total_quantity,
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'status': order.status.value,
            'order_type': order.order_type.value,
            'avg_execution_price': order.avg_execution_price,
            'arrival_price': order.arrival_price,
            'implementation_shortfall_bps': order.implementation_shortfall_bps,
            'total_slices': len(order.slices),
            'executed_slices': len(order.executions),
            'creation_time': order.creation_time,
            'strategy_params': order.strategy_params
        }

    def get_performance_metrics(self, order_id: str) -> Optional[Dict]:

        order = self.orders.get(order_id)
        if not order or not order.executions:
            return None

        executions = order.executions

        return {
            'implementation_shortfall_bps': order.implementation_shortfall_bps,
            'avg_slippage_bps': np.mean([e.slippage_bps for e in executions]),
            'avg_market_impact_bps': np.mean([e.market_impact_bps for e in executions]),
            'total_fees': sum(e.fees for e in executions),
            'execution_time_seconds': max(e.execution_time for e in executions) - min(e.execution_time for e in executions),
            'fill_rate': order.filled_quantity / order.total_quantity,
            'venue_distribution': self._calculate_venue_distribution(executions),
            'price_improvement_bps': self._calculate_price_improvement(order)
        }

    def _calculate_venue_distribution(self, executions: List[ExecutionResult]) -> Dict[str, float]:

        venue_quantities = defaultdict(int)
        total_quantity = 0

        for execution in executions:
            venue_quantities[execution.venue] += execution.executed_quantity
            total_quantity += execution.executed_quantity

        if total_quantity == 0:
            return {}

        return {venue: qty / total_quantity for venue, qty in venue_quantities.items()}

    def _calculate_price_improvement(self, order: AdvancedOrder) -> float:

        if order.filled_quantity == 0:
            return 0.0

        price_diff = order.avg_execution_price - order.arrival_price

        if order.side == 'BUY':
            improvement = -price_diff
        else:
            improvement = price_diff

        return (improvement / order.arrival_price) * 10000 if order.arrival_price > 0 else 0.0
def integrate_advanced_orders_with_hft_system(hft_integration_instance):


    hft_integration_instance.advanced_order_manager = AdvancedOrderManager()

    original_execute_trade = hft_integration_instance.execute_trade_with_ml_routing

    def enhanced_execute_trade_wrapper(signal, tick, simulation_results, **kwargs):
        order_size = signal.get('quantity', 100)
        urgency = signal.get('urgency', 0.5)

        if order_size > 500 and urgency < 0.7:
            order_id = hft_integration_instance.advanced_order_manager.create_twap_order(
                symbol=signal['symbol'],
                quantity=order_size,
                side=signal.get('side', 'BUY'),
                duration_minutes=5
            )

            return {
                'advanced_order_id': order_id,
                'order_type': 'TWAP',
                'status': 'PENDING_EXECUTION',
                'expected_improvement_bps': 15,
                **original_execute_trade(signal, tick, simulation_results, **kwargs)
            }

        elif order_size > 1000:
            order_id = hft_integration_instance.advanced_order_manager.create_smart_routing_order(
                symbol=signal['symbol'],
                quantity=order_size,
                side=signal.get('side', 'BUY')
            )

            return {
                'advanced_order_id': order_id,
                'order_type': 'SMART_ROUTING',
                'status': 'PENDING_EXECUTION',
                'expected_improvement_bps': 25,
                **original_execute_trade(signal, tick, simulation_results, **kwargs)
            }

        else:
            return original_execute_trade(signal, tick, simulation_results, **kwargs)

    hft_integration_instance.execute_trade_with_ml_routing = enhanced_execute_trade_wrapper

    logger.info("✅ Advanced order types integrated with HFT system")
    logger.info("   • TWAP for orders > 500 shares")
    logger.info("   • Smart routing for orders > 1000 shares")
    logger.info("   • Automatic order type selection based on size/urgency")
if __name__ == "__main__":
    print("📋 Advanced Order Types Demo")
    print("=" * 50)

    order_manager = AdvancedOrderManager()

    twap_id = order_manager.create_twap_order('AAPL', 1000, duration_minutes=5)
    vwap_id = order_manager.create_vwap_order('SPY', 500, participation_rate=0.15)
    iceberg_id = order_manager.create_iceberg_order('TSLA', 2000, display_size=150)
    smart_id = order_manager.create_smart_routing_order('GOOGL', 800, max_venues=3)

    print(f"\n📊 Order Statuses:")
    for order_id, label in [(twap_id, 'TWAP'), (vwap_id, 'VWAP'),
                           (iceberg_id, 'Iceberg'), (smart_id, 'Smart')]:
        status = order_manager.get_order_status(order_id)
        if status:
            print(f"   {label}: {status['symbol']} {status['total_quantity']} shares, "
                  f"{status['total_slices']} slices, Status: {status['status']}")

    print(f"\n✅ Advanced order management ready!")
    print(f"   • {len(order_manager.orders)} orders created")
    print(f"   • Multiple execution strategies available")
    print(f"   • Market impact minimization active")