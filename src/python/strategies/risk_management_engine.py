import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import time
import logging
from datetime import datetime, timedelta
import json
from core.trading_simulator import Order, Fill, OrderSide, TradingStrategyType
logger = logging.getLogger(__name__)

class RiskMetric(Enum):

    POSITION = "position"
    EXPOSURE = "exposure"
    CONCENTRATION = "concentration"
    DRAWDOWN = "drawdown"
    VAR = "value_at_risk"
    STRESS = "stress_test"
    OPERATIONAL = "operational"

class RiskLevel(Enum):

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskLimit:

    metric: RiskMetric
    limit_type: str
    threshold: float
    current_value: float = 0.0
    breach_count: int = 0
    last_breach_time: Optional[float] = None
    action: str = "alert"

@dataclass
class RiskAlert:

    timestamp: float
    metric: RiskMetric
    level: RiskLevel
    message: str
    current_value: float
    limit_value: float
    action_taken: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class PnLComponent:

    source: str = ""
    gross_pnl: float = 0.0
    fees: float = 0.0
    rebates: float = 0.0
    market_impact: float = 0.0
    latency_cost: float = 0.0
    net_pnl: float = 0.0
    trade_count: int = 0
    volume: int = 0

    @property
    def pnl_per_trade(self) -> float:
        return self.net_pnl / self.trade_count if self.trade_count > 0 else 0

    @property
    def pnl_per_share(self) -> float:
        return self.net_pnl / self.volume if self.volume > 0 else 0
    
class RiskManager:

    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()

        self.risk_limits = self._initialize_risk_limits()

        self.positions = defaultdict(lambda: defaultdict(float))
        self.exposures = defaultdict(float)

        self.realized_pnl = defaultdict(float)
        self.unrealized_pnl = defaultdict(float)
        self.high_water_mark = 0
        self.current_drawdown = 0

        self.risk_history = deque(maxlen=10000)
        self.var_history = deque(maxlen=1000)

        self.active_alerts = []
        self.alert_history = []
        self.trading_allowed = True
        self.restricted_symbols = set()

        self.market_prices = {}
        self.volatilities = {}
        self.correlations = pd.DataFrame()

        logger.info("RiskManager initialized with limits:")
        for limit_name, limit in self.risk_limits.items():
            logger.info(f"  {limit_name}: {limit.threshold}")

    def _get_default_config(self) -> Dict:

        return {
            'position_limits': {
                'max_position_size': 10000,
                'max_total_positions': 100000,
                'max_position_value': 1000000
            },
            'exposure_limits': {
                'max_gross_exposure': 5000000,
                'max_net_exposure': 2000000,
                'max_sector_exposure': 1000000
            },
            'concentration_limits': {
                'max_symbol_concentration': 0.20,
                'max_venue_concentration': 0.40,
                'max_strategy_concentration': 0.50
            },
            'drawdown_limits': {
                'soft_drawdown_limit': 50000,
                'hard_drawdown_limit': 100000,
                'daily_loss_limit': 25000,
                'trailing_stop_pct': 0.10
            },
            'var_limits': {
                'confidence_level': 0.95,
                'var_limit': 75000,
                'expected_shortfall_limit': 100000
            },
            'operational_limits': {
                'max_order_rate': 1000,
                'max_message_rate': 10000,
                'min_heartbeat_interval': 1.0,
                'max_latency_ms': 10.0
            }
        }

    def _initialize_risk_limits(self) -> Dict[str, RiskLimit]:

        limits = {}

        limits['max_position_size'] = RiskLimit(
            metric=RiskMetric.POSITION,
            limit_type='hard',
            threshold=self.config['position_limits']['max_position_size']
        )

        limits['max_gross_exposure'] = RiskLimit(
            metric=RiskMetric.EXPOSURE,
            limit_type='hard',
            threshold=self.config['exposure_limits']['max_gross_exposure']
        )

        limits['max_concentration'] = RiskLimit(
            metric=RiskMetric.CONCENTRATION,
            limit_type='soft',
            threshold=self.config['concentration_limits']['max_symbol_concentration']
        )

        limits['max_drawdown'] = RiskLimit(
            metric=RiskMetric.DRAWDOWN,
            limit_type='hard',
            threshold=self.config['drawdown_limits']['hard_drawdown_limit'],
            action='stop'
        )

        limits['var_limit'] = RiskLimit(
            metric=RiskMetric.VAR,
            limit_type='soft',
            threshold=self.config['var_limits']['var_limit']
        )

        return limits

    def check_pre_trade_risk(self, order: Order, current_prices: Dict[str, float]) -> Tuple[bool, Optional[str]]:

        if not self.trading_allowed:
            return False, "Trading halted due to risk limits"

        if order.symbol in self.restricted_symbols:
            return False, f"Symbol {order.symbol} is restricted"

        current_position = self.positions[order.strategy.value][order.symbol]
        new_position = current_position

        if order.side == OrderSide.BUY:
            new_position += order.quantity
        else:
            new_position -= order.quantity

        if abs(new_position) > self.risk_limits['max_position_size'].threshold:
            return False, f"Position size limit exceeded: {abs(new_position)} > {self.risk_limits['max_position_size'].threshold}"

        price = current_prices.get(order.symbol, order.price)
        order_value = order.quantity * price

        current_gross_exposure = sum(
            abs(qty) * current_prices.get(sym, 0)
            for positions in self.positions.values()
            for sym, qty in positions.items()
        )

        new_gross_exposure = current_gross_exposure + order_value

        if new_gross_exposure > self.risk_limits['max_gross_exposure'].threshold:
            return False, f"Gross exposure limit exceeded: ${new_gross_exposure:,.0f}"

        total_exposure = sum(abs(exp) for exp in self.exposures.values())
        if total_exposure > 0:
            symbol_concentration = abs(self.exposures.get(order.symbol, 0) + order_value) / (total_exposure + order_value)
            if symbol_concentration > self.risk_limits['max_concentration'].threshold:
                return False, f"Concentration limit exceeded for {order.symbol}: {symbol_concentration:.1%}"

        if self.current_drawdown > self.config['drawdown_limits']['soft_drawdown_limit']:
            if not self._is_risk_reducing(order, current_position):
                return False, "Only risk-reducing trades allowed during drawdown"

        return True, None

    def _is_risk_reducing(self, order: Order, current_position: float) -> bool:

        if current_position == 0:
            return False

        if current_position > 0 and order.side == OrderSide.SELL:
            return True
        if current_position < 0 and order.side == OrderSide.BUY:
            return True

        return False

    def update_position(self, fill: Fill, current_prices: Dict[str, float]):

        strategy = fill.order.strategy.value if hasattr(fill, 'order') else 'unknown'

        if fill.side == OrderSide.BUY:
            self.positions[strategy][fill.symbol] += fill.quantity
        else:
            self.positions[strategy][fill.symbol] -= fill.quantity

        self._update_exposures(current_prices)

        self.market_prices[fill.symbol] = fill.price

        self._check_risk_limits(current_prices)

        self._record_risk_snapshot(current_prices)

    def _update_exposures(self, current_prices: Dict[str, float]):

        self.exposures.clear()

        for strategy_positions in self.positions.values():
            for symbol, quantity in strategy_positions.items():
                price = current_prices.get(symbol, self.market_prices.get(symbol, 0))
                if symbol not in self.exposures:
                    self.exposures[symbol] = 0
                self.exposures[symbol] += quantity * price

    def _check_risk_limits(self, current_prices: Dict[str, float]):

        for strategy, positions in self.positions.items():
            for symbol, quantity in positions.items():
                if abs(quantity) > self.risk_limits['max_position_size'].threshold:
                    self._trigger_risk_alert(
                        RiskMetric.POSITION,
                        RiskLevel.HIGH,
                        f"Position limit breached for {symbol}: {abs(quantity)}",
                        abs(quantity),
                        self.risk_limits['max_position_size'].threshold
                    )

        gross_exposure = sum(abs(exp) for exp in self.exposures.values())
        net_exposure = sum(self.exposures.values())

        self.risk_limits['max_gross_exposure'].current_value = gross_exposure

        if gross_exposure > self.risk_limits['max_gross_exposure'].threshold:
            self._trigger_risk_alert(
                RiskMetric.EXPOSURE,
                RiskLevel.CRITICAL,
                f"Gross exposure limit breached: ${gross_exposure:,.0f}",
                gross_exposure,
                self.risk_limits['max_gross_exposure'].threshold
            )
            self._take_risk_action('reduce_positions')

        if gross_exposure > 0:
            for symbol, exposure in self.exposures.items():
                concentration = abs(exposure) / gross_exposure
                if concentration > self.risk_limits['max_concentration'].threshold:
                    self._trigger_risk_alert(
                        RiskMetric.CONCENTRATION,
                        RiskLevel.MEDIUM,
                        f"Concentration limit breached for {symbol}: {concentration:.1%}",
                        concentration,
                        self.risk_limits['max_concentration'].threshold
                    )

        self._check_drawdown_limits()

    def _check_drawdown_limits(self):

        total_pnl = sum(self.realized_pnl.values()) + sum(self.unrealized_pnl.values())

        if total_pnl > self.high_water_mark:
            self.high_water_mark = total_pnl
            self.current_drawdown = 0
        else:
            self.current_drawdown = self.high_water_mark - total_pnl

        self.risk_limits['max_drawdown'].current_value = self.current_drawdown

        if self.current_drawdown > self.config['drawdown_limits']['hard_drawdown_limit']:
            self._trigger_risk_alert(
                RiskMetric.DRAWDOWN,
                RiskLevel.CRITICAL,
                f"Hard drawdown limit breached: ${self.current_drawdown:,.0f}",
                self.current_drawdown,
                self.config['drawdown_limits']['hard_drawdown_limit']
            )
            self._take_risk_action('stop_trading')

        elif self.current_drawdown > self.config['drawdown_limits']['soft_drawdown_limit']:
            self._trigger_risk_alert(
                RiskMetric.DRAWDOWN,
                RiskLevel.HIGH,
                f"Soft drawdown limit breached: ${self.current_drawdown:,.0f}",
                self.current_drawdown,
                self.config['drawdown_limits']['soft_drawdown_limit']
            )
            self._take_risk_action('reduce_risk')

    def check_all_limits(self):
        alerts = []
        for strategy, positions in self.positions.items():
            for symbol, quantity in positions.items():
                if abs(quantity) > self.risk_limits['max_position_size'].threshold:
                    alert = RiskAlert(
                        timestamp=time.time(),
                        metric=RiskMetric.POSITION,
                        level=RiskLevel.HIGH,
                        message=f"Position limit breached for {symbol}: {abs(quantity)}",
                        current_value=abs(quantity),
                        limit_value=self.risk_limits['max_position_size'].threshold,
                        action_taken="monitoring"
                    )
                    alerts.append(alert)
        gross_exposure = sum(abs(exp) for exp in self.exposures.values())
        if gross_exposure > self.risk_limits['max_gross_exposure'].threshold:
            alert = RiskAlert(
                timestamp=time.time(),
                metric=RiskMetric.EXPOSURE,
                level=RiskLevel.CRITICAL,
                message=f"Gross exposure limit breached: ${gross_exposure:,.0f}",
                current_value=gross_exposure,
                limit_value=self.risk_limits['max_gross_exposure'].threshold,
                action_taken="monitoring"
            )
            alerts.append(alert)
        if gross_exposure > 0:
            for symbol, exposure in self.exposures.items():
                concentration = abs(exposure) / gross_exposure
                if concentration > self.risk_limits['max_concentration'].threshold:
                    alert = RiskAlert(
                        timestamp=time.time(),
                        metric=RiskMetric.CONCENTRATION,
                        level=RiskLevel.MEDIUM,
                        message=f"Concentration limit breached for {symbol}: {concentration:.1%}",
                        current_value=concentration,
                        limit_value=self.risk_limits['max_concentration'].threshold,
                        action_taken="monitoring"
                    )
                    alerts.append(alert)
        if self.current_drawdown > self.config['drawdown_limits']['hard_drawdown_limit']:
            alert = RiskAlert(
                timestamp=time.time(),
                metric=RiskMetric.DRAWDOWN,
                level=RiskLevel.CRITICAL,
                message=f"Hard drawdown limit breached: ${self.current_drawdown:,.0f}",
                current_value=self.current_drawdown,
                limit_value=self.config['drawdown_limits']['hard_drawdown_limit'],
                action_taken="monitoring"
            )
            alerts.append(alert)
        elif self.current_drawdown > self.config['drawdown_limits']['soft_drawdown_limit']:
            alert = RiskAlert(
                timestamp=time.time(),
                metric=RiskMetric.DRAWDOWN,
                level=RiskLevel.HIGH,
                message=f"Soft drawdown limit breached: ${self.current_drawdown:,.0f}",
                current_value=self.current_drawdown,
                limit_value=self.config['drawdown_limits']['soft_drawdown_limit'],
                action_taken="monitoring"
            )
            alerts.append(alert)
        if self.var_history:
            current_var = self.var_history[-1]['var']
            if current_var > self.risk_limits['var_limit'].threshold:
                alert = RiskAlert(
                    timestamp=time.time(),
                    metric=RiskMetric.VAR,
                    level=RiskLevel.HIGH,
                    message=f"VaR limit exceeded: ${current_var:,.0f}",
                    current_value=current_var,
                    limit_value=self.risk_limits['var_limit'].threshold,
                    action_taken="monitoring"
                )
                alerts.append(alert)
        return alerts

    def _trigger_risk_alert(self, metric: RiskMetric, level: RiskLevel,
                           message: str, current_value: float, limit_value: float):

        alert = RiskAlert(
            timestamp=time.time(),
            metric=metric,
            level=level,
            message=message,
            current_value=current_value,
            limit_value=limit_value,
            action_taken="monitoring"
        )

        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        logger.warning(f"RISK ALERT [{level.value}]: {message}")

        for limit in self.risk_limits.values():
            if limit.metric == metric:
                limit.breach_count += 1
                limit.last_breach_time = time.time()

    def _take_risk_action(self, action: str):

        if action == 'stop_trading':
            self.trading_allowed = False
            logger.critical("TRADING HALTED due to risk limits")

        elif action == 'reduce_positions':
            logger.warning("Position reduction required")

        elif action == 'reduce_risk':
            logger.warning("Risk reduction mode activated")

    def update_pnl(self, strategy: str, realized: float = 0, unrealized: float = 0):

        if realized != 0:
            self.realized_pnl[strategy] += realized
        if unrealized != 0:
            self.unrealized_pnl[strategy] = unrealized

    def calculate_var(self, confidence_level: float = 0.95,
                     horizon_days: int = 1) -> Dict[str, float]:

        if len(self.risk_history) < 100:
            return {'var': 0, 'expected_shortfall': 0, 'confidence_level': confidence_level}

        pnl_changes = []
        for i in range(1, len(self.risk_history)):
            prev_pnl = self.risk_history[i-1].get('total_pnl', 0)
            curr_pnl = self.risk_history[i].get('total_pnl', 0)
            pnl_changes.append(curr_pnl - prev_pnl)

        if not pnl_changes:
            return {'var': 0, 'expected_shortfall': 0, 'confidence_level': confidence_level}

        pnl_changes = np.array(pnl_changes)
        var_percentile = (1 - confidence_level) * 100
        var = -np.percentile(pnl_changes, var_percentile)

        losses = -pnl_changes[pnl_changes < -var]
        expected_shortfall = np.mean(losses) if len(losses) > 0 else var

        var *= np.sqrt(horizon_days)
        expected_shortfall *= np.sqrt(horizon_days)

        result = {
            'var': var,
            'expected_shortfall': expected_shortfall,
            'confidence_level': confidence_level,
            'horizon_days': horizon_days,
            'sample_size': len(pnl_changes)
        }

        self.risk_limits['var_limit'].current_value = var
        if var > self.risk_limits['var_limit'].threshold:
            self._trigger_risk_alert(
                RiskMetric.VAR,
                RiskLevel.HIGH,
                f"VaR limit exceeded: ${var:,.0f}",
                var,
                self.risk_limits['var_limit'].threshold
            )

        self.var_history.append(result)
        return result

    def run_stress_test(self, scenarios: Dict[str, Dict]) -> Dict[str, Dict]:

        results = {}

        for scenario_name, shocks in scenarios.items():
            scenario_pnl = 0

            for symbol, positions_by_strategy in self.get_all_positions().items():
                shock = shocks.get(symbol, shocks.get('default', 0))
                current_price = self.market_prices.get(symbol, 100)
                shocked_price = current_price * (1 + shock / 100)

                total_position = sum(positions_by_strategy.values())
                position_pnl = total_position * (shocked_price - current_price)
                scenario_pnl += position_pnl

            results[scenario_name] = {
                'total_pnl_impact': scenario_pnl,
                'pct_of_capital': scenario_pnl / self.config['exposure_limits']['max_gross_exposure'] * 100,
                'would_breach_limits': abs(scenario_pnl) > self.config['drawdown_limits']['hard_drawdown_limit']
            }

        return results

    def get_all_positions(self) -> Dict[str, Dict[str, float]]:

        all_positions = defaultdict(dict)

        for strategy, positions in self.positions.items():
            for symbol, quantity in positions.items():
                if quantity != 0:
                    all_positions[symbol][strategy] = quantity

        return dict(all_positions)

    def _record_risk_snapshot(self, current_prices: Dict[str, float]):

        gross_exposure = sum(abs(exp) for exp in self.exposures.values())
        net_exposure = sum(self.exposures.values())
        total_pnl = sum(self.realized_pnl.values()) + sum(self.unrealized_pnl.values())

        snapshot = {
            'timestamp': time.time(),
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'total_pnl': total_pnl,
            'drawdown': self.current_drawdown,
            'position_count': sum(len(p) for p in self.positions.values()),
            'active_alerts': len(self.active_alerts)
        }

        self.risk_history.append(snapshot)

    def get_risk_report(self) -> Dict[str, Any]:

        return {
            'summary': {
                'trading_allowed': self.trading_allowed,
                'gross_exposure': self.risk_limits['max_gross_exposure'].current_value,
                'current_drawdown': self.current_drawdown,
                'active_alerts': len(self.active_alerts),
                'restricted_symbols': list(self.restricted_symbols)
            },
            'limits': {
                name: {
                    'current': limit.current_value,
                    'threshold': limit.threshold,
                    'breach_count': limit.breach_count,
                    'last_breach': limit.last_breach_time
                }
                for name, limit in self.risk_limits.items()
            },
            'positions': self.get_all_positions(),
            'var_metrics': self.var_history[-1] if self.var_history else {},
            'recent_alerts': self.alert_history[-10:] if self.alert_history else []
        }
    
class PnLAttribution:

    def __init__(self):
        self.pnl_components = defaultdict(lambda: defaultdict(lambda: PnLComponent()))
        self.attribution_history = []
        self.fee_tracker = FeeTracker()
        self.latency_cost_model = LatencyCostModel()

    def attribute_fill(self, fill: Fill, order: Order, market_state: Dict):

        strategy = order.strategy.value
        venue = fill.venue
        hour = datetime.fromtimestamp(fill.timestamp).hour
        regime = market_state.get('regime', 'normal')

        if order.strategy == TradingStrategyType.MARKET_MAKING:
            revenue_source = 'spread_capture'
            gross_pnl = self._calculate_spread_capture(fill, market_state)
        elif order.strategy == TradingStrategyType.ARBITRAGE:
            revenue_source = 'arbitrage'
            gross_pnl = self._calculate_arbitrage_pnl(fill, market_state)
        elif order.strategy == TradingStrategyType.MOMENTUM:
            revenue_source = 'momentum'
            gross_pnl = 0
        else:
            revenue_source = 'other'
            gross_pnl = 0

        fees = fill.fees
        rebates = fill.rebate
        market_impact = fill.market_impact_bps * fill.price * fill.quantity / 10000
        latency_cost = self.latency_cost_model.calculate_cost(fill, order)

        for key in [(strategy, revenue_source), (venue, 'venue'), (hour, 'hour'), (regime, 'regime')]:
            component = self.pnl_components[key[0]][key[1]]
            component.source = key[1]
            component.gross_pnl += gross_pnl
            component.fees += fees
            component.rebates += rebates
            component.market_impact += market_impact
            component.latency_cost += latency_cost
            component.net_pnl += gross_pnl - fees + rebates - market_impact - latency_cost
            component.trade_count += 1
            component.volume += fill.quantity

    def _calculate_spread_capture(self, fill: Fill, market_state: Dict) -> float:

        mid_price = market_state.get('mid_price', fill.price)

        if fill.side == OrderSide.BUY:
            spread_capture = (mid_price - fill.price) * fill.quantity
        else:
            spread_capture = (fill.price - mid_price) * fill.quantity

        return max(0, spread_capture)

    def _calculate_arbitrage_pnl(self, fill: Fill, market_state: Dict) -> float:

        return fill.quantity * 0.05

    def close_position(self, symbol: str, strategy: str, avg_entry_price: float,
                      exit_price: float, quantity: int):

        if strategy == TradingStrategyType.MOMENTUM.value:
            pnl = (exit_price - avg_entry_price) * quantity
            component = self.pnl_components[strategy]['momentum']
            component.gross_pnl += pnl
            component.net_pnl += pnl

    def get_attribution_report(self) -> Dict[str, Any]:

        report = {
            'total_pnl': 0,
            'by_strategy': {},
            'by_venue': {},
            'by_hour': {},
            'by_regime': {},
            'by_source': {},
            'cost_breakdown': {
                'total_fees': 0,
                'total_rebates': 0,
                'total_market_impact': 0,
                'total_latency_cost': 0
            }
        }

        for primary_key, components in self.pnl_components.items():
            for secondary_key, component in components.items():
                if secondary_key == 'venue':
                    report['by_venue'][primary_key] = self._component_to_dict(component)
                elif secondary_key == 'hour':
                    report['by_hour'][primary_key] = self._component_to_dict(component)
                elif secondary_key == 'regime':
                    report['by_regime'][primary_key] = self._component_to_dict(component)
                elif secondary_key in ['spread_capture', 'arbitrage', 'momentum']:
                    report['by_source'][secondary_key] = self._component_to_dict(component)
                    if primary_key not in report['by_strategy']:
                        report['by_strategy'][primary_key] = self._component_to_dict(component)

                report['total_pnl'] += component.net_pnl
                report['cost_breakdown']['total_fees'] += component.fees
                report['cost_breakdown']['total_rebates'] += component.rebates
                report['cost_breakdown']['total_market_impact'] += component.market_impact
                report['cost_breakdown']['total_latency_cost'] += component.latency_cost

        return report

    def _component_to_dict(self, component: PnLComponent) -> Dict:

        return {
            'gross_pnl': component.gross_pnl,
            'fees': component.fees,
            'rebates': component.rebates,
            'market_impact': component.market_impact,
            'latency_cost': component.latency_cost,
            'net_pnl': component.net_pnl,
            'trade_count': component.trade_count,
            'volume': component.volume,
            'pnl_per_trade': component.pnl_per_trade,
            'pnl_per_share': component.pnl_per_share
        }
class FeeTracker:


    def __init__(self):
        self.fee_schedule = {
            'NYSE': {'maker': -0.0020, 'taker': 0.0030, 'remove': 0.0030},
            'NASDAQ': {'maker': -0.0025, 'taker': 0.0030, 'remove': 0.0030},
            'CBOE': {'maker': -0.0023, 'taker': 0.0028, 'remove': 0.0028},
            'IEX': {'maker': 0.0000, 'taker': 0.0009, 'remove': 0.0009},
            'ARCA': {'maker': -0.0020, 'taker': 0.0030, 'remove': 0.0030}
        }

        self.monthly_volume = defaultdict(int)
        self.monthly_fees = defaultdict(float)
        self.tier_thresholds = {
            'tier1': 0,
            'tier2': 10_000_000,
            'tier3': 50_000_000,
            'tier4': 100_000_000
        }

    def calculate_fee(self, venue: str, order_type: str, volume: int,
                     price: float, is_maker: bool) -> Tuple[float, float]:

        fee_structure = self.fee_schedule.get(venue, {})

        if is_maker:
            rate = fee_structure.get('maker', 0)
        else:
            rate = fee_structure.get('taker', 0)

        tier = self._get_volume_tier(venue)
        tier_discount = 0.0001 * (tier - 1)

        adjusted_rate = rate + tier_discount if rate > 0 else rate - tier_discount

        notional = volume * price

        if adjusted_rate > 0:
            fee = notional * adjusted_rate
            rebate = 0
        else:
            fee = 0
            rebate = -notional * adjusted_rate

        self.monthly_volume[venue] += volume
        self.monthly_fees[venue] += fee - rebate

        return fee, rebate

    def _get_volume_tier(self, venue: str) -> int:

        volume = self.monthly_volume[venue]

        for tier in range(4, 0, -1):
            if volume >= self.tier_thresholds[f'tier{tier}']:
                return tier

        return 1

    def optimize_venue_selection(self, venues: List[str], order_size: int,
                               price: float, can_be_maker: bool) -> str:

        best_venue = venues[0]
        best_cost = float('inf')

        for venue in venues:
            fee, rebate = self.calculate_fee(venue, 'limit', order_size, price, can_be_maker)
            net_cost = fee - rebate

            if net_cost < best_cost:
                best_cost = net_cost
                best_venue = venue

        return best_venue
class LatencyCostModel:


    def __init__(self):
        self.base_decay_rate = 0.0001
        self.volatility_multiplier = 2.0
        self.competition_factor = 1.5

    def calculate_cost(self, fill: Fill, order: Order) -> float:

        latency_ms = fill.latency_us / 1000

        base_cost = self.base_decay_rate * latency_ms * fill.price * fill.quantity

        if hasattr(order, 'market_regime') and order.market_regime == 'volatile':
            base_cost *= self.volatility_multiplier

        if order.strategy == TradingStrategyType.ARBITRAGE:
            base_cost *= self.competition_factor

        actual_cost = min(base_cost, fill.slippage_bps * fill.price * fill.quantity / 10000 * 0.5)

        return actual_cost

    def estimate_latency_alpha(self, avg_latency_us: float,
                             baseline_latency_us: float,
                             daily_volume: int, avg_price: float) -> float:

        latency_improvement = baseline_latency_us - avg_latency_us

        if latency_improvement <= 0:
            return 0

        sensitive_volume = daily_volume * 0.20
        bp_improvement = (latency_improvement / 100) * 1.0

        daily_alpha = sensitive_volume * avg_price * bp_improvement / 10000

        return daily_alpha
class CostAnalysis:


    def __init__(self):
        self.fee_tracker = FeeTracker()
        self.latency_cost_model = LatencyCostModel()
        self.cost_history = []

    def analyze_costs(self, fills: List[Fill], orders: Dict[str, Order]) -> Dict[str, Any]:

        analysis = {
            'total_costs': 0,
            'by_type': {
                'fees': 0,
                'market_impact': 0,
                'latency_cost': 0,
                'opportunity_cost': 0
            },
            'by_venue': defaultdict(float),
            'by_strategy': defaultdict(float),
            'cost_per_share': 0,
            'cost_as_pct_of_volume': 0,
            'potential_savings': {}
        }

        total_volume = 0
        total_notional = 0

        for fill in fills:
            order = orders.get(fill.order_id)
            if not order:
                continue

            net_fee = fill.fees - fill.rebate
            analysis['by_type']['fees'] += net_fee

            impact_cost = fill.market_impact_bps * fill.price * fill.quantity / 10000
            analysis['by_type']['market_impact'] += impact_cost

            latency_cost = self.latency_cost_model.calculate_cost(fill, order)
            analysis['by_type']['latency_cost'] += latency_cost

            total_cost = net_fee + impact_cost + latency_cost
            analysis['total_costs'] += total_cost

            analysis['by_venue'][fill.venue] += total_cost

            analysis['by_strategy'][order.strategy.value] += total_cost

            total_volume += fill.quantity
            total_notional += fill.quantity * fill.price

        if total_volume > 0:
            analysis['cost_per_share'] = analysis['total_costs'] / total_volume

        if total_notional > 0:
            analysis['cost_as_pct_of_volume'] = analysis['total_costs'] / total_notional * 100

        analysis['potential_savings'] = self._identify_savings(analysis, fills)

        return analysis

    def _identify_savings(self, analysis: Dict, fills: List[Fill]) -> Dict[str, float]:

        savings = {}

        current_taker_pct = sum(1 for f in fills if f.fees > 0) / len(fills) if fills else 0
        if current_taker_pct > 0.5:
            potential_maker_fees = analysis['by_type']['fees'] * -0.5
            savings['increase_maker_percentage'] = abs(potential_maker_fees - analysis['by_type']['fees'])

        avg_latency = np.mean([f.latency_us for f in fills]) if fills else 1000
        if avg_latency > 500:
            latency_improvement_pct = (avg_latency - 500) / avg_latency
            savings['reduce_latency_to_500us'] = analysis['by_type']['latency_cost'] * latency_improvement_pct

        venue_costs = analysis['by_venue']
        if venue_costs:
            cheapest_venue_cost = min(venue_costs.values())
            total_venue_cost = sum(venue_costs.values())
            if len(venue_costs) > 1:
                savings['optimize_venue_selection'] = total_venue_cost - (cheapest_venue_cost * len(venue_costs))

        return savings
    
class OperationalRiskManager:

    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_order_rate': 1000,
            'max_error_rate': 0.01,
            'max_latency_ms': 10.0,
            'min_heartbeat_interval': 1.0,
            'circuit_breaker_threshold': 100
        }

        self.order_timestamps = deque(maxlen=10000)
        self.error_counts = defaultdict(int)
        self.latency_measurements = deque(maxlen=1000)
        self.last_heartbeat = defaultdict(float)

        self.system_healthy = True
        self.venue_status = defaultdict(lambda: True)
        self.error_messages = deque(maxlen=100)

    def check_order_rate(self, timestamp: float) -> bool:

        self.order_timestamps.append(timestamp)

        cutoff = timestamp - 1.0
        recent_orders = sum(1 for t in self.order_timestamps if t > cutoff)

        if recent_orders > self.config['max_order_rate']:
            self._log_error(
                'order_rate_exceeded',
                f"Order rate {recent_orders}/s exceeds limit {self.config['max_order_rate']}/s"
            )
            return False

        return True

    def record_latency(self, latency_ms: float, venue: str):

        self.latency_measurements.append({
            'timestamp': time.time(),
            'latency_ms': latency_ms,
            'venue': venue
        })

        if latency_ms > self.config['max_latency_ms']:
            self._log_error(
                'high_latency',
                f"High latency detected for {venue}: {latency_ms:.1f}ms"
            )

        recent_venue_latencies = [
            m['latency_ms'] for m in self.latency_measurements
            if m['venue'] == venue and m['timestamp'] > time.time() - 60
        ]

        if recent_venue_latencies:
            avg_latency = np.mean(recent_venue_latencies)
            if avg_latency > self.config['max_latency_ms']:
                self.venue_status[venue] = False
                self._log_error(
                    'venue_degraded',
                    f"Venue {venue} degraded with avg latency {avg_latency:.1f}ms"
                )

    def record_error(self, error_type: str, message: str, venue: str = None):

        self.error_counts[error_type] += 1

        error_entry = {
            'timestamp': time.time(),
            'type': error_type,
            'message': message,
            'venue': venue
        }

        self.error_messages.append(error_entry)

        total_errors = sum(self.error_counts.values())
        if total_errors > self.config['circuit_breaker_threshold']:
            self.system_healthy = False
            self._log_error(
                'circuit_breaker_triggered',
                f"Circuit breaker triggered after {total_errors} errors"
            )

    def update_heartbeat(self, venue: str):

        self.last_heartbeat[venue] = time.time()

    def check_heartbeats(self) -> Dict[str, bool]:

        current_time = time.time()
        heartbeat_status = {}

        for venue, last_beat in self.last_heartbeat.items():
            time_since_beat = current_time - last_beat
            is_alive = time_since_beat < self.config['min_heartbeat_interval'] * 2

            heartbeat_status[venue] = is_alive

            if not is_alive:
                self.venue_status[venue] = False
                self._log_error(
                    'heartbeat_timeout',
                    f"Heartbeat timeout for {venue}: {time_since_beat:.1f}s"
                )

        return heartbeat_status

    def _log_error(self, error_type: str, message: str):

        logger.error(f"OPERATIONAL RISK: [{error_type}] {message}")

    def get_health_report(self) -> Dict[str, Any]:

        total_orders = len(self.order_timestamps)
        total_errors = sum(self.error_counts.values())
        error_rate = total_errors / max(total_orders, 1)

        if self.latency_measurements:
            avg_latency = np.mean([m['latency_ms'] for m in self.latency_measurements])
            p99_latency = np.percentile([m['latency_ms'] for m in self.latency_measurements], 99)
        else:
            avg_latency = 0
            p99_latency = 0

        return {
            'system_healthy': self.system_healthy,
            'venue_status': dict(self.venue_status),
            'metrics': {
                'error_rate': error_rate,
                'total_errors': total_errors,
                'avg_latency_ms': avg_latency,
                'p99_latency_ms': p99_latency,
                'order_count': total_orders
            },
            'error_breakdown': dict(self.error_counts),
            'recent_errors': list(self.error_messages)[-10:]
        }
class VenueAnalyzer:

    def __init__(self):
        self.venue_metrics = defaultdict(lambda: {
            'orders_sent': 0,
            'orders_filled': 0,
            'total_volume': 0,
            'total_fees': 0,
            'total_rebates': 0,
            'latencies': [],
            'slippages': [],
            'ml_routing_score': 0
        })

    def update_metrics(self, order: Order, fill: Optional[Fill] = None):

        metrics = self.venue_metrics[order.venue]
        metrics['orders_sent'] += 1

        if fill:
            metrics['orders_filled'] += 1
            metrics['total_volume'] += fill.quantity
            metrics['total_fees'] += fill.fees
            metrics['total_rebates'] += fill.rebate
            metrics['latencies'].append(fill.latency_us)
            metrics['slippages'].append(fill.slippage_bps)

            if hasattr(order, 'predicted_latency_us') and order.predicted_latency_us:
                prediction_error = abs(fill.latency_us - order.predicted_latency_us) / order.predicted_latency_us
                routing_score = max(0, 1 - prediction_error)
                n = metrics['orders_filled']
                metrics['ml_routing_score'] = (
                    (metrics['ml_routing_score'] * (n - 1) + routing_score) / n
                )

    def analyze_venue_performance(self) -> Dict[str, Dict]:

        analysis = {}

        for venue, metrics in self.venue_metrics.items():
            fill_rate = metrics['orders_filled'] / max(metrics['orders_sent'], 1)

            if metrics['latencies']:
                avg_latency = np.mean(metrics['latencies'])
                p50_latency = np.percentile(metrics['latencies'], 50)
                p99_latency = np.percentile(metrics['latencies'], 99)
            else:
                avg_latency = p50_latency = p99_latency = 0

            if metrics['slippages']:
                avg_slippage = np.mean(metrics['slippages'])
            else:
                avg_slippage = 0

            net_fees = metrics['total_fees'] - metrics['total_rebates']

            analysis[venue] = {
                'fill_rate': fill_rate,
                'total_volume': metrics['total_volume'],
                'net_fees': net_fees,
                'fee_per_share': net_fees / max(metrics['total_volume'], 1),
                'latency_stats': {
                    'mean': avg_latency,
                    'p50': p50_latency,
                    'p99': p99_latency
                },
                'avg_slippage_bps': avg_slippage,
                'ml_routing_score': metrics['ml_routing_score'],
                'efficiency_score': self._calculate_efficiency_score(metrics, fill_rate, avg_latency, avg_slippage)
            }

        return analysis

    def _calculate_efficiency_score(self, metrics: Dict, fill_rate: float,
                                   avg_latency: float, avg_slippage: float) -> float:

        weights = {
            'fill_rate': 0.3,
            'latency': 0.3,
            'cost': 0.2,
            'slippage': 0.2
        }

        fill_score = fill_rate * 100

        latency_score = max(0, min(100, (2000 - avg_latency) / 18))

        net_fee_per_share = (metrics['total_fees'] - metrics['total_rebates']) / max(metrics['total_volume'], 1)
        cost_score = max(0, min(100, (0.005 + net_fee_per_share) / 0.005 * 100))

        slippage_score = max(0, min(100, (10 - avg_slippage) * 10))

        efficiency_score = (
            weights['fill_rate'] * fill_score +
            weights['latency'] * latency_score +
            weights['cost'] * cost_score +
            weights['slippage'] * slippage_score
        )

        return efficiency_score

    def recommend_venue_allocation(self) -> Dict[str, float]:

        venue_analysis = self.analyze_venue_performance()

        if not venue_analysis:
            return {}

        scores = {
            venue: data['efficiency_score']
            for venue, data in venue_analysis.items()
        }

        total_score = sum(scores.values())
        if total_score == 0:
            return {venue: 1.0 / len(scores) for venue in scores}

        allocations = {}
        for venue, score in scores.items():
            raw_allocation = score / total_score
            allocations[venue] = max(0.1, raw_allocation)

        total_allocation = sum(allocations.values())
        for venue in allocations:
            allocations[venue] /= total_allocation

        return allocations
    
def create_integrated_risk_system() -> Dict[str, Any]:

    risk_manager = RiskManager()
    pnl_attribution = PnLAttribution()
    cost_analysis = CostAnalysis()
    operational_risk = OperationalRiskManager()
    venue_analyzer = VenueAnalyzer()

    return {
        'risk_manager': risk_manager,
        'pnl_attribution': pnl_attribution,
        'cost_analysis': cost_analysis,
        'operational_risk': operational_risk,
        'venue_analyzer': venue_analyzer
    }

def generate_risk_report(risk_system: Dict, fills: List[Fill],
                        orders: Dict[str, Order], current_prices: Dict[str, float]) -> Dict[str, Any]:


    for fill in fills:
        order = orders.get(fill.order_id)
        if order:
            fill.order = order
            risk_system['pnl_attribution'].attribute_fill(
                fill, order,
                {'mid_price': current_prices.get(fill.symbol, fill.price)}
            )
            risk_system['venue_analyzer'].update_metrics(order, fill)

    risk_report = risk_system['risk_manager'].get_risk_report()
    pnl_report = risk_system['pnl_attribution'].get_attribution_report()
    cost_report = risk_system['cost_analysis'].analyze_costs(fills, orders)
    health_report = risk_system['operational_risk'].get_health_report()
    venue_report = risk_system['venue_analyzer'].analyze_venue_performance()

    return {
        'timestamp': time.time(),
        'risk_metrics': risk_report,
        'pnl_attribution': pnl_report,
        'cost_analysis': cost_report,
        'operational_health': health_report,
        'venue_performance': venue_report,
        'recommendations': {
            'venue_allocation': risk_system['venue_analyzer'].recommend_venue_allocation(),
            'cost_savings': cost_report.get('potential_savings', {}),
            'risk_actions': _get_risk_recommendations(risk_report)
        }
    }
def _get_risk_recommendations(risk_report: Dict) -> List[str]:

    recommendations = []

    if risk_report['summary']['current_drawdown'] > 0:
        recommendations.append(f"Reduce position sizes due to ${risk_report['summary']['current_drawdown']:,.0f} drawdown")

    limits = risk_report['limits']
    if 'max_concentration' in limits and limits['max_concentration']['current'] > limits['max_concentration']['threshold'] * 0.8:
        recommendations.append("Diversify positions to reduce concentration risk")

    if risk_report['summary']['active_alerts'] > 5:
        recommendations.append("Review and address multiple active risk alerts")

    return recommendations
if __name__ == "__main__":
    risk_system = create_integrated_risk_system()

    from simulator.trading_simulator import Order, OrderSide, OrderType, TradingStrategy

    test_order = Order(
        order_id="TEST_001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=1000,
        order_type=OrderType.LIMIT,
        price=150.00,
        venue="NYSE",
        strategy=TradingStrategy.MARKET_MAKING
    )

    current_prices = {"AAPL": 150.00}
    is_allowed, reason = risk_system['risk_manager'].check_pre_trade_risk(test_order, current_prices)
    print(f"Order allowed: {is_allowed}, Reason: {reason}")

    report = generate_risk_report(risk_system, [], {}, current_prices)
    print(f"\nRisk Report Summary:")
    print(f"Trading Allowed: {report['risk_metrics']['summary']['trading_allowed']}")
    print(f"Current Drawdown: ${report['risk_metrics']['summary']['current_drawdown']:,.0f}")
    print(f"Active Alerts: {report['risk_metrics']['summary']['active_alerts']}")

    stress_scenarios = {
        'market_crash': {'default': -10},
        'flash_crash': {'default': -5},
        'volatility_spike': {'default': -3}
    }

    stress_results = risk_system['risk_manager'].run_stress_test(stress_scenarios)
    print(f"\nStress Test Results:")
    for scenario, result in stress_results.items():
        print(f"{scenario}: P&L Impact ${result['total_pnl_impact']:,.0f}, "
              f"Breaches Limits: {result['would_breach_limits']}")