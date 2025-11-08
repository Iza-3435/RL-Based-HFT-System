"""Main risk management orchestrator."""

import time
from typing import Dict, Tuple, Optional, Any, Set
from collections import defaultdict

from src.core.logging_config import get_logger
from src.risk.types import RiskMetric, RiskLevel, RiskLimit, RiskAlert
from src.risk.limits import LimitManager
from src.risk.position_tracker import PositionTracker
from src.risk.var_calculator import VaRCalculator
from src.trading.types import Order, Fill

logger = get_logger()


class RiskManager:
    """Comprehensive risk management system for HFT trading."""

    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()

        self.limit_manager = LimitManager(self.config)
        self.position_tracker = PositionTracker()
        self.var_calculator = VaRCalculator()

        self.realized_pnl = defaultdict(float)
        self.unrealized_pnl = defaultdict(float)
        self.high_water_mark = 0
        self.current_drawdown = 0

        self.active_alerts = []
        self.alert_history = []
        self.trading_allowed = True
        self.restricted_symbols: Set[str] = set()

        logger.info("Risk manager initialized")

    def _get_default_config(self) -> Dict:
        """Default risk management configuration."""
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
            }
        }

    def check_pre_trade_risk(
        self,
        order: Order,
        current_prices: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """Pre-trade risk checks."""
        if not self.trading_allowed:
            return False, "Trading halted due to risk limits"

        if order.symbol in self.restricted_symbols:
            return False, f"Symbol {order.symbol} is restricted"

        current_position = self.position_tracker.get_position(
            order.strategy.value, order.symbol
        )

        is_allowed, reason = self.limit_manager.check_position_limit(
            order, current_position
        )
        if not is_allowed:
            return False, reason

        price = current_prices.get(order.symbol, order.price)
        order_value = order.quantity * price
        current_gross_exposure = self.position_tracker.get_gross_exposure()

        is_allowed, reason = self.limit_manager.check_exposure_limit(
            order_value, current_gross_exposure
        )
        if not is_allowed:
            return False, reason

        symbol_exposure = self.position_tracker.exposures.get(order.symbol, 0)
        total_exposure = self.position_tracker.get_gross_exposure()

        is_allowed, reason = self.limit_manager.check_concentration_limit(
            symbol_exposure + order_value, total_exposure + order_value
        )
        if not is_allowed:
            return False, reason

        if self.current_drawdown > self.config['drawdown_limits']['soft_drawdown_limit']:
            if not self.limit_manager.is_risk_reducing(order, current_position):
                return False, "Only risk-reducing trades allowed during drawdown"

        return True, None

    def update_position(self, fill: Fill, current_prices: Dict[str, float]):
        """Update positions and risk metrics after fill."""
        strategy = fill.order.strategy.value if hasattr(fill, 'order') else 'unknown'

        self.position_tracker.update_position(fill, strategy, current_prices)
        self._check_risk_limits(current_prices)
        self._record_risk_snapshot(current_prices)

    def _check_risk_limits(self, current_prices: Dict[str, float]):
        """Check all risk limits and trigger alerts if needed."""
        for strategy, positions in self.position_tracker.positions.items():
            for symbol, quantity in positions.items():
                limit = self.limit_manager.limits['max_position_size']
                if abs(quantity) > limit.threshold:
                    self._trigger_risk_alert(
                        RiskMetric.POSITION,
                        RiskLevel.HIGH,
                        f"Position limit breached for {symbol}: {abs(quantity)}",
                        abs(quantity),
                        limit.threshold
                    )

        gross_exposure = self.position_tracker.get_gross_exposure()
        limit = self.limit_manager.limits['max_gross_exposure']
        limit.current_value = gross_exposure

        if gross_exposure > limit.threshold:
            self._trigger_risk_alert(
                RiskMetric.EXPOSURE,
                RiskLevel.CRITICAL,
                f"Gross exposure limit breached: ${gross_exposure:,.0f}",
                gross_exposure,
                limit.threshold
            )
            self._take_risk_action('reduce_positions')

        self._check_drawdown_limits()

    def _check_drawdown_limits(self):
        """Check drawdown limits and trigger circuit breakers."""
        total_pnl = sum(self.realized_pnl.values()) + sum(self.unrealized_pnl.values())

        if total_pnl > self.high_water_mark:
            self.high_water_mark = total_pnl
            self.current_drawdown = 0
        else:
            self.current_drawdown = self.high_water_mark - total_pnl

        limit = self.limit_manager.limits['max_drawdown']
        limit.current_value = self.current_drawdown

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

    def _trigger_risk_alert(
        self,
        metric: RiskMetric,
        level: RiskLevel,
        message: str,
        current_value: float,
        limit_value: float
    ):
        """Create and process risk alert."""
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

    def _take_risk_action(self, action: str):
        """Take risk management action."""
        if action == 'stop_trading':
            self.trading_allowed = False
            logger.critical("TRADING HALTED due to risk limits")

        elif action == 'reduce_positions':
            logger.warning("Position reduction required")

        elif action == 'reduce_risk':
            logger.warning("Risk reduction mode activated")

    def _record_risk_snapshot(self, current_prices: Dict[str, float]):
        """Record current risk metrics."""
        self.var_calculator.record_risk_snapshot(
            gross_exposure=self.position_tracker.get_gross_exposure(),
            net_exposure=self.position_tracker.get_net_exposure(),
            total_pnl=sum(self.realized_pnl.values()) + sum(self.unrealized_pnl.values()),
            drawdown=self.current_drawdown,
            position_count=sum(len(p) for p in self.position_tracker.positions.values()),
            active_alerts=len(self.active_alerts)
        )

    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        return {
            'summary': {
                'trading_allowed': self.trading_allowed,
                'gross_exposure': self.position_tracker.get_gross_exposure(),
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
                for name, limit in self.limit_manager.limits.items()
            },
            'positions': self.position_tracker.get_all_positions(),
            'var_metrics': (
                self.var_calculator.var_history[-1]
                if self.var_calculator.var_history else {}
            ),
            'recent_alerts': self.alert_history[-10:] if self.alert_history else []
        }
