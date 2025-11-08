"""Risk limit management."""

from typing import Dict, Tuple, Optional

from src.risk.types import RiskMetric, RiskLimit
from src.trading.types import Order, OrderSide


class LimitManager:
    """Manage and check risk limits."""

    def __init__(self, config: Dict):
        self.config = config
        self.limits = self._initialize_limits()

    def _initialize_limits(self) -> Dict[str, RiskLimit]:
        """Initialize risk limit tracking."""
        return {
            'max_position_size': RiskLimit(
                metric=RiskMetric.POSITION,
                limit_type='hard',
                threshold=self.config['position_limits']['max_position_size']
            ),
            'max_gross_exposure': RiskLimit(
                metric=RiskMetric.EXPOSURE,
                limit_type='hard',
                threshold=self.config['exposure_limits']['max_gross_exposure']
            ),
            'max_concentration': RiskLimit(
                metric=RiskMetric.CONCENTRATION,
                limit_type='soft',
                threshold=self.config['concentration_limits']['max_symbol_concentration']
            ),
            'max_drawdown': RiskLimit(
                metric=RiskMetric.DRAWDOWN,
                limit_type='hard',
                threshold=self.config['drawdown_limits']['hard_drawdown_limit'],
                action='stop'
            ),
            'var_limit': RiskLimit(
                metric=RiskMetric.VAR,
                limit_type='soft',
                threshold=self.config['var_limits']['var_limit']
            )
        }

    def check_position_limit(
        self,
        order: Order,
        current_position: float
    ) -> Tuple[bool, Optional[str]]:
        """Check position size limit."""
        new_position = current_position
        if order.side == OrderSide.BUY:
            new_position += order.quantity
        else:
            new_position -= order.quantity

        limit = self.limits['max_position_size']
        if abs(new_position) > limit.threshold:
            return False, f"Position limit exceeded: {abs(new_position)} > {limit.threshold}"

        return True, None

    def check_exposure_limit(
        self,
        order_value: float,
        current_gross_exposure: float
    ) -> Tuple[bool, Optional[str]]:
        """Check exposure limit."""
        new_exposure = current_gross_exposure + order_value
        limit = self.limits['max_gross_exposure']

        if new_exposure > limit.threshold:
            return False, f"Exposure limit exceeded: ${new_exposure:,.0f}"

        return True, None

    def check_concentration_limit(
        self,
        symbol_exposure: float,
        total_exposure: float
    ) -> Tuple[bool, Optional[str]]:
        """Check concentration limit."""
        if total_exposure == 0:
            return True, None

        concentration = abs(symbol_exposure) / total_exposure
        limit = self.limits['max_concentration']

        if concentration > limit.threshold:
            return False, f"Concentration limit exceeded: {concentration:.1%}"

        return True, None

    def is_risk_reducing(self, order: Order, current_position: float) -> bool:
        """Check if order reduces risk."""
        if current_position == 0:
            return False

        if current_position > 0 and order.side == OrderSide.SELL:
            return True
        if current_position < 0 and order.side == OrderSide.BUY:
            return True

        return False
