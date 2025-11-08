"""VaR calculation and stress testing."""

from typing import Dict, List
from collections import deque
import numpy as np

from src.core.logging_config import get_logger
from src.risk.types import RiskMetric, RiskLevel, RiskLimit

logger = get_logger()


class VaRCalculator:
    """Calculate Value at Risk and run stress tests."""

    def __init__(self):
        self.risk_history = deque(maxlen=10000)
        self.var_history = deque(maxlen=1000)

    def calculate_var(
        self,
        confidence_level: float = 0.95,
        horizon_days: int = 1
    ) -> Dict[str, float]:
        """Calculate Value at Risk using historical simulation."""
        if len(self.risk_history) < 100:
            return {
                'var': 0,
                'expected_shortfall': 0,
                'confidence_level': confidence_level
            }

        pnl_changes = []
        for i in range(1, len(self.risk_history)):
            prev_pnl = self.risk_history[i-1].get('total_pnl', 0)
            curr_pnl = self.risk_history[i].get('total_pnl', 0)
            pnl_changes.append(curr_pnl - prev_pnl)

        if not pnl_changes:
            return {
                'var': 0,
                'expected_shortfall': 0,
                'confidence_level': confidence_level
            }

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

        self.var_history.append(result)
        return result

    def run_stress_test(
        self,
        scenarios: Dict[str, Dict],
        positions: Dict[str, Dict[str, float]],
        market_prices: Dict[str, float],
        max_drawdown_limit: float
    ) -> Dict[str, Dict]:
        """Run stress tests on current positions."""
        results = {}

        for scenario_name, shocks in scenarios.items():
            scenario_pnl = 0

            for symbol, positions_by_strategy in positions.items():
                shock = shocks.get(symbol, shocks.get('default', 0))
                current_price = market_prices.get(symbol, 100)
                shocked_price = current_price * (1 + shock / 100)

                total_position = sum(positions_by_strategy.values())
                position_pnl = total_position * (shocked_price - current_price)
                scenario_pnl += position_pnl

            results[scenario_name] = {
                'total_pnl_impact': scenario_pnl,
                'pct_of_capital': scenario_pnl / 5000000 * 100,
                'would_breach_limits': abs(scenario_pnl) > max_drawdown_limit
            }

        return results

    def record_risk_snapshot(
        self,
        gross_exposure: float,
        net_exposure: float,
        total_pnl: float,
        drawdown: float,
        position_count: int,
        active_alerts: int
    ):
        """Record current risk metrics."""
        snapshot = {
            'timestamp': np.datetime64('now').item().timestamp(),
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'total_pnl': total_pnl,
            'drawdown': drawdown,
            'position_count': position_count,
            'active_alerts': active_alerts
        }

        self.risk_history.append(snapshot)
