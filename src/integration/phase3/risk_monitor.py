"""Risk monitoring and P&L tracking."""

import logging
import numpy as np
from typing import Dict, Any
from .config import get_risk_limits

logger = logging.getLogger(__name__)


class RiskMonitor:
    """Monitor risk and track P&L."""

    def __init__(self, execution_pipeline: Any):
        self.execution_pipeline = execution_pipeline
        self.total_pnl = 0.0
        self.current_positions = {}
        self.risk_limits = get_risk_limits()

    async def update_pnl_and_risk(self, tick: Any, simulation_results: Dict) -> None:
        """Update P&L and check risk limits."""
        try:
            current_prices = {tick.symbol: getattr(tick, "mid_price", 100.0)}

            simulation_results["pnl_history"].append(
                {
                    "timestamp": tick.timestamp,
                    "total_pnl": self.total_pnl,
                    "unrealized_pnl": 0,
                    "realized_pnl": self.total_pnl,
                }
            )

            if self.total_pnl < self.risk_limits["max_loss"]:
                logger.critical("Risk limit hit: Trading halted")
                self.execution_pipeline.halt_trading = True

                simulation_results["risk_events"].append(
                    {
                        "timestamp": tick.timestamp,
                        "level": "CRITICAL",
                        "action": "EMERGENCY_HALT",
                        "reason": f"P&L loss limit exceeded: ${self.total_pnl:.2f}",
                        "current_value": self.total_pnl,
                        "threshold": self.risk_limits["max_loss"],
                    }
                )

            elif len(simulation_results["pnl_history"]) > 10:
                recent_pnl = [p["total_pnl"] for p in simulation_results["pnl_history"][-10:]]
                pnl_volatility = np.std(recent_pnl)

                if pnl_volatility > self.risk_limits["pnl_volatility_threshold"]:
                    simulation_results["risk_events"].append(
                        {
                            "timestamp": tick.timestamp,
                            "level": "HIGH",
                            "metric": "PNL_VOLATILITY",
                            "message": f"High P&L volatility detected: {pnl_volatility:.2f}",
                            "current_value": pnl_volatility,
                            "threshold": self.risk_limits["pnl_volatility_threshold"],
                        }
                    )
                    logger.warning(f"Risk alert: High P&L volatility: {pnl_volatility:.2f}")

        except Exception as e:
            logger.debug(f"P&L update error: {e}")

    def calculate_max_drawdown(self, pnl_history: list) -> float:
        """Calculate maximum drawdown."""
        if not pnl_history:
            return 0

        pnl_values = [p["total_pnl"] for p in pnl_history]
        peak = pnl_values[0]
        max_drawdown = 0

        for pnl in pnl_values:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def calculate_risk_adjusted_return(self, pnl_history: list) -> float:
        """Calculate Sharpe-like risk adjusted return."""
        if len(pnl_history) < 2:
            return 0

        pnl_values = [p["total_pnl"] for p in pnl_history]
        returns = np.diff(pnl_values)

        if np.std(returns) == 0:
            return 0

        return np.mean(returns) / np.std(returns) * np.sqrt(252)
