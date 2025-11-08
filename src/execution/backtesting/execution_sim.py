"""Trade execution simulation for backtesting."""

from typing import Dict, Optional
import numpy as np

from src.core.logging_config import get_logger
from src.execution.backtesting.config import BacktestConfig

logger = get_logger()


class BacktestExecutionSimulator:
    """Simulates trade execution in backtest."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    async def execute(self, signal: Dict, tick: Dict, current_capital: float) -> Optional[Dict]:
        """Simulate trade execution."""
        try:
            if not self._validate_signal(signal, current_capital):
                return None

            fill_price = self._calculate_fill_price(signal, tick)
            commission = self._calculate_commission(signal, fill_price)
            slippage = self._calculate_slippage(signal, tick, fill_price)

            total_cost = fill_price * signal['quantity'] + commission + slippage

            if signal['side'] == 'buy' and total_cost > current_capital:
                logger.debug(f"Insufficient capital for {signal['symbol']}")
                return None

            pnl = self._calculate_pnl(signal, fill_price, commission, slippage)

            return {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'quantity': signal['quantity'],
                'fill_price': fill_price,
                'commission': commission,
                'slippage': slippage,
                'pnl': pnl,
                'timestamp': tick['timestamp']
            }

        except Exception as e:
            logger.error(f"Execution simulation error: {e}")
            return None

    def _validate_signal(self, signal: Dict, current_capital: float) -> bool:
        """Validate trading signal."""
        if signal['quantity'] <= 0:
            return False

        if signal['quantity'] > self.config.max_position_size:
            logger.debug(f"Signal exceeds max position size")
            return False

        return True

    def _calculate_fill_price(self, signal: Dict, tick: Dict) -> float:
        """Calculate fill price with slippage."""
        base_price = tick.get('close', 100.0)

        market_impact = self._calculate_market_impact(signal, tick)
        fill_price = base_price * (1 + market_impact)

        return fill_price

    def _calculate_market_impact(self, signal: Dict, tick: Dict) -> float:
        """Calculate market impact."""
        volume = tick.get('volume', 100000)
        participation_rate = signal['quantity'] / volume

        impact = participation_rate * 0.01

        if signal['side'] == 'buy':
            return impact
        else:
            return -impact

    def _calculate_commission(self, signal: Dict, fill_price: float) -> float:
        """Calculate commission."""
        trade_value = signal['quantity'] * fill_price
        return trade_value * (self.config.commission_bps / 10000)

    def _calculate_slippage(self, signal: Dict, tick: Dict, fill_price: float) -> float:
        """Calculate slippage."""
        trade_value = signal['quantity'] * fill_price
        return trade_value * (self.config.slippage_bps / 10000)

    def _calculate_pnl(self, signal: Dict, fill_price: float, commission: float, slippage: float) -> float:
        """Calculate P&L for trade."""
        gross_value = signal['quantity'] * fill_price

        if signal['side'] == 'buy':
            return -(gross_value + commission + slippage)
        else:
            return gross_value - commission - slippage
