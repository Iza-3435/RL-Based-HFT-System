"""Base trading strategy."""

from typing import Dict, List
from collections import defaultdict

from src.trading.types import TradingStrategyType, Order, Fill, Position


class TradingStrategy:
    """Base class for trading strategies."""

    def __init__(self, strategy_type: TradingStrategyType, params: Dict = None):
        self.strategy_type = strategy_type
        self.params = params or {}
        self.positions = defaultdict(Position)
        self.orders = []
        self.fills = []
        self.pnl_history = []

    async def generate_signals(
        self,
        market_data: Dict,
        ml_predictions: Dict
    ) -> List[Order]:
        """Generate trading signals."""
        raise NotImplementedError

    def update_positions(self, fill: Fill, current_prices: Dict):
        """Update positions with new fill."""
        position = self.positions[fill.symbol]
        position.update_position(fill, current_prices[fill.symbol])
        self.fills.append(fill)

    def get_total_pnl(self) -> Dict[str, float]:
        """Calculate total P&L across all positions."""
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_volume = sum(pos.total_volume for pos in self.positions.values())

        total_fees = sum(fill.fees - fill.rebate for fill in self.fills)

        return {
            'realized_pnl': total_realized,
            'unrealized_pnl': total_unrealized,
            'total_pnl': total_realized + total_unrealized - total_fees,
            'fees_paid': total_fees,
            'total_volume': total_volume
        }
