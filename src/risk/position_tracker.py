"""Position and exposure tracking."""

from typing import Dict
from collections import defaultdict

from src.core.logging_config import get_logger
from src.trading.types import Fill, OrderSide

logger = get_logger()


class PositionTracker:
    """Track positions and exposures across strategies."""

    def __init__(self):
        self.positions = defaultdict(lambda: defaultdict(float))
        self.exposures = defaultdict(float)
        self.market_prices = {}

    def update_position(
        self,
        fill: Fill,
        strategy: str,
        current_prices: Dict[str, float]
    ):
        """Update positions and exposures after fill."""
        if fill.side == OrderSide.BUY:
            self.positions[strategy][fill.symbol] += fill.quantity
        else:
            self.positions[strategy][fill.symbol] -= fill.quantity

        self.market_prices[fill.symbol] = fill.price
        self._update_exposures(current_prices)

    def _update_exposures(self, current_prices: Dict[str, float]):
        """Update dollar exposures."""
        self.exposures.clear()

        for strategy_positions in self.positions.values():
            for symbol, quantity in strategy_positions.items():
                price = current_prices.get(
                    symbol,
                    self.market_prices.get(symbol, 0)
                )
                if symbol not in self.exposures:
                    self.exposures[symbol] = 0
                self.exposures[symbol] += quantity * price

    def get_gross_exposure(self) -> float:
        """Calculate gross exposure."""
        return sum(abs(exp) for exp in self.exposures.values())

    def get_net_exposure(self) -> float:
        """Calculate net exposure."""
        return sum(self.exposures.values())

    def get_position(self, strategy: str, symbol: str) -> float:
        """Get position for strategy and symbol."""
        return self.positions[strategy][symbol]

    def get_all_positions(self) -> Dict[str, Dict[str, float]]:
        """Get all positions by symbol and strategy."""
        all_positions = defaultdict(dict)

        for strategy, positions in self.positions.items():
            for symbol, quantity in positions.items():
                if quantity != 0:
                    all_positions[symbol][strategy] = quantity

        return dict(all_positions)

    def get_symbol_concentration(self, symbol: str) -> float:
        """Get concentration for a symbol."""
        gross_exposure = self.get_gross_exposure()
        if gross_exposure == 0:
            return 0

        return abs(self.exposures.get(symbol, 0)) / gross_exposure
