"""Production market data generator with clean interface."""

from typing import List, Dict, Optional
from src.core.logging_config import get_logger
from src.core.types import VenueConfig, MarketTick
from data.real_market_data_generator import UltraRealisticMarketDataGenerator

logger = get_logger()


class ProductionMarketDataGenerator:
    """Production wrapper for market data generation."""

    def __init__(
        self, symbols: List[str], venues: Dict[str, VenueConfig], mode: str = "production"
    ):
        self.symbols = symbols
        self.venues = venues
        self.mode = mode

        self._generator = UltraRealisticMarketDataGenerator(symbols, mode=mode)
        self._generator.venues = venues

        logger.verbose("Market data generator initialized", symbols=len(symbols), mode=mode)

    async def get_next_tick(self) -> Optional[MarketTick]:
        """Get next market data tick."""
        try:
            tick_data = await self._generator.get_next_tick()
            if not tick_data:
                return None

            return MarketTick(
                symbol=tick_data.get("symbol", "UNKNOWN"),
                timestamp=tick_data.get("timestamp"),
                bid=tick_data.get("bid_price", 0.0),
                ask=tick_data.get("ask_price", 0.0),
                bid_size=tick_data.get("bid_size", 0),
                ask_size=tick_data.get("ask_size", 0),
                last_price=tick_data.get("last_price", 0.0),
                volume=tick_data.get("volume", 0),
                venue=tick_data.get("venue", "UNKNOWN"),
            )

        except Exception as e:
            logger.debug(f"Tick generation error: {e}")
            return None

    def start(self):
        """Start market data generation."""
        if hasattr(self._generator, "start"):
            self._generator.start()

    def stop(self):
        """Stop market data generation."""
        if hasattr(self._generator, "stop"):
            self._generator.stop()
