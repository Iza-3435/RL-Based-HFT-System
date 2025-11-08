"""Historical data management for backtesting."""

from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

from src.core.logging_config import get_logger
from src.execution.backtesting.config import BacktestConfig

logger = get_logger()


class HistoricalDataManager:
    """Manages historical market data for backtesting."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_cache: Dict[str, List[Dict]] = {}
        self.loaded = False

    async def load_data(self):
        """Load historical data for all symbols."""
        logger.info("Loading historical data", symbols=len(self.config.symbols))

        for symbol in self.config.symbols:
            try:
                data = await self._fetch_historical_data(symbol)
                self.data_cache[symbol] = data
                logger.verbose(f"Loaded {len(data)} ticks for {symbol}")

            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
                raise

        self.loaded = True
        logger.info("Historical data loaded")

    async def _fetch_historical_data(self, symbol: str) -> List[Dict]:
        """Fetch historical data for symbol."""
        return []

    async def get_ticks(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get ticks in date range."""
        if not self.loaded:
            await self.load_data()

        all_ticks = []
        for symbol, ticks in self.data_cache.items():
            filtered = [
                t for t in ticks
                if start_date <= t['timestamp'] <= end_date
            ]
            all_ticks.extend(filtered)

        all_ticks.sort(key=lambda x: x['timestamp'])
        return all_ticks

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol."""
        if symbol not in self.data_cache or not self.data_cache[symbol]:
            return None

        return self.data_cache[symbol][-1].get('close')


class StressDataManager(HistoricalDataManager):
    """Data manager for stress testing scenarios."""

    def __init__(self, config: BacktestConfig, stress_scenarios: List[str]):
        super().__init__(config)
        self.stress_scenarios = stress_scenarios

    async def load_data(self):
        """Load data with stress scenarios applied."""
        await super().load_data()

        for scenario in self.stress_scenarios:
            self._apply_stress_scenario(scenario)

        logger.info("Stress scenarios applied", scenarios=len(self.stress_scenarios))

    def _apply_stress_scenario(self, scenario: str):
        """Apply stress scenario to data."""
        if scenario == 'flash_crash':
            self._simulate_flash_crash()
        elif scenario == 'high_volatility':
            self._simulate_high_volatility()
        elif scenario == 'liquidity_crisis':
            self._simulate_liquidity_crisis()

    def _simulate_flash_crash(self):
        """Simulate sudden price drop."""
        for symbol in self.data_cache:
            data = self.data_cache[symbol]
            crash_point = len(data) // 2

            for i in range(crash_point, min(crash_point + 100, len(data))):
                data[i]['close'] *= 0.9

    def _simulate_high_volatility(self):
        """Simulate increased volatility."""
        for symbol in self.data_cache:
            data = self.data_cache[symbol]

            for tick in data:
                tick['close'] *= (1 + np.random.normal(0, 0.05))

    def _simulate_liquidity_crisis(self):
        """Simulate reduced liquidity."""
        for symbol in self.data_cache:
            data = self.data_cache[symbol]

            for tick in data:
                tick['volume'] = int(tick.get('volume', 1000) * 0.3)
