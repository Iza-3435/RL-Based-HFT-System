"""Phase 3: Trading execution and risk management."""

from datetime import datetime, timedelta
from typing import Dict, List
from src.core.types import VenueConfig
from src.core.logging_config import get_logger

logger = get_logger()


class Phase3Manager:
    """Manages trading execution and risk management."""

    def __init__(self, symbols: List[str], venues: Dict[str, VenueConfig]):
        self.symbols = symbols
        self.venues = venues

        self.trading_simulator = None
        self.risk_manager = None
        self.pnl_attribution = None
        self.backtesting_engine = None
        self.cost_model = None

    async def initialize(self):
        """Initialize Phase 3 trading components."""
        logger.debug("Initializing trading execution and risk management")

        try:
            await self._init_trading_simulator()
            await self._init_risk_management()
            await self._init_backtesting()

            logger.debug("Phase 3 initialization complete")
        except Exception as e:
            logger.error(f"Phase 3 initialization failed: {e}")
            raise

    async def _init_trading_simulator(self):
        """Initialize trading simulator with enhanced features."""
        from src.execution.trading_engine import ProductionTradingEngine

        self.trading_simulator = ProductionTradingEngine(
            symbols=self.symbols, venues=list(self.venues.keys())
        )

        await self._integrate_cost_model()
        logger.verbose("Trading simulator initialized")

    async def _integrate_cost_model(self):
        """Integrate enhanced cost modeling if available."""
        try:
            from simulator.enhanced_execution_cost_model import integrate_enhanced_cost_model

            self.trading_simulator = integrate_enhanced_cost_model(self.trading_simulator)
            self.cost_model = self.trading_simulator.enhanced_impact_model
            logger.verbose("Enhanced cost modeling integrated")

        except ImportError:
            logger.warning("Enhanced cost model unavailable, using basic model")

    async def _init_risk_management(self):
        """Initialize risk management system."""
        from src.risk.risk_manager import RiskManager

        self.risk_manager = RiskManager()
        self.pnl_attribution = None

        logger.verbose("Risk management initialized")

    async def _init_backtesting(self):
        """Initialize backtesting engine."""
        from simulator.backtesting_framework import BacktestingEngine, BacktestConfig

        config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            symbols=self.symbols,
            venues=list(self.venues.keys()),
            initial_capital=1_000_000,
        )
        self.backtesting_engine = BacktestingEngine(config)
        logger.verbose("Backtesting engine initialized")

    def get_components(self) -> dict:
        """Get all Phase 3 components."""
        return {
            "trading_simulator": self.trading_simulator,
            "risk_manager": self.risk_manager,
            "pnl_attribution": self.pnl_attribution,
            "backtesting_engine": self.backtesting_engine,
            "cost_model": self.cost_model,
        }
