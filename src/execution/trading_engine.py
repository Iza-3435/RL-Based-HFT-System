"""Production trading engine with clean interface."""

from typing import Dict, List, Optional
from src.core.logging_config import get_logger
from src.core.types import OrderRequest, Trade
from simulator.trading_simulator_integration import create_enhanced_trading_simulator

logger = get_logger()


class ProductionTradingEngine:
    """Production wrapper for trading execution."""

    def __init__(self, symbols: List[str], venues: List[str]):
        self.symbols = symbols
        self.venues = venues

        self._simulator = create_enhanced_trading_simulator(
            symbols=symbols,
            venues=venues,
            config={
                "enable_latency_simulation": True,
                "venue_optimization": True,
                "strategy_latency_optimization": True,
            },
        )

        # Expose execution_engine for cost model integration
        if hasattr(self._simulator, 'execution_engine'):
            self.execution_engine = self._simulator.execution_engine

        logger.verbose("Trading engine initialized", symbols=len(symbols), venues=len(venues))

    async def execute(self, order: OrderRequest) -> Optional[Trade]:
        """Execute trade order."""
        try:
            result = await self._simulator.execute_order(order)

            if result:
                return Trade(
                    id=result.get("id", ""),
                    symbol=result.get("symbol", order.symbol),
                    side=order.side,
                    quantity=result.get("quantity", order.quantity),
                    price=result.get("price", 0.0),
                    venue=result.get("venue", order.venue or ""),
                    timestamp=result.get("timestamp"),
                    latency_us=result.get("latency_us", 0),
                    fees=result.get("fees", 0.0),
                )

        except Exception as e:
            logger.error(f"Trade execution error: {e}")

        return None

    def get_metrics(self) -> Dict:
        """Get trading metrics."""
        try:
            if hasattr(self._simulator, "get_metrics"):
                return self._simulator.get_metrics()
        except:
            pass

        return {}
