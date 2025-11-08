"""Production routing optimization with clean interface."""

from typing import Dict, List, Optional
from src.core.logging_config import get_logger
from .routing_manager import RoutingEnvironment

logger = get_logger()


class ProductionRouteOptimizer:
    """Production wrapper for routing optimization."""

    def __init__(
        self,
        venues: List[str],
        latency_predictor,
        market_generator,
        network_simulator,
        order_book_manager,
        feature_extractor,
    ):
        self.venues = venues

        self._environment = RoutingEnvironment(
            latency_predictor=latency_predictor,
            market_generator=market_generator,
            network_simulator=network_simulator,
            order_book_manager=order_book_manager,
            feature_extractor=feature_extractor,
            venue_list=venues,
        )

        logger.verbose("Route optimizer initialized", venues=len(venues))

    def get_best_venue(self, tick_data: Dict, prediction: Dict) -> Optional[str]:
        """Get optimal venue for execution."""
        try:
            return self._environment.get_best_venue(tick_data, prediction)

        except Exception as e:
            logger.debug(f"Routing error: {e}")
            return self.venues[0] if self.venues else None

    def optimize(self, symbol: str, features: Dict) -> str:
        """Optimize routing for symbol."""
        try:
            if hasattr(self._environment, "optimize_route"):
                return self._environment.optimize_route(symbol, features)
        except:
            pass

        return self.venues[0] if self.venues else "NYSE"
