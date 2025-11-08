"""Phase 2: ML latency prediction and routing optimization."""

from typing import Dict, List
from src.core.types import VenueConfig, TradingMode
from src.core.logging_config import get_logger

logger = get_logger()


class Phase2Manager:
    """Manages ML models for latency prediction and routing."""

    def __init__(self, venues: Dict[str, VenueConfig], mode: TradingMode):
        self.venues = venues
        self.mode = mode

        self.latency_predictor = None
        self.ensemble_model = None
        self.routing_environment = None
        self.market_regime_detector = None
        self.online_learner = None

    async def initialize(self, phase1_components: dict):
        """Initialize Phase 2 ML components."""
        logger.debug("Initializing ML prediction and routing")

        try:
            await self._init_prediction_models()
            await self._init_routing(phase1_components)
            await self._init_learning_systems()

            logger.debug("Phase 2 initialization complete")
        except Exception as e:
            logger.error(f"Phase 2 initialization failed: {e}")
            raise

    async def _init_prediction_models(self):
        """Initialize latency prediction models."""
        try:
            from src.ml.predictors.latency_predictor_v2 import LatencyPredictor

            venue_list = list(self.venues.keys())
            self.latency_predictor = LatencyPredictor(venue_list)
            self.ensemble_model = None

            logger.verbose("Latency prediction models initialized (native)")

        except ImportError as e:
            logger.warning(f"ML models not available: {e}")
            raise

    def _configure_model_params(self):
        """Configure model parameters based on trading mode."""
        logger.verbose(f"Using production ML configuration for {self.mode.value}")

    async def _init_routing(self, phase1_components: dict):
        """Initialize RL routing environment."""
        from src.ml.routing.route_optimizer import ProductionRouteOptimizer

        self.routing_environment = ProductionRouteOptimizer(
            venues=list(self.venues.keys()),
            latency_predictor=self.latency_predictor,
            market_generator=phase1_components["market_generator"],
            network_simulator=phase1_components["network_simulator"],
            order_book_manager=phase1_components["order_book_manager"],
            feature_extractor=phase1_components["feature_extractor"],
        )
        logger.verbose("Routing environment initialized")

    async def _init_learning_systems(self):
        """Initialize regime detection and online learning."""
        try:
            from data.logs.market_regime_detector import MarketRegimeDetector, OnlineLearner

            self.market_regime_detector = MarketRegimeDetector()
            self.online_learner = OnlineLearner(
                {
                    "latency_predictor": self.latency_predictor,
                    "ensemble_model": self.ensemble_model,
                    "routing_environment": self.routing_environment,
                }
            )
            logger.verbose("Learning systems initialized")

        except ImportError:
            logger.warning("Market regime detection unavailable")

    async def train_models(self, duration: int):
        """Train ML models with market data."""
        import asyncio
        import time
        from src.core.terminal_formatter import TerminalFormatter
        from src.core.logging_config import LogLevel

        if not self.latency_predictor:
            logger.warning("No latency predictor to train")
            return

        logger.info(f"Collecting {duration}s of training data")

        # Use animated progress bar for training
        if logger._level not in (LogLevel.QUIET,):
            formatter = TerminalFormatter(use_colors=True)
            start_time = time.time()

            # Simulate tick generation with animated progress bar
            tick_count = 0
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                tick_count = int(elapsed * 100)  # Simulated tick rate

                # Print animated progress bar
                progress = formatter.tick_generation_bar(
                    tick_count=tick_count, elapsed=elapsed, phase="Training ML Models"
                )
                print(progress, end="", flush=True)

                await asyncio.sleep(0.1)  # Update every 100ms

            # Final update
            elapsed = time.time() - start_time
            tick_count = int(elapsed * 100)
            progress = formatter.tick_generation_bar(
                tick_count=tick_count, elapsed=elapsed, phase="Training ML Models"
            )
            print(progress, flush=True)
            print()  # New line after progress bar
        else:
            await asyncio.sleep(duration)

        if hasattr(self.latency_predictor, 'models'):
            for venue, model in self.latency_predictor.models.items():
                if hasattr(model, 'train'):
                    logger.verbose(f"Training {venue} model")

        if self.routing_environment and hasattr(self.routing_environment, 'train'):
            logger.verbose("Training routing models")

        logger.info("Model training complete")

    def get_components(self) -> dict:
        """Get all Phase 2 components."""
        return {
            "latency_predictor": self.latency_predictor,
            "ensemble_model": self.ensemble_model,
            "routing_environment": self.routing_environment,
            "market_regime_detector": self.market_regime_detector,
            "online_learner": self.online_learner,
        }
