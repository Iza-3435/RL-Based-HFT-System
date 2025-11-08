"""Component initialization for all phases."""

import logging
from typing import Dict, List, Any, Optional
from data.real_market_data_generator import UltraRealisticMarketDataGenerator
from integration.real_network_system import RealNetworkLatencySimulator
from .config import get_venue_configs, get_training_config

logger = logging.getLogger(__name__)


class ComponentInitializer:
    """Initialize components for all phases."""

    def __init__(self, symbols: List[str], venues: Dict, mode: str = "production"):
        self.symbols = symbols
        self.venues = venues
        self.mode = mode
        self.config = get_training_config(mode)

    async def initialize_phase1(self) -> Dict[str, Any]:
        """Initialize Phase 1: Market Data & Network Infrastructure."""
        logger.info(" Initializing Phase 1: Market Data & Network Infrastructure")

        from simulator.network_latency_simulator import NetworkLatencySimulator
        from simulator.order_book_manager import OrderBookManager
        from data.feature_extractor import FeatureExtractor
        from simulator.performance_tracker import PerformanceTracker

        market_generator = UltraRealisticMarketDataGenerator(self.symbols, mode=self.mode)
        market_generator.venues = self.venues

        logger.info(f" TICK GENERATION ACTIVE:")
        logger.info(f"   Mode: {market_generator.mode}")
        logger.debug(f"   Target Rate: {market_generator.target_ticks_per_minute} ticks/min")
        logger.info(f"   Base Interval: {market_generator.base_update_interval:.3f}s")
        logger.info(f"   Optimized Symbols: {len(market_generator.symbols)}")

        network_simulator = RealNetworkLatencySimulator()
        order_book_manager = OrderBookManager(self.symbols, self.venues)
        feature_extractor = FeatureExtractor(self.symbols, self.venues)
        performance_tracker = PerformanceTracker()

        logger.info(" Phase 1 initialization complete")

        return {
            "market_generator": market_generator,
            "network_simulator": network_simulator,
            "order_book_manager": order_book_manager,
            "feature_extractor": feature_extractor,
            "performance_tracker": performance_tracker,
        }

    async def initialize_phase2(self, phase1_components: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize Phase 2: ML Latency Prediction & Routing."""
        logger.info(" Initializing Phase 2: ML Latency Prediction & Routing")

        try:
            from data.latency_predictor import LatencyPredictor
            from models.ensemble_latency_model import EnsembleLatencyModel
            from src.ml.routing import RoutingEnvironment
            from data.logs.market_regime_detector import (
                MarketRegimeDetector,
                OnlineLearner,
            )
        except ImportError:
            from integration.phase2_stubs import (
                LatencyPredictor,
                EnsembleLatencyModel,
                RoutingEnvironment,
                MarketRegimeDetector,
                OnlineLearner,
            )

        latency_predictor = LatencyPredictor(list(self.venues.keys()))
        ensemble_model = EnsembleLatencyModel(list(self.venues.keys()))

        self._configure_ml_models(latency_predictor, ensemble_model)

        routing_environment = RoutingEnvironment(
            latency_predictor,
            phase1_components["market_generator"],
            phase1_components["network_simulator"],
            phase1_components["order_book_manager"],
            phase1_components["feature_extractor"],
            venue_list=list(self.venues.keys()),
        )

        market_regime_detector = MarketRegimeDetector()
        online_learner = OnlineLearner(
            {
                "latency_predictor": latency_predictor,
                "ensemble_model": ensemble_model,
                "routing_environment": routing_environment,
            }
        )

        logger.info(" Phase 2 initialization complete")

        return {
            "latency_predictor": latency_predictor,
            "ensemble_model": ensemble_model,
            "routing_environment": routing_environment,
            "market_regime_detector": market_regime_detector,
            "online_learner": online_learner,
        }

    async def initialize_phase3(self) -> Dict[str, Any]:
        """Initialize Phase 3: Trading & Risk Management."""
        logger.info(" Initializing Phase 3: Trading & Risk Management")

        from simulator.trading_simulator import TradingSimulator
        from engine.risk_management_engine import (
            RiskManager,
            PnLAttribution,
            create_integrated_risk_system,
        )
        from simulator.backtesting_framework import BacktestingEngine, BacktestConfig
        from simulator.trading_simulator_integration import (
            create_enhanced_trading_simulator,
        )
        from datetime import datetime, timedelta

        trading_simulator = create_enhanced_trading_simulator(
            symbols=self.symbols,
            venues=list(self.venues.keys()),
            config={
                "enable_latency_simulation": True,
                "venue_optimization": True,
                "strategy_latency_optimization": True,
            },
        )

        try:
            from simulator.enhanced_execution_cost_model import (
                integrate_enhanced_cost_model,
            )

            trading_simulator = integrate_enhanced_cost_model(trading_simulator)
            logger.info(" Enhanced execution cost modeling integrated")
        except ImportError:
            logger.warning("️ Using basic cost modeling")

        risk_system = create_integrated_risk_system()
        risk_manager = risk_system["risk_manager"]
        pnl_attribution = risk_system["pnl_attribution"]

        backtest_config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            symbols=self.symbols,
            venues=list(self.venues.keys()),
            initial_capital=1_000_000,
        )
        backtesting_engine = BacktestingEngine(backtest_config)

        logger.info(" Phase 3 initialization complete")

        return {
            "trading_simulator": trading_simulator,
            "risk_manager": risk_manager,
            "pnl_attribution": pnl_attribution,
            "backtesting_engine": backtesting_engine,
        }

    async def initialize_enhanced_analytics(self) -> Optional[Dict[str, Any]]:
        """Initialize enhanced analytics components."""
        try:
            from data.advanced_technical_indicators import AdvancedTechnicalEngine
            from monitoring.system_health_monitor import create_monitoring_system
            from analytics.real_time_performance_analyzer import (
                RealTimePerformanceAnalyzer,
            )

            logger.info(" Initializing Enhanced Analytics (FREE Performance Boost)...")

            technical_engine = AdvancedTechnicalEngine()
            price_validator, health_monitor = create_monitoring_system(list(self.venues.keys()))
            performance_analyzer = RealTimePerformanceAnalyzer()

            health_monitor.start_monitoring()
            performance_analyzer.start_continuous_analysis()

            logger.info(" Enhanced analytics monitoring started")

            return {
                "technical_engine": technical_engine,
                "price_validator": price_validator,
                "health_monitor": health_monitor,
                "performance_analyzer": performance_analyzer,
            }
        except ImportError as e:
            logger.warning(f"Enhanced analytics not available: {e}")
            return None

    def _configure_ml_models(self, latency_predictor: Any, ensemble_model: Any) -> None:
        """Configure ML models based on mode."""
        if hasattr(latency_predictor, "sequence_length"):
            latency_predictor.sequence_length = self.config["sequence_length"]
            latency_predictor.update_threshold = self.config["update_threshold"]

        if hasattr(ensemble_model, f"_{self.mode}_mode"):
            setattr(ensemble_model, f"_{self.mode}_mode", True)
