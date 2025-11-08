"""Main orchestrator for Phase 3 complete integration."""

import logging
import time
from typing import Any, Dict, List, Optional

from .analytics_generator import AnalyticsGenerator
from .backtesting import BacktestingEngine
from .component_initializers import ComponentInitializer
from .config import EXPANDED_STOCK_LIST, get_risk_limits, get_venue_configs
from .reporting import ReportGenerator
from .risk_monitor import RiskMonitor
from .simulation_runner import SimulationRunner
from .trade_executor import TradeExecutor
from .training_manager import TrainingManager

logger = logging.getLogger(__name__)


class Phase3CompleteIntegration:
    """Complete Phase 3 integration orchestrator."""

    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or EXPANDED_STOCK_LIST
        self.venues = get_venue_configs()

        logger.info(
            f"Trading {len(self.symbols)} symbols: {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}"
        )

        self.market_generator = None
        self.network_simulator = None
        self.order_book_manager = None
        self.feature_extractor = None
        self.latency_predictor = None
        self.ensemble_model = None
        self.routing_environment = None
        self.market_regime_detector = None
        self.online_learner = None
        self.execution_pipeline = None

        self.system_state = "initializing"
        self.current_positions = {}
        self.total_pnl = 0.0
        self.trade_count = 0
        self.risk_alerts = []

        self.integration_metrics = {
            "ticks_processed": 0,
            "predictions_made": 0,
            "trades_executed": 0,
            "risk_checks": 0,
            "regime_changes": 0,
            "ml_routing_benefit": 0.0,
        }

        self.component_initializer = ComponentInitializer(self.symbols, self.venues)
        self.training_manager = None
        self.simulation_runner = None
        self.trade_executor = None
        self.risk_monitor = None
        self.analytics_generator = None
        self.backtesting_engine = None
        self.report_generator = ReportGenerator()

        logger.info("Phase 3 Complete Integration initialized")

    async def initialize_all_phases(self):
        """Initialize all components from Phases 1, 2, and 3."""
        logger.info("Initializing HFT Production System - All Phases")

        phase1_components = await self.component_initializer.initialize_phase1()
        self.market_generator = phase1_components["market_generator"]
        self.network_simulator = phase1_components["network_simulator"]
        self.order_book_manager = phase1_components["order_book_manager"]
        self.feature_extractor = phase1_components["feature_extractor"]

        phase2_components = await self.component_initializer.initialize_phase2()
        self.latency_predictor = phase2_components["latency_predictor"]
        self.ensemble_model = phase2_components["ensemble_model"]
        self.routing_environment = phase2_components["routing_environment"]
        self.market_regime_detector = phase2_components["market_regime_detector"]
        self.online_learner = phase2_components["online_learner"]

        phase3_components = await self.component_initializer.initialize_phase3()
        self.execution_pipeline = phase3_components["execution_pipeline"]

        self.training_manager = TrainingManager(
            self.market_generator,
            self.network_simulator,
            self.order_book_manager,
            self.feature_extractor,
            self.latency_predictor,
            self.ensemble_model,
            self.routing_environment,
            self.market_regime_detector,
        )

        self.trade_executor = TradeExecutor(
            self.routing_environment,
            phase3_components.get("risk_manager"),
        )

        self.risk_monitor = RiskMonitor(self.execution_pipeline)

        self.simulation_runner = SimulationRunner(
            self.market_generator,
            self.network_simulator,
            self.order_book_manager,
            self.feature_extractor,
            self.execution_pipeline,
            self.online_learner,
            self.venues,
        )

        self.analytics_generator = AnalyticsGenerator()

        self.backtesting_engine = BacktestingEngine(
            self.market_generator,
            self.routing_environment,
            phase3_components.get("risk_manager"),
        )

        self.system_state = "ready"
        logger.info("All phases initialized - System ready for trading")

    async def train_ml_models(self, training_duration_minutes: int = 10):
        """Train all ML models."""
        logger.info(f"Training ML models for {training_duration_minutes} minutes")
        await self.training_manager.train_all_models(training_duration_minutes)
        logger.info("ML model training complete")

    async def run_production_simulation(self, duration_minutes: int = 30):
        """Run production simulation."""
        logger.info(f"Running production simulation for {duration_minutes} minutes")

        start_time = time.time()

        simulation_data = await self.simulation_runner.run_simulation(
            duration_minutes, self.trade_executor, self.risk_monitor
        )

        simulation_results = simulation_data["simulation_results"]
        tick_count = simulation_data["tick_count"]

        final_results = self.analytics_generator.generate_final_results(
            simulation_results, start_time, tick_count, self.risk_monitor
        )

        logger.info("Production simulation complete")
        return final_results

    async def run_backtesting_validation(self):
        """Run backtesting validation."""
        logger.info("Running backtesting validation")
        backtest_results = await self.backtesting_engine.run_backtesting_validation()
        logger.info("Backtesting validation complete")
        return backtest_results

    async def generate_comprehensive_report(
        self, simulation_results: Dict, backtest_results: Optional[Dict] = None
    ):
        """Generate comprehensive report."""
        logger.info("Generating comprehensive report")
        report = await self.report_generator.generate_comprehensive_report(
            simulation_results, backtest_results
        )
        self.report_generator.print_executive_summary(report)
        return report

    async def run_complete_workflow(
        self, training_minutes: int = 10, simulation_minutes: int = 30
    ):
        """Run complete workflow: initialize, train, simulate, backtest, report."""
        logger.info("Starting complete workflow")

        await self.initialize_all_phases()

        await self.train_ml_models(training_minutes)

        simulation_results = await self.run_production_simulation(simulation_minutes)

        backtest_results = await self.run_backtesting_validation()

        report = await self.generate_comprehensive_report(
            simulation_results, backtest_results
        )

        logger.info("Complete workflow finished")
        return {
            "simulation_results": simulation_results,
            "backtest_results": backtest_results,
            "report": report,
        }
