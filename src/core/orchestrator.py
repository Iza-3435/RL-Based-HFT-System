
"""Main orchestrator for production HFT system."""

import time
from typing import List, Optional
from datetime import datetime

from src.core.config import ConfigManager
from src.core.logging_config import get_logger, LogLevel
from src.core.terminal_formatter import TerminalFormatter
from src.core.trade_logger import get_trade_logger
from src.infra.phase1_manager import Phase1Manager
from src.ml.phase2_manager import Phase2Manager
from src.execution.phase3_manager import Phase3Manager
from src.execution.production_pipeline import ProductionExecutionPipeline

logger = get_logger()


class HFTSystemOrchestrator:
    """Orchestrates complete HFT trading system."""

    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()

        logger.set_level(self.config.system.log_level)

        self.phase1: Optional[Phase1Manager] = None
        self.phase2: Optional[Phase2Manager] = None
        self.phase3: Optional[Phase3Manager] = None
        self.pipeline: Optional[ProductionExecutionPipeline] = None

        self.formatter = TerminalFormatter(use_colors=True)
        self.trade_logger = get_trade_logger()
        self.state = "initializing"
        self.duration_seconds = 0

    async def initialize(self):
        """Initialize all system components."""
        # Print banner (unless quiet mode)
        if logger._level != LogLevel.QUIET:
            print(
                self.formatter.banner(
                    mode=self.config.system.mode.value,
                    duration=self.duration_seconds,
                    symbols=len(self.config.trading.symbols),
                )
            )

        try:
            # Phase 1
            start = time.time()
            if logger._level not in (LogLevel.QUIET, LogLevel.NORMAL):
                print(self.formatter.phase_start("Phase 1", "Market data & network"))
            await self._init_phase1()
            if logger._level not in (LogLevel.QUIET, LogLevel.NORMAL):
                print(self.formatter.phase_complete("Phase 1", (time.time() - start) * 1000))

            # Phase 2
            start = time.time()
            if logger._level not in (LogLevel.QUIET, LogLevel.NORMAL):
                print(self.formatter.phase_start("Phase 2", "ML models & routing"))
            await self._init_phase2()
            if logger._level not in (LogLevel.QUIET, LogLevel.NORMAL):
                print(self.formatter.phase_complete("Phase 2", (time.time() - start) * 1000))

            # Phase 3
            start = time.time()
            if logger._level not in (LogLevel.QUIET, LogLevel.NORMAL):
                print(self.formatter.phase_start("Phase 3", "Execution & risk"))
            await self._init_phase3()
            if logger._level not in (LogLevel.QUIET, LogLevel.NORMAL):
                print(self.formatter.phase_complete("Phase 3", (time.time() - start) * 1000))

            # Pipeline
            await self._create_pipeline()

            self.state = "ready"

            # Show system ready message
            if logger._level != LogLevel.QUIET:
                print(self.formatter.system_ready())

        except Exception as e:
            print(self.formatter.error(f"System initialization failed: {e}"))
            self.state = "failed"
            raise

    async def _init_phase1(self):
        """Initialize Phase 1."""
        self.phase1 = Phase1Manager(
            symbols=self.config.trading.symbols,
            venues=self.config.network.venues,
            mode=self.config.system.mode.value,
        )
        await self.phase1.initialize()

    async def _init_phase2(self):
        """Initialize Phase 2."""
        self.phase2 = Phase2Manager(venues=self.config.network.venues, mode=self.config.system.mode)
        await self.phase2.initialize(self.phase1.get_components())

    async def _init_phase3(self):
        """Initialize Phase 3."""
        self.phase3 = Phase3Manager(
            symbols=self.config.trading.symbols, venues=self.config.network.venues
        )
        await self.phase3.initialize()

    async def _create_pipeline(self):
        """Create execution pipeline."""
        self.pipeline = ProductionExecutionPipeline(
            phase1_components=self.phase1.get_components(),
            phase2_components=self.phase2.get_components(),
            phase3_components=self.phase3.get_components(),
        )
        logger.verbose("Execution pipeline created")

    async def train_models(self, training_duration: int = 120):
        """Train ML models with generated data."""
        if self.state != "ready":
            raise RuntimeError(f"System not ready: {self.state}")

        logger.info(f"Training ML models with {training_duration}s of data")

        if self.phase1.market_generator:
            self.phase1.market_generator.start()

        await self.phase2.train_models(training_duration)

        if self.phase1.market_generator:
            self.phase1.market_generator.stop()

        logger.info("ML models trained successfully")

    async def run(self, duration_seconds: int = 600, skip_training: bool = False):
        """Run production trading."""
        if self.state != "ready":
            raise RuntimeError(f"System not ready: {self.state}")

        self.duration_seconds = duration_seconds

        if not skip_training:
            mode = self.config.system.mode.value
            training_time = 30 if mode == "fast" else (120 if mode == "balanced" else 300)
            await self.train_models(training_time)

        if self.phase1 and self.phase1.market_generator:
            self.phase1.market_generator.start()

        start_time = datetime.now()
        await self.pipeline.start(duration_seconds)
        elapsed = (datetime.now() - start_time).total_seconds()

        if self.phase1 and self.phase1.market_generator:
            self.phase1.market_generator.stop()

        metrics = self.pipeline.get_metrics()

        if logger._level != LogLevel.QUIET:
            win_rate = (self.formatter.win_count / self.pipeline.trade_count * 100) if self.pipeline.trade_count > 0 else 0
            print(
                self.formatter.summary(
                    duration=elapsed,
                    trades=self.pipeline.trade_count,
                    total_pnl=self.pipeline.total_pnl,
                    win_rate=win_rate,
                    sharpe=metrics.get("sharpe_ratio"),
                    max_dd=metrics.get("max_drawdown"),
                )
            )

        return metrics

    async def shutdown(self):
        """Shutdown system gracefully."""
        if self.pipeline and self.pipeline.is_running:
            await self.pipeline.stop()

        self.state = "shutdown"
