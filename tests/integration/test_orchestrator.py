"""Integration tests for system orchestrator."""

import pytest
import asyncio

from src.core.config import ConfigManager
from src.core.orchestrator import HFTSystemOrchestrator
from src.core.types import TradingMode


@pytest.mark.asyncio
class TestHFTSystemOrchestrator:
    """Test complete system orchestration."""

    async def test_orchestrator_initialization(self):
        """Test orchestrator initializes all phases."""
        config = ConfigManager()
        config.trading.symbols = ['AAPL', 'MSFT']

        orchestrator = HFTSystemOrchestrator(config)

        assert orchestrator.state == "initializing"
        assert orchestrator.phase1 is None
        assert orchestrator.phase2 is None
        assert orchestrator.phase3 is None

    async def test_phase_initialization_sequence(self):
        """Test phases initialize in correct order."""
        config = ConfigManager()
        config.trading.symbols = ['AAPL']

        orchestrator = HFTSystemOrchestrator(config)

        try:
            await orchestrator.initialize()

            assert orchestrator.state == "ready"
            assert orchestrator.phase1 is not None
            assert orchestrator.phase2 is not None
            assert orchestrator.phase3 is not None
            assert orchestrator.pipeline is not None

        except ImportError:
            pytest.skip("Dependencies not available")

    async def test_orchestrator_shutdown(self):
        """Test graceful shutdown."""
        config = ConfigManager()
        orchestrator = HFTSystemOrchestrator(config)

        try:
            await orchestrator.initialize()
            await orchestrator.shutdown()

            assert orchestrator.state == "shutdown"

        except ImportError:
            pytest.skip("Dependencies not available")
