"""Main entry point for production HFT system."""

import argparse
import asyncio
import sys
from pathlib import Path

from src.core.config import ConfigManager
from src.core.logging_config import get_logger, LogLevel
from src.core.orchestrator import HFTSystemOrchestrator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HFT Network Optimizer - Production Trading System"
    )

    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "production"],
        default="production",
        help="Trading mode (default: production)",
    )

    parser.add_argument(
        "--duration", type=int, default=600, help="Trading duration in seconds (default: 600)"
    )

    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols to trade")

    parser.add_argument("--config", type=Path, help="Path to configuration file")

    parser.add_argument(
        "--log-level",
        choices=["quiet", "normal", "verbose", "debug"],
        default="normal",
        help="Logging verbosity (default: normal)",
    )

    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtesting phase")

    return parser.parse_args()


async def main():
    """Main execution function."""
    args = parse_args()

    logger = get_logger()
    log_level = LogLevel[args.log_level.upper()]
    logger.set_level(log_level)

    logger.info("HFT Network Optimizer starting")
    logger.verbose("Arguments", **vars(args))

    try:
        config = ConfigManager(args.config)

        if args.symbols:
            config.trading.symbols = args.symbols.split(",")

        from src.core.types import TradingMode
        config.system.mode = TradingMode[args.mode.upper()]

        orchestrator = HFTSystemOrchestrator(config)
        await orchestrator.initialize()

        metrics = await orchestrator.run(duration_seconds=args.duration)

        logger.info("Trading complete", **metrics)

        await orchestrator.shutdown()
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
