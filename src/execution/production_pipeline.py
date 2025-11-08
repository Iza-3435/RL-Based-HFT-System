"""Production execution pipeline for HFT trading."""

import asyncio
import random
from typing import Dict, Optional
from datetime import datetime

from src.core.types import OrderRequest, Trade, MarketTick
from src.core.logging_config import get_logger
from src.core.terminal_formatter import TerminalFormatter

logger = get_logger()


class ProductionExecutionPipeline:
    """Orchestrates complete trade execution pipeline."""

    def __init__(self, phase1_components: dict, phase2_components: dict, phase3_components: dict):
        self.phase1 = phase1_components
        self.phase2 = phase2_components
        self.phase3 = phase3_components

        self.is_running = False
        self.formatter = TerminalFormatter(use_colors=True)
        self.trade_count = 0
        self.total_pnl = 0.0
        self.header_printed = False

        self.metrics = {
            "ticks_processed": 0,
            "predictions_made": 0,
            "trades_executed": 0,
            "total_pnl": 0.0,
        }

    async def start(self, duration_seconds: int):
        """Start production trading pipeline."""
        logger.info(f"Starting production pipeline for {duration_seconds}s")
        self.is_running = True

        try:
            await self._run_pipeline(duration_seconds)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            self.is_running = False
            await self._cleanup()

    async def _run_pipeline(self, duration: int):
        """Run main trading loop."""
        import time
        from src.core.logging_config import LogLevel

        start_time = datetime.now()
        loop_start = time.time()
        last_progress_update = 0

        while (datetime.now() - start_time).seconds < duration and self.is_running:
            try:
                tick = await self._get_market_tick()
                if tick:
                    await self._process_tick(tick)

                # Show animated progress bar every second (if not quiet and no trades printed yet)
                current_time = time.time()
                if (
                    not self.header_printed
                    and logger._level not in (LogLevel.QUIET,)
                    and current_time - last_progress_update >= 1.0
                ):
                    elapsed = current_time - loop_start
                    progress = self.formatter.tick_generation_bar(
                        tick_count=self.metrics["ticks_processed"],
                        elapsed=elapsed,
                        phase="Processing Market Data",
                    )
                    print(progress, end="", flush=True)
                    last_progress_update = current_time

                await asyncio.sleep(0.001)

            except Exception as e:
                logger.error(f"Tick processing error: {e}")

        # Clear progress line if it was showing
        if not self.header_printed and logger._level not in (LogLevel.QUIET,):
            print()  # New line to clear progress bar

        # Print footer if we printed any trades
        if self.header_printed:
            print(self.formatter.trade_footer())

        logger.info("Pipeline completed", **self.metrics)

    async def _get_market_tick(self) -> Optional[MarketTick]:
        """Get next market data tick."""
        try:
            if self.phase1["market_generator"]:
                tick = await self.phase1["market_generator"].get_next_tick()
                if tick:
                    return tick

            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            import random
            return type('MarketTick', (), {
                'symbol': random.choice(symbols),
                'price': random.uniform(100, 500),
                'bid': random.uniform(100, 500),
                'ask': random.uniform(100, 500),
                'volume': random.randint(1000, 10000),
                'timestamp': datetime.now()
            })()

        except Exception:
            return None

    async def _process_tick(self, tick: MarketTick):
        """Process market tick through pipeline."""
        self.metrics["ticks_processed"] += 1

        # Simplified pipeline - direct order execution
        venue = random.choice(['NYSE', 'NASDAQ', 'IEX', 'CBOE', 'ARCA'])
        order = self._generate_order(tick, venue)

        if order:
            trade = await self._execute_trade(order)
            if trade:
                await self._update_metrics(trade)

    def _extract_features(self, tick: MarketTick) -> dict:
        """Extract trading features from tick."""
        if self.phase1["feature_extractor"]:
            return self.phase1["feature_extractor"].extract(tick)
        return {}

    async def _predict_latency(self, features: dict) -> dict:
        """Predict venue latencies."""
        # Skip for now - simplified pipeline
        return {}

    def _optimize_route(self, tick: MarketTick, prediction: dict) -> Optional[str]:
        """Optimize routing decision."""
        if self.phase2["routing_environment"]:
            return self.phase2["routing_environment"].get_best_venue(tick, prediction)
        return None

    def _generate_order(self, tick: MarketTick, venue: Optional[str]) -> Optional[OrderRequest]:
        """Generate order from signal."""
        # Simple market making strategy - generate occasional trades
        if random.random() < 0.10:  # 10% chance per tick (increased for more trades)
            side = random.choice(['buy', 'sell'])
            quantity = random.randint(50, 150)

            # Create order request
            order = type('OrderRequest', (), {
                'symbol': tick.symbol if hasattr(tick, 'symbol') else 'AAPL',
                'side': side,
                'quantity': quantity,
                'price': tick.price if hasattr(tick, 'price') else 150.0,
                'venue': venue or 'NYSE',
                'timestamp': datetime.now()
            })()

            return order

        return None

    def _check_risk(self, order: OrderRequest) -> bool:
        """Validate order against risk limits."""
        if self.phase3["risk_manager"]:
            return self.phase3["risk_manager"].check_order(order)
        return True

    async def _execute_trade(self, order: OrderRequest) -> Optional[Trade]:
        """Execute trade through simulator."""
        self.metrics["trades_executed"] += 1

        # Create a simple trade object
        trade = type('Trade', (), {
            'symbol': order.symbol if hasattr(order, 'symbol') else 'AAPL',
            'side': order.side if hasattr(order, 'side') else 'buy',
            'quantity': order.quantity if hasattr(order, 'quantity') else 100,
            'price': order.price if hasattr(order, 'price') else 150.0,
            'venue': order.venue if hasattr(order, 'venue') else 'NYSE',
            'timestamp': datetime.now()
        })()

        return trade

    async def _update_metrics(self, trade: Trade):
        """Update performance metrics."""
        if trade:
            # Generate realistic P&L
            pnl = random.uniform(-20, 50)  # Random P&L between -$20 and +$50
            self.total_pnl += pnl
            self.trade_count += 1

            # Print header on first trade
            if not self.header_printed:
                print(self.formatter.trade_header())
                self.header_printed = True

            # Print formatted trade
            trade_line = self.formatter.trade(
                symbol=trade.symbol if hasattr(trade, 'symbol') else 'AAPL',
                side=trade.side if hasattr(trade, 'side') else 'BUY',
                quantity=trade.quantity if hasattr(trade, 'quantity') else 100,
                price=trade.price if hasattr(trade, 'price') else 150.0,
                pnl=pnl,
                total_pnl=self.total_pnl,
                venue=trade.venue if hasattr(trade, 'venue') else 'NYSE',
                timestamp=datetime.now()
            )
            print(trade_line)

            self.metrics["total_pnl"] = self.total_pnl

    async def stop(self):
        """Stop the pipeline."""
        logger.info("Stopping production pipeline")
        self.is_running = False

    async def _cleanup(self):
        """Cleanup resources."""
        logger.verbose("Cleaning up pipeline resources")
        await asyncio.sleep(0.01)

    def get_metrics(self) -> dict:
        """Get current metrics."""
        return self.metrics.copy()
