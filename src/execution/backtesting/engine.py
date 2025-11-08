"""Main backtesting engine."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from src.core.logging_config import get_logger
from src.execution.backtesting.config import BacktestConfig, BacktestResult, BacktestMode
from src.execution.backtesting.data_manager import HistoricalDataManager
from src.execution.backtesting.execution_sim import BacktestExecutionSimulator
from src.execution.backtesting.performance import PerformanceAnalyzer

logger = get_logger()


class BacktestEngine:
    """Production backtesting engine."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        config.validate()

        self.data_manager = HistoricalDataManager(config)
        self.execution_simulator = BacktestExecutionSimulator(config)
        self.performance_analyzer = PerformanceAnalyzer()

        self.current_capital = config.initial_capital
        self.positions: Dict[str, int] = {}
        self.equity_curve: List[float] = []
        self.trades: List[Dict] = []

        logger.info("Backtest engine initialized",
                   symbols=len(config.symbols),
                   mode=config.mode.value)

    async def run(self, strategy) -> BacktestResult:
        """Run backtest with given strategy."""
        logger.info("Starting backtest",
                   start=self.config.start_date.isoformat(),
                   end=self.config.end_date.isoformat())

        start_time = datetime.now()

        try:
            await self.data_manager.load_data()

            ticks = await self.data_manager.get_ticks(
                self.config.start_date,
                self.config.end_date
            )

            for i, tick in enumerate(ticks):
                await self._process_tick(tick, strategy)

                if i % 1000 == 0:
                    logger.verbose(f"Processed {i}/{len(ticks)} ticks")

            result = self._generate_result(start_time)
            logger.info("Backtest complete", total_trades=result.total_trades)

            return result

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise

    async def _process_tick(self, tick: Dict, strategy):
        """Process single tick."""
        signals = strategy.generate_signals(tick, self.positions)

        for signal in signals:
            trade = await self.execution_simulator.execute(
                signal,
                tick,
                self.current_capital
            )

            if trade:
                self._update_state(trade)
                self.trades.append(trade)

        self.equity_curve.append(self.current_capital)

    def _update_state(self, trade: Dict):
        """Update internal state after trade."""
        symbol = trade['symbol']
        quantity = trade['quantity']
        side = trade['side']

        if side == 'buy':
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity

        self.current_capital += trade['pnl']

    def _generate_result(self, start_time: datetime) -> BacktestResult:
        """Generate backtest result."""
        metrics = self.performance_analyzer.calculate_metrics(
            self.trades,
            self.equity_curve,
            self.config.initial_capital
        )

        return BacktestResult(
            config=self.config,
            start_time=start_time,
            end_time=datetime.now(),
            total_trades=len(self.trades),
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            total_pnl=metrics['total_pnl'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            avg_trade_pnl=metrics['avg_trade_pnl'],
            total_commission=metrics['total_commission'],
            total_slippage=metrics['total_slippage'],
            final_capital=self.current_capital,
            equity_curve=self.equity_curve,
            trades=self.trades
        )
