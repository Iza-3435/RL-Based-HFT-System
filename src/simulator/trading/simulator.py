"""Main trading simulator for strategy execution and performance measurement."""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .arbitrage import ArbitrageStrategy
from .dataclasses import Fill, Order
from .execution_engine import OrderExecutionEngine
from .market_making import MarketMakingStrategy
from .momentum import MomentumStrategy

logger = logging.getLogger(__name__)


class TradingSimulator:
    """Main trading simulator that orchestrates strategy execution and performance measurement."""

    def __init__(self, venues: List[str], symbols: List[str]):
        self.venues = venues
        self.symbols = symbols

        self.execution_engine = OrderExecutionEngine()

        self.strategies = {
            "market_making": MarketMakingStrategy(),
            "arbitrage": ArbitrageStrategy(),
            "momentum": MomentumStrategy(),
        }

        self.total_pnl = 0
        self.trade_count = 0
        self.simulation_start_time = None
        self.performance_history = []

        self.pending_orders = []
        self.order_history = []
        self.fill_history = []

        self.max_drawdown = 0
        self.high_water_mark = 0

        logger.info(
            f"TradingSimulator initialized for {len(venues)} venues, {len(symbols)} symbols"
        )

    async def simulate_trading(
        self, market_data_generator, ml_predictor, duration_seconds: int = 300
    ) -> Dict[str, Any]:
        """Run complete trading simulation."""
        logger.info(f"Starting trading simulation for {duration_seconds} seconds")

        self.simulation_start_time = time.time()
        tick_count = 0

        market_state = {}
        current_prices = {}

        async for tick in market_data_generator.generate_market_data_stream(
            duration_seconds
        ):
            symbol_venue_key = f"{tick.symbol}_{tick.venue}"
            market_state[symbol_venue_key] = {
                "bid_price": tick.bid_price,
                "ask_price": tick.ask_price,
                "mid_price": tick.mid_price,
                "spread_bps": (tick.ask_price - tick.bid_price)
                / tick.mid_price
                * 10000,
                "volume": tick.volume,
                "bid_size": tick.bid_size,
                "ask_size": tick.ask_size,
                "volatility": tick.volatility,
                "average_daily_volume": 1000000,
                "average_trade_size": 100,
            }

            current_prices[tick.symbol] = tick.mid_price

            if tick_count % 10 == 0:
                aggregated_market_data = {
                    "symbols": self.symbols,
                    "venues": self.venues,
                    **market_state,
                }

                ml_predictions = await self._get_ml_predictions(
                    ml_predictor, tick, market_state
                )

                all_orders = []
                for strategy_name, strategy in self.strategies.items():
                    orders = await strategy.generate_signals(
                        aggregated_market_data, ml_predictions
                    )
                    all_orders.extend(orders)

                for order in all_orders:
                    fill = await self._execute_order(order, market_state)

                    if fill:
                        strategy = self.strategies[order.strategy.value]
                        strategy.update_positions(fill, current_prices)

                        self.fill_history.append(fill)
                        self.trade_count += 1

                self._update_performance_metrics(current_prices)

            tick_count += 1

            if tick_count % 1000 == 0:
                elapsed = time.time() - self.simulation_start_time
                logger.info(
                    f"Processed {tick_count} ticks in {elapsed:.1f}s, "
                    f"Total P&L: ${self.total_pnl:.2f}"
                )

        results = self._generate_simulation_results()

        logger.info(
            f"Simulation complete: {tick_count} ticks, {self.trade_count} trades"
        )
        return results

    async def _get_ml_predictions(
        self, ml_predictor, tick, market_state
    ) -> Dict:
        """Get ML predictions for routing and signals."""
        predictions = {}

        for venue in self.venues:
            routing_key = f"routing_{tick.symbol}_{venue}"

            routing_decision = ml_predictor.make_routing_decision(tick.symbol)

            predictions[routing_key] = {
                "venue": routing_decision.venue,
                "predicted_latency_us": routing_decision.expected_latency_us,
                "confidence": routing_decision.confidence,
            }

        regime_detection = ml_predictor.detect_market_regime(market_state)
        predictions["regime"] = regime_detection.regime.value

        predictions["volatility_forecast"] = (
            market_state.get(f"{tick.symbol}_{tick.venue}", {}).get("volatility", 0.01)
        )

        predictions[f"momentum_signal_{tick.symbol}"] = np.random.randn() * 0.5

        return predictions

    async def _execute_order(
        self, order: Order, market_state: Dict
    ) -> Optional[Fill]:
        """Execute order through execution engine."""
        state_key = f"{order.symbol}_{order.venue}"
        if state_key not in market_state:
            logger.warning(f"No market data for {state_key}")
            return None

        venue_market_state = market_state[state_key]

        if order.predicted_latency_us:
            latency_noise = np.random.normal(0, order.predicted_latency_us * 0.1)
            actual_latency_us = max(50, order.predicted_latency_us + latency_noise)
        else:
            actual_latency_us = np.random.normal(1000, 200)

        fill = await self.execution_engine.execute_order(
            order, venue_market_state, actual_latency_us
        )

        return fill

    def _update_performance_metrics(self, current_prices: Dict):
        """Update P&L and risk metrics."""
        total_pnl = 0

        for strategy in self.strategies.values():
            strategy_pnl = strategy.get_total_pnl()
            total_pnl += strategy_pnl["total_pnl"]

        self.total_pnl = total_pnl

        if total_pnl > self.high_water_mark:
            self.high_water_mark = total_pnl

        drawdown = self.high_water_mark - total_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.performance_history.append(
            {
                "timestamp": time.time(),
                "total_pnl": total_pnl,
                "trade_count": self.trade_count,
                "drawdown": drawdown,
            }
        )

    def _generate_simulation_results(self) -> Dict[str, Any]:
        """Generate comprehensive simulation results."""
        results = {
            "summary": {
                "total_pnl": self.total_pnl,
                "trade_count": self.trade_count,
                "simulation_duration": time.time() - self.simulation_start_time,
                "max_drawdown": self.max_drawdown,
                "final_positions": self._get_final_positions(),
            },
            "strategy_performance": {},
            "execution_stats": self.execution_engine.get_execution_stats(),
            "venue_analysis": self._analyze_venue_performance(),
            "ml_routing_impact": self._analyze_ml_impact(),
        }

        for strategy_name, strategy in self.strategies.items():
            pnl_data = strategy.get_total_pnl()

            if strategy_name == "market_making":
                strategy_results = {
                    **pnl_data,
                    "quotes_posted": strategy.quote_count,
                    "spread_captured": strategy.spread_captured,
                }
            elif strategy_name == "arbitrage":
                strategy_results = {
                    **pnl_data,
                    "opportunities_detected": strategy.opportunities_detected,
                    "opportunities_captured": strategy.opportunities_captured,
                    "capture_rate": strategy.opportunities_captured
                    / max(strategy.opportunities_detected, 1),
                    "arbitrage_pnl": strategy.arb_pnl,
                }
            elif strategy_name == "momentum":
                strategy_results = {
                    **pnl_data,
                    "signals_generated": len(strategy.orders),
                    "positions_taken": len(strategy.entry_prices),
                }

            results["strategy_performance"][strategy_name] = strategy_results

        if self.performance_history:
            pnl_series = [p["total_pnl"] for p in self.performance_history]
            returns = np.diff(pnl_series)

            if len(returns) > 0:
                results["risk_metrics"] = {
                    "sharpe_ratio": self._calculate_sharpe_ratio(returns),
                    "max_drawdown_pct": self.max_drawdown
                    / max(abs(self.high_water_mark), 1)
                    * 100,
                    "profit_factor": self._calculate_profit_factor(),
                    "win_rate": self._calculate_win_rate(),
                }

        return results

    def _get_final_positions(self) -> Dict[str, Dict]:
        """Get final positions across all strategies."""
        final_positions = {}

        for strategy_name, strategy in self.strategies.items():
            for symbol, position in strategy.positions.items():
                if position.quantity != 0:
                    final_positions[f"{strategy_name}_{symbol}"] = {
                        "quantity": position.quantity,
                        "average_cost": position.average_cost,
                        "unrealized_pnl": position.unrealized_pnl,
                    }

        return final_positions

    def _analyze_venue_performance(self) -> Dict[str, Dict]:
        """Analyze performance by venue."""
        from collections import defaultdict

        venue_stats = defaultdict(
            lambda: {
                "fill_count": 0,
                "total_volume": 0,
                "total_fees": 0,
                "total_rebates": 0,
                "avg_latency_us": 0,
                "avg_slippage_bps": 0,
            }
        )

        for fill in self.fill_history:
            stats = venue_stats[fill.venue]
            stats["fill_count"] += 1
            stats["total_volume"] += fill.quantity
            stats["total_fees"] += fill.fees
            stats["total_rebates"] += fill.rebate

            n = stats["fill_count"]
            stats["avg_latency_us"] = (
                stats["avg_latency_us"] * (n - 1) + fill.latency_us
            ) / n
            stats["avg_slippage_bps"] = (
                stats["avg_slippage_bps"] * (n - 1) + fill.slippage_bps
            ) / n

        return dict(venue_stats)

    def _analyze_ml_impact(self) -> Dict[str, float]:
        """Analyze impact of ML routing decisions."""
        ml_routed_fills = [
            f
            for f in self.fill_history
            if f.order_id.startswith(("MM", "ARB", "MOM"))
        ]

        if not ml_routed_fills:
            return {}

        latency_errors = []
        for fill in ml_routed_fills:
            order = next(
                (o for o in self.order_history if o.order_id == fill.order_id), None
            )
            if order and order.predicted_latency_us:
                error = abs(fill.latency_us - order.predicted_latency_us)
                latency_errors.append(error)

        actual_latencies = [f.latency_us for f in ml_routed_fills]
        if actual_latencies:
            baseline_latency = np.percentile(actual_latencies, 90)
            avg_latency = np.mean(actual_latencies)
            latency_improvement = (
                (baseline_latency - avg_latency) / baseline_latency * 100
            )
        else:
            latency_improvement = 0

        return {
            "fills_with_ml_routing": len(ml_routed_fills),
            "avg_latency_prediction_error_us": np.mean(latency_errors)
            if latency_errors
            else 0,
            "latency_improvement_pct": latency_improvement,
            "estimated_pnl_improvement": latency_improvement * self.total_pnl / 100,
        }

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0

        periods_per_year = 252 * 6.5 * 3600 / (
            self.performance_history[1]["timestamp"]
            - self.performance_history[0]["timestamp"]
        )

        return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profits / gross losses)."""
        from .enums import OrderSide

        profits = sum(
            f.quantity * (f.price - p.average_cost)
            for s in self.strategies.values()
            for p in s.positions.values()
            for f in s.fills
            if f.side == OrderSide.SELL and f.price > p.average_cost
        )

        losses = abs(
            sum(
                f.quantity * (f.price - p.average_cost)
                for s in self.strategies.values()
                for p in s.positions.values()
                for f in s.fills
                if f.side == OrderSide.SELL and f.price < p.average_cost
            )
        )

        return profits / losses if losses > 0 else float("inf")

    def _calculate_win_rate(self) -> float:
        """Calculate percentage of profitable trades."""
        from collections import defaultdict

        from .enums import OrderSide

        winning_trades = 0
        total_trades = 0

        for strategy in self.strategies.values():
            symbol_fills = defaultdict(list)
            for fill in strategy.fills:
                symbol_fills[fill.symbol].append(fill)

            for symbol, fills in symbol_fills.items():
                buys = [f for f in fills if f.side == OrderSide.BUY]
                sells = [f for f in fills if f.side == OrderSide.SELL]

                for buy, sell in zip(buys, sells):
                    total_trades += 1
                    if sell.price > buy.price:
                        winning_trades += 1

        return winning_trades / total_trades if total_trades > 0 else 0
