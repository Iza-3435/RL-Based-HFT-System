"""Enhanced trading simulator with comprehensive latency analytics."""

import logging
import time
from typing import Any, Dict, List

from simulator.enhanced_latency_simulation import (
    EnhancedOrderExecutionEngine,
    LatencyAnalytics,
    LatencySimulator,
    create_latency_configuration,
    integrate_latency_simulation,
)
from simulator.network_latency_simulator import NetworkLatencySimulator
from simulator.trading_simulator import TradingSimulator

from .execution_analytics import ExecutionAnalytics
from .market_analysis import MarketConditionAnalyzer
from .ml_predictions import MLPredictionEnhancer
from .order_routing import OrderRoutingEnhancer
from .performance_analysis import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class EnhancedTradingSimulator(TradingSimulator):
    """Enhanced Trading Simulator with realistic latency modeling."""

    def __init__(
        self, venues: List[str], symbols: List[str], enable_latency_simulation: bool = True
    ):
        super().__init__(venues, symbols)

        if enable_latency_simulation:
            self.execution_engine = EnhancedOrderExecutionEngine(venues)
            logger.info("Enhanced latency simulation enabled")
        else:
            self.latency_simulator = LatencySimulator(venues)
            integrate_latency_simulation(self.execution_engine, self.latency_simulator)
            logger.info("Basic latency simulation enabled")

        self.latency_analytics = LatencyAnalytics(self.execution_engine)

        self.latency_performance_history = []
        self.venue_latency_rankings = []

        self.latency_config = create_latency_configuration()

        self.network_simulator = NetworkLatencySimulator()

        self.ml_predictor_enhancer = MLPredictionEnhancer(
            venues, self.execution_engine
        )
        self.order_router = OrderRoutingEnhancer(venues, self.latency_config)
        self.execution_analytics = ExecutionAnalytics(
            self.execution_engine, self.strategies
        )
        self.market_analyzer = MarketConditionAnalyzer(self.execution_engine)

        logger.info("EnhancedTradingSimulator initialized with latency analytics")

    async def simulate_trading(
        self, market_data_generator, ml_predictor, duration_seconds: int = 300
    ) -> Dict[str, Any]:
        """Enhanced trading simulation with comprehensive latency analysis."""
        logger.info(
            f"Starting enhanced trading simulation for {duration_seconds} seconds"
        )

        self.simulation_start_time = time.time()
        tick_count = 0
        latency_snapshots = []

        market_state = {}
        current_prices = {}

        async for tick in market_data_generator.generate_market_data_stream(
            duration_seconds
        ):
            self.network_simulator.update_market_conditions(
                volatility=getattr(tick, "volatility", 0.02),
                volume_factor=getattr(tick, "volume", 1000) / 1000,
            )

            symbol_venue_key = f"{tick.symbol}_{tick.venue}"

            network_latency = self.network_simulator.get_current_latency(tick.venue)

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
                "volatility": getattr(tick, "volatility", 0.02),
                "average_daily_volume": 1000000,
                "average_trade_size": 100,
                "current_network_latency_us": network_latency,
                "regime": MarketConditionAnalyzer.detect_market_regime(tick),
            }

            if hasattr(self.execution_engine, "latency_simulator"):
                self.execution_engine.latency_simulator.update_market_conditions(
                    symbol=tick.symbol,
                    volatility=getattr(tick, "volatility", 0.02),
                    volume_factor=tick.volume / 1000000
                    if hasattr(tick, "volume")
                    else 1.0,
                    timestamp=getattr(tick, "timestamp", time.time()),
                )

            current_prices[tick.symbol] = tick.mid_price

            if tick_count % 10 == 0:
                await self._process_trading_cycle(
                    market_state, current_prices, ml_predictor, tick
                )

                if tick_count % 100 == 0:
                    snapshot = self._capture_latency_snapshot()
                    latency_snapshots.append(snapshot)

            tick_count += 1

            if tick_count % 1000 == 0:
                await self._log_enhanced_progress(tick_count, latency_snapshots)

        results = await self._generate_enhanced_results(latency_snapshots)

        logger.info(
            f"Enhanced simulation complete: {tick_count} ticks, {self.trade_count} trades"
        )
        return results

    async def _process_trading_cycle(
        self, market_state: Dict, current_prices: Dict, ml_predictor, current_tick
    ) -> None:
        """Process one trading cycle with enhanced latency handling."""
        aggregated_market_data = {
            "symbols": self.symbols,
            "venues": self.venues,
            **market_state,
        }

        ml_predictions = await self.ml_predictor_enhancer.get_enhanced_predictions(
            ml_predictor, current_tick, market_state
        )

        all_orders = []
        for strategy_name, strategy in self.strategies.items():
            orders = await strategy.generate_signals(
                aggregated_market_data, ml_predictions
            )

            enhanced_orders = self.order_router.enhance_orders_with_latency_routing(
                orders, ml_predictions, market_state
            )

            all_orders.extend(enhanced_orders)

        execution_results = await self.execution_analytics.execute_orders_with_analytics(
            all_orders, market_state, current_prices
        )

        self._update_performance_metrics(current_prices)

    def _capture_latency_snapshot(self) -> Dict[str, Any]:
        """Capture current latency performance snapshot."""
        snapshot = {
            "timestamp": time.time(),
            "venue_latencies": {},
            "congestion_level": "normal",
            "prediction_accuracy": 0.0,
            "total_trades": self.trade_count,
        }

        if hasattr(self.execution_engine, "get_venue_latency_rankings"):
            rankings = self.execution_engine.get_venue_latency_rankings()
            snapshot["venue_latencies"] = dict(rankings)

        if hasattr(self.execution_engine, "latency_simulator"):
            congestion = self.execution_engine.latency_simulator.get_congestion_analysis()
            snapshot["congestion_level"] = congestion["current_congestion_level"]

            pred_stats = self.execution_engine.latency_simulator.get_prediction_accuracy_stats()
            snapshot["prediction_accuracy"] = pred_stats.get(
                "prediction_within_10pct", 0
            )

        return snapshot

    async def _log_enhanced_progress(
        self, tick_count: int, latency_snapshots: List[Dict]
    ) -> None:
        """Log progress with enhanced latency metrics."""
        elapsed = time.time() - self.simulation_start_time

        logger.info(
            f"Processed {tick_count} ticks in {elapsed:.1f}s, "
            f"Total P&L: ${self.total_pnl:.2f}, Trades: {self.trade_count}"
        )

        if latency_snapshots:
            latest_snapshot = latency_snapshots[-1]

            if latest_snapshot["venue_latencies"]:
                best_venue = min(
                    latest_snapshot["venue_latencies"].items(), key=lambda x: x[1]
                )
                worst_venue = max(
                    latest_snapshot["venue_latencies"].items(), key=lambda x: x[1]
                )

                logger.info(
                    f"Latency: Best={best_venue[0]}({best_venue[1]:.0f}μs), "
                    f"Worst={worst_venue[0]}({worst_venue[1]:.0f}μs), "
                    f"Congestion={latest_snapshot['congestion_level']}, "
                    f"Prediction={latest_snapshot['prediction_accuracy']:.1f}%"
                )

    async def _generate_enhanced_results(
        self, latency_snapshots: List[Dict]
    ) -> Dict[str, Any]:
        """Generate enhanced simulation results with comprehensive latency analysis."""
        base_results = self._generate_simulation_results()

        latency_analysis = self.latency_analytics.generate_latency_report(
            timeframe_minutes=int((time.time() - self.simulation_start_time) / 60)
        )

        performance_analyzer = PerformanceAnalyzer(
            self.execution_engine,
            self.strategies,
            self.execution_analytics.fill_history,
        )

        performance_attribution = performance_analyzer.calculate_enhanced_pnl_attribution(
            self.total_pnl
        )

        strategy_latency_impact = (
            performance_analyzer.analyze_strategy_latency_impact()
        )

        market_condition_analysis = (
            self.market_analyzer.analyze_market_condition_impact(latency_snapshots)
        )

        optimization_recommendations = (
            performance_analyzer.generate_optimization_recommendations(
                strategy_latency_impact
            )
        )

        enhanced_results = {
            **base_results,
            "latency_analysis": latency_analysis,
            "performance_attribution": performance_attribution,
            "strategy_latency_impact": strategy_latency_impact,
            "market_condition_analysis": market_condition_analysis,
            "optimization_recommendations": optimization_recommendations,
            "latency_snapshots": latency_snapshots,
        }

        return enhanced_results
