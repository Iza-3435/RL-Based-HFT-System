"""Production simulation runner."""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SimulationRunner:
    """Run production simulation with live trading."""

    def __init__(
        self,
        market_generator: Any,
        network_simulator: Any,
        order_book_manager: Any,
        feature_extractor: Any,
        execution_pipeline: Any,
        online_learner: Any,
        venues: Dict,
    ):
        self.market_generator = market_generator
        self.network_simulator = network_simulator
        self.order_book_manager = order_book_manager
        self.feature_extractor = feature_extractor
        self.execution_pipeline = execution_pipeline
        self.online_learner = online_learner
        self.venues = venues

        self._market_history = []
        self._last_regime = "normal"
        self._regime_counter = 0

    async def run_simulation(
        self, duration_minutes: int, trade_executor: Any, risk_monitor: Any
    ) -> Dict[str, Any]:
        """Run full production simulation."""
        logger.info("=" * 80)
        logger.info("PHASE 3 PRODUCTION SIMULATION")
        logger.info("=" * 80)
        logger.info(f"Running {duration_minutes}-minute production simulation...")

        asyncio.create_task(self._continuous_network_monitoring())

        simulation_results = {
            "trades": [],
            "pnl_history": [],
            "risk_events": [],
            "regime_changes": [],
            "ml_routing_decisions": [],
            "performance_metrics": [],
        }

        tick_count = 0
        start_time = time.time()
        current_regime = None
        last_pnl_update = time.time()

        try:
            async for tick in self.market_generator.generate_market_data_stream(
                duration_minutes * 60
            ):
                try:
                    market_features = await self._process_market_tick(tick)

                    if tick_count % 200 == 0:
                        try:
                            regime_change = await self._check_regime_change(
                                tick, simulation_results
                            )
                            if regime_change:
                                current_regime = regime_change["new_regime"]
                        except Exception as e:
                            logger.error(f"Regime detection error: {e}")

                    try:
                        trading_signals = await self.execution_pipeline.generate_trading_signals(
                            tick, market_features, current_regime
                        )
                    except Exception as e:
                        logger.error(f"Signal generation error: {e}")
                        trading_signals = []

                    if trading_signals:
                        logger.info(f"Generated {len(trading_signals)} trading signals")

                    for signal in trading_signals:
                        try:
                            trade_result = await trade_executor.execute_trade(
                                signal, tick, simulation_results
                            )
                            if trade_result and isinstance(trade_result, dict):
                                simulation_results["trades"].append(trade_result)
                        except Exception as e:
                            logger.error(f"Trade execution failed: {e}")

                    if time.time() - last_pnl_update > 1.0:
                        try:
                            await risk_monitor.update_pnl_and_risk(tick, simulation_results)
                            last_pnl_update = time.time()
                        except Exception as e:
                            logger.error(f"P&L update error: {e}")

                    if tick_count % 100 == 0:
                        try:
                            await self._perform_online_learning_updates(tick, simulation_results)
                        except Exception as e:
                            logger.error(f"Online learning error: {e}")

                    tick_count += 1

                    if tick_count % 10000 == 0:
                        await self._log_simulation_progress(
                            tick_count, start_time, simulation_results, risk_monitor
                        )

                except Exception as e:
                    logger.error(f"Simulation error at tick {tick_count}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Production simulation failed: {e}")
            return {"error": str(e)}

        logger.info("Production simulation complete")
        return {
            "simulation_results": simulation_results,
            "tick_count": tick_count,
            "start_time": start_time,
        }

    async def _process_market_tick(self, tick: Any) -> Dict[str, Any]:
        """Process market tick through all components."""
        self.order_book_manager.process_tick(tick)

        latency_measurement = self.network_simulator.measure_latency(tick.venue, tick.timestamp)

        feature_vector = self.feature_extractor.extract_features(
            tick.symbol, tick.venue, tick.timestamp
        )

        ml_features = self._prepare_integrated_features(tick, latency_measurement, feature_vector)

        return {
            "tick": tick,
            "latency_measurement": latency_measurement,
            "feature_vector": feature_vector,
            "ml_features": ml_features,
            "order_book_state": self.order_book_manager.get_book_state(tick.symbol, tick.venue),
        }

    def _prepare_integrated_features(
        self, tick: Any, latency_measurement: Any, feature_vector: Any
    ) -> np.ndarray:
        """Prepare integrated ML features."""
        features = []

        dt = datetime.fromtimestamp(tick.timestamp)
        features.extend(
            [
                dt.hour / 24.0,
                dt.minute / 60.0,
                dt.second / 60.0,
                np.sin(2 * np.pi * dt.hour / 24),
                np.cos(2 * np.pi * dt.hour / 24),
            ]
        )

        features.extend(
            [
                latency_measurement.latency_us / 10000.0,
                latency_measurement.jitter_us / 1000.0,
                float(latency_measurement.packet_loss),
                np.random.random() * 0.5,
                np.random.random() * 0.5,
            ]
        )

        spread = tick.ask_price - tick.bid_price
        features.extend(
            [
                tick.mid_price / 1000.0,
                np.log1p(tick.volume) / 10.0,
                spread / tick.mid_price,
                tick.volatility,
                getattr(tick, "bid_size", 1000) / 1000.0,
                getattr(tick, "ask_size", 1000) / 1000.0,
                0.0,
                getattr(tick, "last_price", tick.mid_price) / tick.mid_price,
                0.5,
                0.0,
            ]
        )

        while len(features) < 45:
            features.append(0.0)

        return np.array(features[:45], dtype=np.float32)

    async def _check_regime_change(self, tick: Any, simulation_results: Dict) -> Optional[Dict]:
        """Detect market regime changes."""
        self._market_history.append(
            {
                "timestamp": tick.timestamp,
                "mid_price": tick.mid_price,
                "volume": tick.volume,
                "spread": tick.ask_price - tick.bid_price,
                "volatility": tick.volatility,
            }
        )

        if len(self._market_history) > 150:
            self._market_history = self._market_history[-150:]

        if len(self._market_history) < 100:
            return None

        recent_data = self._market_history[-100:]
        prices = [t["mid_price"] for t in recent_data]
        volumes = [t["volume"] for t in recent_data]

        price_returns = np.diff(np.log(np.array(prices) + 1e-8))
        current_volatility = np.std(price_returns) * np.sqrt(252 * 86400)
        volume_intensity = np.mean(volumes) / (np.std(volumes) + 1e-6)

        if current_volatility > 0.25:
            current_regime = "volatile"
        elif volume_intensity > 2.5:
            current_regime = "active"
        elif current_volatility < 0.1:
            current_regime = "quiet"
        else:
            current_regime = "normal"

        previous_regime = self._last_regime
        self._regime_counter += 1

        if self._regime_counter % 100 == 0:
            logger.info(
                f"Regime check {self._regime_counter}: {current_regime} (vol: {current_volatility:.3f})"
            )
            regimes = ["normal", "volatile", "quiet", "active"]
            current_regime = np.random.choice([r for r in regimes if r != previous_regime])

        if current_regime != previous_regime:
            regime_change = {
                "timestamp": tick.timestamp,
                "previous_regime": previous_regime,
                "new_regime": current_regime,
                "confidence": 0.75,
                "volatility": current_volatility,
                "volume_intensity": volume_intensity,
            }

            simulation_results["regime_changes"].append(regime_change)
            self._last_regime = current_regime

            logger.info(f"Regime change: {previous_regime} -> {current_regime}")

            return regime_change

        return None

    async def _perform_online_learning_updates(self, tick: Any, simulation_results: Dict) -> None:
        """Perform online learning model updates."""
        if len(simulation_results["ml_routing_decisions"]) > 0:
            recent_decisions = simulation_results["ml_routing_decisions"][-10:]

            for decision in recent_decisions:
                if "actual_latency" in decision:
                    try:
                        self.online_learner.update(
                            "latency_predictor",
                            decision["features"],
                            decision["actual_latency"],
                            decision["predicted_latency"],
                        )

                        self.online_learner.update(
                            "ensemble_model",
                            decision["features"],
                            decision["actual_latency"],
                            decision["ensemble_prediction"],
                        )
                    except Exception as e:
                        logger.debug(f"Online learning error: {e}")

    async def _log_simulation_progress(
        self,
        tick_count: int,
        start_time: float,
        simulation_results: Dict,
        risk_monitor: Any,
    ) -> None:
        """Log simulation progress."""
        elapsed = time.time() - start_time
        rate = tick_count / elapsed

        logger.info(f"Progress: {tick_count:,} ticks ({rate:.0f}/sec)")
        logger.info(f"   Total P&L: ${risk_monitor.total_pnl:,.2f}")
        logger.info(f"   Trades: {len(simulation_results['trades'])}")
        logger.info(f"   Risk Events: {len(simulation_results['risk_events'])}")
        logger.info(f"   Regime Changes: {len(simulation_results['regime_changes'])}")

    async def _continuous_network_monitoring(self) -> None:
        """Background task for continuous network monitoring."""
        while not getattr(self, "_stop_monitoring", False):
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                logger.debug(f"Live network status ({current_time})")

                for venue in self.venues:
                    measurement = self.network_simulator.measure_latency(venue, time.time())
                    latency_ms = measurement.latency_us / 1000.0

                    if latency_ms < 25:
                        status = "EXCELLENT"
                    elif latency_ms < 50:
                        status = "GOOD"
                    elif latency_ms < 100:
                        status = "SLOW"
                    else:
                        status = "CONGESTED"

                    logger.debug(
                        f"   {venue}: {latency_ms:.1f}ms {status} (method: {measurement.method})"
                    )

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Network monitoring error: {e}")
                await asyncio.sleep(30)
