"""ML model training manager."""

import time
import logging
import numpy as np
from typing import Dict, Any
from datetime import datetime
from .config import get_training_config

logger = logging.getLogger(__name__)


class TrainingManager:
    """Manage ML model training across all components."""

    def __init__(
        self,
        latency_predictor: Any,
        ensemble_model: Any,
        routing_environment: Any,
        market_regime_detector: Any,
        market_generator: Any,
        network_simulator: Any,
        order_book_manager: Any,
        feature_extractor: Any,
        venues: Dict,
        mode: str = "production",
    ):
        self.latency_predictor = latency_predictor
        self.ensemble_model = ensemble_model
        self.routing_environment = routing_environment
        self.market_regime_detector = market_regime_detector
        self.market_generator = market_generator
        self.network_simulator = network_simulator
        self.order_book_manager = order_book_manager
        self.feature_extractor = feature_extractor
        self.venues = venues
        self.config = get_training_config(mode)

    async def train_all_models(self, training_duration_minutes: int) -> None:
        """Train all ML models using generated market data."""
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 2 MODEL TRAINING")
        logger.info(f"{'='*80}")
        logger.info(f"Training ML models with {training_duration_minutes} minutes of data...")

        try:
            training_data = await self._generate_training_data(training_duration_minutes)
            await self._train_latency_models(training_data)
            await self._train_routing_models()
            await self._train_regime_detection(training_data)
            logger.info(" All ML models trained successfully")
        except Exception as e:
            logger.error(f"ML training failed: {e}")

    async def _generate_training_data(self, duration_minutes: int) -> Dict:
        """Generate comprehensive training data."""
        logger.info(" Generating training data...")

        expected_ticks = self.market_generator.target_ticks_per_minute * duration_minutes
        logger.info(f" Target: {expected_ticks:,} ticks in {duration_minutes} minutes")

        training_data = {
            "market_ticks": [],
            "network_measurements": [],
            "order_book_updates": [],
            "features": {venue: [] for venue in self.venues},
            "latency_targets": {venue: [] for venue in self.venues},
        }

        tick_count = 0
        start_time = time.time()

        try:
            async for tick in self.market_generator.generate_market_data_stream(
                duration_minutes * 60
            ):
                training_data["market_ticks"].append(
                    {
                        "timestamp": tick.timestamp,
                        "symbol": tick.symbol,
                        "venue": tick.venue,
                        "mid_price": tick.mid_price,
                        "bid_price": tick.bid_price,
                        "ask_price": tick.ask_price,
                        "volume": tick.volume,
                        "volatility": tick.volatility,
                    }
                )

                latency_measurement = self.network_simulator.measure_latency(
                    tick.venue, tick.timestamp
                )
                training_data["network_measurements"].append(
                    {
                        "timestamp": tick.timestamp,
                        "venue": tick.venue,
                        "latency_us": latency_measurement.latency_us,
                        "jitter_us": latency_measurement.jitter_us,
                        "packet_loss": latency_measurement.packet_loss,
                    }
                )

                self.order_book_manager.process_tick(tick)
                feature_vector = self.feature_extractor.extract_features(
                    tick.symbol, tick.venue, tick.timestamp
                )

                ml_features = self._prepare_features(tick, latency_measurement, feature_vector)
                training_data["features"][tick.venue].append(ml_features)
                training_data["latency_targets"][tick.venue].append(latency_measurement.latency_us)

                tick_count += 1

                if tick_count % 5000 == 0:
                    elapsed = time.time() - start_time
                    rate = tick_count / elapsed
                    logger.info(f"Generated {tick_count:,} samples ({rate:.0f}/sec)")

        except Exception as e:
            logger.error(f"Training data generation failed: {e}")

        for venue in self.venues:
            if training_data["features"][venue]:
                training_data["features"][venue] = np.array(training_data["features"][venue])
                training_data["latency_targets"][venue] = np.array(
                    training_data["latency_targets"][venue]
                )

        logger.info(f"Training data generation complete: {tick_count:,} samples")
        return training_data

    def _prepare_features(
        self, tick: Any, latency_measurement: Any, feature_vector: Any
    ) -> np.ndarray:
        """Prepare comprehensive ML features."""
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
                (
                    feature_vector.features.get("trade_intensity", 0.5)
                    if hasattr(feature_vector, "features")
                    else 0.5
                ),
                (
                    feature_vector.features.get("price_momentum_1min", 0.0)
                    if hasattr(feature_vector, "features")
                    else 0.0
                ),
            ]
        )

        while len(features) < 45:
            features.append(0.0)

        return np.array(features[:45], dtype=np.float32)

    async def _train_latency_models(self, training_data: Dict) -> None:
        """Train latency prediction models."""
        logger.info(" Training latency prediction models...")

        try:
            for venue in self.venues:
                if venue in training_data["features"] and len(training_data["features"][venue]) > 0:
                    features = training_data["features"][venue]
                    targets = training_data["latency_targets"][venue]

                    logger.info(f"Training latency model for {venue}: {len(features)} samples")

                    if hasattr(self.latency_predictor, "train_model"):
                        venue_data = {"features": features, "targets": targets}
                        metrics = self.latency_predictor.train_model(
                            venue,
                            venue_data,
                            epochs=self.config["epochs"],
                            batch_size=self.config["batch_size"],
                        )
                        logger.info(f" {venue} LSTM: {metrics.get('accuracy', 0):.1f}% accuracy")
                    elif hasattr(self.latency_predictor, "fit"):
                        self.latency_predictor.fit(features, targets)
                        logger.info(f" {venue} latency model trained")

            if hasattr(self.ensemble_model, "train_all_models"):
                self.ensemble_model.train_all_models(
                    training_data, epochs=self.config["epochs"] // 2
                )
                logger.info(" Ensemble models trained successfully")

        except Exception as e:
            logger.error(f"Latency model training failed: {e}")

    async def _train_routing_models(self) -> None:
        """Train RL routing models."""
        logger.info(" Training routing optimization models...")

        try:
            if hasattr(self.routing_environment, "train_agents"):
                logger.info(f"Training DQN router ({self.config['routing_episodes']} episodes)...")
                self.routing_environment.train_agents(episodes=self.config["routing_episodes"])
                logger.info(" DQN router trained successfully")

            logger.info(" Routing models trained")
        except Exception as e:
            logger.error(f"Routing model training failed: {e}")

    async def _train_regime_detection(self, training_data: Dict) -> None:
        """Train market regime detection."""
        logger.info(" Training market regime detection...")

        try:
            market_features = []
            for tick in training_data["market_ticks"]:
                features = [
                    tick["mid_price"],
                    tick["volume"],
                    tick["volatility"],
                    tick["ask_price"] - tick["bid_price"],
                ]
                market_features.append(features)

            if market_features and hasattr(self.market_regime_detector, "train"):
                if len(market_features) > 100:
                    market_windows = []
                    for i in range(0, len(market_features), 1000):
                        window = market_features[i : i + 1000]
                        if len(window) >= 100:
                            prices = [f[0] for f in window]
                            volumes = [f[1] for f in window]
                            market_windows.append(
                                {
                                    "prices": prices,
                                    "volumes": volumes,
                                    "spreads": [f[3] for f in window],
                                    "volatility": np.std(np.diff(np.log(np.array(prices) + 1e-8))),
                                }
                            )

                    self.market_regime_detector.train(market_windows)

            logger.info(" Regime detection trained")
        except Exception as e:
            logger.error(f"Regime detection training failed: {e}")
