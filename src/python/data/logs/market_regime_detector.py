import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Deque
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import time
import json
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
class MarketRegime(Enum):

    NORMAL = "normal"
    VOLATILE = "volatile"
    TRENDING = "trending"
    QUIET = "quiet"
    STRESSED = "stressed"
    AUCTION = "auction"
    UNKNOWN = "unknown"
@dataclass
class RegimeDetection:

    timestamp: float
    regime: MarketRegime
    confidence: float
    features: Dict[str, float]
    regime_probabilities: Dict[MarketRegime, float]
    change_detected: bool
    previous_regime: Optional[MarketRegime]
@dataclass
class ModelUpdate:

    timestamp: float
    model_name: str
    version: str
    update_type: str
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    samples_processed: int
    update_time_ms: float
class MarketRegimeDetector:

    def __init__(self, n_regimes: int = 3, feature_window: int = 1000):
        self.n_regimes = min(n_regimes, 3)
        self.feature_window = feature_window

        self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        self.gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
        self.scaler = StandardScaler()

        self.feature_names = [
            'volatility', 'volume_ratio', 'spread_ratio', 'order_imbalance',
            'trade_intensity', 'price_momentum', 'bid_ask_ratio', 'liquidity_score',
            'cross_venue_dispersion', 'microstructure_noise', 'hurst_exponent',
            'entropy', 'autocorrelation', 'jump_intensity', 'volume_clock_speed'
        ]

        self.regime_map = {}
        self.regime_characteristics = {}

        self.feature_buffer = deque(maxlen=feature_window)
        self.regime_history = deque(maxlen=1000)
        self.transition_matrix = np.zeros((self.n_regimes, self.n_regimes))

        self.current_regime = MarketRegime.UNKNOWN
        self.regime_start_time = time.time()
        self.regime_durations = {regime: [] for regime in MarketRegime}

        self.detection_confidence_threshold = 0.7
        self.is_trained = False
        self.primary_model = 'gmm'

        logger.info(f"MarketRegimeDetector initialized with {self.n_regimes} regimes")

    def extract_regime_features(self, market_data: Dict[str, Any]) -> np.ndarray:

        features = []

        if 'prices' in market_data:
            prices = np.array(market_data['prices'])
            if len(prices) > 1:
                returns = np.diff(np.log(prices + 1e-8))

                volatility = np.std(returns) * np.sqrt(252 * 390 * 60)
                features.append(volatility)
            else:
                features.append(0.01)
        else:
            features.append(0.01)

        if 'volumes' in market_data:
            volumes = np.array(market_data['volumes'])
            volume_ratio = volumes[-1] / np.mean(volumes) if len(volumes) > 0 else 1.0
            features.append(volume_ratio)
        else:
            features.append(1.0)

        if 'spreads' in market_data:
            spreads = np.array(market_data['spreads'])
            if 'prices' in market_data and len(prices) > 0:
                spread_ratio = np.mean(spreads) / (np.mean(prices) + 1e-8)
            else:
                spread_ratio = 0.0001
            features.append(spread_ratio)
        else:
            features.append(0.0001)

        features.append(market_data.get('order_imbalance', 0.0))

        if 'trade_counts' in market_data:
            trade_intensity = len(market_data['trade_counts']) / self.feature_window
            features.append(trade_intensity)
        else:
            features.append(0.5)

        if 'prices' in market_data and len(prices) > 20:
            momentum = (prices[-1] - prices[-20]) / prices[-20]
            features.append(momentum)
        else:
            features.append(0.0)

        if 'bid_sizes' in market_data and 'ask_sizes' in market_data:
            bid_sizes = np.array(market_data['bid_sizes'])
            ask_sizes = np.array(market_data['ask_sizes'])
            bid_ask_ratio = np.sum(bid_sizes) / (np.sum(ask_sizes) + 1e-8)
            liquidity_score = np.log1p(np.sum(bid_sizes) + np.sum(ask_sizes))
            features.extend([bid_ask_ratio, liquidity_score])
        else:
            features.extend([1.0, 10.0])

        if 'venue_prices' in market_data:
            venue_prices = market_data['venue_prices']
            if len(venue_prices) > 1:
                price_dispersion = np.std(list(venue_prices.values())) / np.mean(list(venue_prices.values()))
            else:
                price_dispersion = 0.0
            features.append(price_dispersion)
        else:
            features.append(0.0)

        while len(features) < len(self.feature_names):
            features.append(0.0)

        return np.array(features[:len(self.feature_names)], dtype=np.float32)

    def _estimate_garch_volatility(self, returns: np.ndarray) -> float:

        if len(returns) < 10:
            return np.std(returns)

        omega = 0.00001
        alpha = 0.1
        beta = 0.85

        variance = np.var(returns)

        for r in returns[-10:]:
            variance = omega + alpha * r**2 + beta * variance

        return np.sqrt(variance)

    def train(self, historical_data: List[Dict[str, Any]]):

        logger.info("Training market regime detector...")

        feature_matrix = []
        for data_point in historical_data:
            features = self.extract_regime_features(data_point)
            feature_matrix.append(features)

        feature_matrix = np.array(feature_matrix)

        scaled_features = self.scaler.fit_transform(feature_matrix)

        self.kmeans.fit(scaled_features)
        self.gmm.fit(scaled_features)

        kmeans_score = silhouette_score(scaled_features, self.kmeans.labels_)
        gmm_labels = self.gmm.predict(scaled_features)
        gmm_score = silhouette_score(scaled_features, gmm_labels)

        logger.info(f"K-means silhouette score: {kmeans_score:.3f}")
        logger.info(f"GMM silhouette score: {gmm_score:.3f}")

        if gmm_score > kmeans_score:
            labels = gmm_labels
            self.primary_model = 'gmm'
        else:
            labels = self.kmeans.labels_
            self.primary_model = 'kmeans'

        self._characterize_regimes(feature_matrix, labels)

        self._build_transition_matrix(labels)

        self.is_trained = True
        logger.info("Market regime detector training complete")

    def _characterize_regimes(self, features: np.ndarray, labels: np.ndarray):

        for regime_id in range(self.n_regimes):
            regime_mask = labels == regime_id
            regime_features = features[regime_mask]

            if len(regime_features) > 0:
                feature_means = np.mean(regime_features, axis=0)
                feature_stds = np.std(regime_features, axis=0)

                characteristics = {}
                for i, feature_name in enumerate(self.feature_names):
                    characteristics[feature_name] = {
                        'mean': float(feature_means[i]),
                        'std': float(feature_stds[i])
                    }

                regime_type = self._classify_regime_type(characteristics)
                self.regime_map[regime_id] = regime_type
                self.regime_characteristics[regime_type] = characteristics

                logger.info(f"Regime {regime_id} classified as {regime_type.value}")

    def _classify_regime_type(self, characteristics: Dict) -> MarketRegime:

        volatility = characteristics.get('volatility', {}).get('mean', 0.01)
        volume_ratio = characteristics.get('volume_ratio', {}).get('mean', 1.0)
        spread_ratio = characteristics.get('spread_ratio', {}).get('mean', 0.0001)
        momentum = characteristics.get('price_momentum', {}).get('mean', 0.0)

        if volatility > 0.03 and volume_ratio > 2.0:
            return MarketRegime.STRESSED
        elif volatility > 0.02:
            return MarketRegime.VOLATILE
        elif abs(momentum) > 0.02:
            return MarketRegime.TRENDING
        elif volume_ratio < 0.5 and spread_ratio > 0.0002:
            return MarketRegime.QUIET
        elif spread_ratio > 0.0005:
            return MarketRegime.AUCTION
        else:
            return MarketRegime.NORMAL

    def train(self, historical_data: List[Dict[str, Any]]):

        logger.info("Training market regime detector...")

        feature_matrix = []
        for data_point in historical_data:
            features = self.extract_regime_features(data_point)
            feature_matrix.append(features)

        feature_matrix = np.array(feature_matrix)
        n_samples = len(feature_matrix)

        if n_samples < 10:
            self.n_regimes = 2
            logger.warning(f"Small dataset ({n_samples} samples), using {self.n_regimes} regimes")
        elif n_samples < 50:
            self.n_regimes = 2
        else:
            self.n_regimes = min(3, n_samples // 20)

        self.n_regimes = max(2, min(self.n_regimes, n_samples - 1))

        self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        self.gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
        self.transition_matrix = np.zeros((self.n_regimes, self.n_regimes))

        try:
            scaled_features = self.scaler.fit_transform(feature_matrix)

            self.kmeans.fit(scaled_features)
            self.gmm.fit(scaled_features)

            kmeans_score = silhouette_score(scaled_features, self.kmeans.labels_)
            gmm_labels = self.gmm.predict(scaled_features)
            gmm_score = silhouette_score(scaled_features, gmm_labels)

            logger.info(f"K-means silhouette score: {kmeans_score:.3f}")
            logger.info(f"GMM silhouette score: {gmm_score:.3f}")

            if gmm_score > kmeans_score:
                labels = gmm_labels
                self.primary_model = 'gmm'
            else:
                labels = self.kmeans.labels_
                self.primary_model = 'kmeans'

            self._characterize_regimes(feature_matrix, labels)

            self._build_transition_matrix(labels)

            self.is_trained = True
            logger.info(f"✅ Market regime detector training complete with {self.n_regimes} regimes")
            return True

        except Exception as e:
            logger.error(f"❌ Regime detection training failed: {e}")
            self.n_regimes = 2
            self.kmeans = KMeans(n_clusters=2, random_state=42)
            self.gmm = GaussianMixture(n_components=2, random_state=42)

            try:
                scaled_features = self.scaler.fit_transform(feature_matrix)
                self.kmeans.fit(scaled_features)
                self.gmm.fit(scaled_features)
                self.is_trained = True
                logger.info("✅ Fallback training successful with 2 regimes")
                return True
            except Exception as fallback_error:
                logger.error(f"❌ Even fallback training failed: {fallback_error}")
                return False

    def _build_transition_matrix(self, labels: np.ndarray):

        for i in range(len(labels) - 1):
            from_regime = labels[i]
            to_regime = labels[i + 1]
            self.transition_matrix[from_regime, to_regime] += 1

        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(
            self.transition_matrix,
            row_sums,
            where=row_sums != 0
        )

    def detect_regime(self, market_data: Dict[str, Any]) -> RegimeDetection:

        if not self.is_trained:
            return RegimeDetection(
                timestamp=time.time(),
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                features={},
                regime_probabilities={regime: 0.0 for regime in MarketRegime},
                change_detected=False,
                previous_regime=None
            )

        features = self.extract_regime_features(market_data)
        self.feature_buffer.append(features)

        scaled_features = self.scaler.transform(features.reshape(1, -1))

        if self.primary_model == 'gmm':
            probabilities = self.gmm.predict_proba(scaled_features)[0]
            regime_id = np.argmax(probabilities)
            confidence = float(probabilities[regime_id])
        else:
            regime_id = self.kmeans.predict(scaled_features)[0]
            distances = np.linalg.norm(
                scaled_features - self.kmeans.cluster_centers_, axis=1
            )
            confidence = 1.0 / (1.0 + distances[regime_id])
            probabilities = np.zeros(self.n_regimes)
            probabilities[regime_id] = confidence

        detected_regime = self.regime_map.get(regime_id, MarketRegime.UNKNOWN)

        regime_probs = {}
        for i, prob in enumerate(probabilities):
            regime_type = self.regime_map.get(i, MarketRegime.UNKNOWN)
            regime_probs[regime_type] = float(prob)

        previous_regime = self.current_regime
        change_detected = (
            detected_regime != self.current_regime and
            confidence > self.detection_confidence_threshold
        )

        if change_detected:
            duration = time.time() - self.regime_start_time
            self.regime_durations[self.current_regime].append(duration)

            self.current_regime = detected_regime
            self.regime_start_time = time.time()

            logger.info(f"Regime change detected: {previous_regime.value} → {detected_regime.value}")

        detection = RegimeDetection(
            timestamp=time.time(),
            regime=detected_regime,
            confidence=confidence,
            features={name: float(features[i]) for i, name in enumerate(self.feature_names)},
            regime_probabilities=regime_probs,
            change_detected=change_detected,
            previous_regime=previous_regime if change_detected else None
        )

        self.regime_history.append(detection)

        return detection

    def get_regime_forecast(self, horizon: int = 10) -> Dict[MarketRegime, float]:

        if not self.is_trained or self.current_regime == MarketRegime.UNKNOWN:
            return {regime: 1.0 / len(MarketRegime) for regime in MarketRegime}

        current_regime_id = None
        for regime_id, regime_type in self.regime_map.items():
            if regime_type == self.current_regime:
                current_regime_id = regime_id
                break

        if current_regime_id is None:
            return {regime: 0.0 for regime in MarketRegime}

        current_probs = np.zeros(self.n_regimes)
        current_probs[current_regime_id] = 1.0

        future_probs = current_probs @ np.linalg.matrix_power(
            self.transition_matrix, horizon
        )

        regime_forecast = {}
        for regime_id, prob in enumerate(future_probs):
            regime_type = self.regime_map.get(regime_id, MarketRegime.UNKNOWN)
            regime_forecast[regime_type] = float(prob)

        return regime_forecast

    def adapt_strategy_parameters(self, regime: MarketRegime) -> Dict[str, Any]:

        params = {
            'latency_threshold_us': 1000,
            'min_edge_bps': 2.0,
            'position_size_multiplier': 1.0,
            'max_positions': 10,
            'routing_urgency': 0.5,
            'use_aggressive_routing': False,
            'enable_market_making': True,
            'risk_multiplier': 1.0
        }

        if regime == MarketRegime.VOLATILE:
            params.update({
                'latency_threshold_us': 500,
                'min_edge_bps': 5.0,
                'position_size_multiplier': 0.5,
                'max_positions': 5,
                'routing_urgency': 0.8,
                'risk_multiplier': 0.5
            })

        elif regime == MarketRegime.STRESSED:
            params.update({
                'latency_threshold_us': 300,
                'min_edge_bps': 10.0,
                'position_size_multiplier': 0.2,
                'max_positions': 3,
                'routing_urgency': 1.0,
                'use_aggressive_routing': True,
                'enable_market_making': False,
                'risk_multiplier': 0.2
            })

        elif regime == MarketRegime.QUIET:
            params.update({
                'latency_threshold_us': 2000,
                'min_edge_bps': 1.0,
                'position_size_multiplier': 1.5,
                'max_positions': 20,
                'routing_urgency': 0.3,
                'risk_multiplier': 1.5
            })

        elif regime == MarketRegime.TRENDING:
            params.update({
                'latency_threshold_us': 700,
                'min_edge_bps': 3.0,
                'position_size_multiplier': 1.2,
                'routing_urgency': 0.7,
                'use_aggressive_routing': True
            })

        elif regime == MarketRegime.AUCTION:
            params.update({
                'latency_threshold_us': 5000,
                'min_edge_bps': 0.5,
                'position_size_multiplier': 2.0,
                'enable_market_making': False
            })

        return params

    def get_regime_statistics(self) -> Dict[str, Any]:

        stats = {
            'current_regime': self.current_regime.value,
            'regime_duration': time.time() - self.regime_start_time,
            'regime_counts': {},
            'average_durations': {},
            'transition_probabilities': {},
            'regime_characteristics': {}
        }

        for detection in self.regime_history:
            regime = detection.regime.value
            stats['regime_counts'][regime] = stats['regime_counts'].get(regime, 0) + 1

        for regime, durations in self.regime_durations.items():
            if durations:
                stats['average_durations'][regime.value] = np.mean(durations)

        if self.is_trained:
            stats['transition_probabilities'] = {
                f"{self.regime_map.get(i, MarketRegime.UNKNOWN).value}_to_{self.regime_map.get(j, MarketRegime.UNKNOWN).value}":
                float(self.transition_matrix[i, j])
                for i in range(self.n_regimes)
                for j in range(self.n_regimes)
                if self.transition_matrix[i, j] > 0.01
            }

        for regime, chars in self.regime_characteristics.items():
            stats['regime_characteristics'][regime.value] = {
                'volatility': chars.get('volatility', {}).get('mean', 0),
                'volume_ratio': chars.get('volume_ratio', {}).get('mean', 0),
                'spread_ratio': chars.get('spread_ratio', {}).get('mean', 0)
            }

        return stats
class OnlineLearner:


    def __init__(self, models: Dict[str, Any], update_frequency: int = 100):
        self.models = models
        self.update_frequency = update_frequency

        self.update_history: List[ModelUpdate] = []
        self.sample_buffer: Dict[str, List[Tuple[np.ndarray, float]]] = {
            model_name: [] for model_name in models
        }

        self.performance_window = 1000
        self.performance_buffer: Dict[str, deque] = {
            model_name: deque(maxlen=self.performance_window)
            for model_name in models
        }

        self.drift_detector = DriftDetector()
        self.drift_threshold = 0.1

        self.model_versions: Dict[str, str] = {
            model_name: f"v1.0_{int(time.time())}"
            for model_name in models
        }
        self.model_checkpoints: Dict[str, List[Dict]] = {
            model_name: [] for model_name in models
        }

        self.ab_tests: Dict[str, 'ABTest'] = {}

        logger.info(f"OnlineLearner initialized with {len(models)} models")

    def update(self, model_name: str, features: np.ndarray,
               actual_value: float, predicted_value: float):

        if model_name not in self.models:
            return

        self.sample_buffer[model_name].append((features, actual_value))

        error = abs(actual_value - predicted_value) / (actual_value + 1e-8)
        self.performance_buffer[model_name].append(error)

        if len(self.performance_buffer[model_name]) >= 100:
            drift_score = self.drift_detector.detect_drift(
                list(self.performance_buffer[model_name])
            )

            if drift_score > self.drift_threshold:
                logger.warning(f"Drift detected in {model_name}: score={drift_score:.3f}")
                self._trigger_adaptation(model_name)

        if len(self.sample_buffer[model_name]) >= self.update_frequency:
            self._perform_online_update(model_name)

    def _perform_online_update(self, model_name: str):

        start_time = time.time()

        current_performance = self._evaluate_model_performance(model_name)

        features_batch = np.array([f for f, _ in self.sample_buffer[model_name]])
        targets_batch = np.array([t for _, t in self.sample_buffer[model_name]])

        checkpoint = self._create_checkpoint(model_name)

        try:
            if hasattr(self.models[model_name], 'partial_fit'):
                self.models[model_name].partial_fit(features_batch, targets_batch)
            elif hasattr(self.models[model_name], 'update_online'):
                self.models[model_name].update_online(features_batch, targets_batch)
            else:
                logger.warning(f"Model {model_name} does not support online updates")
                return

            new_performance = self._evaluate_model_performance(model_name)

            if new_performance['accuracy'] < current_performance['accuracy'] * 0.95:
                logger.warning(f"Performance degradation detected for {model_name}")
                self._rollback_model(model_name, checkpoint)
                new_performance = current_performance
            else:
                self.model_versions[model_name] = f"v{self._get_next_version(model_name)}_{int(time.time())}"
                self.model_checkpoints[model_name].append(checkpoint)

                if len(self.model_checkpoints[model_name]) > 5:
                    self.model_checkpoints[model_name].pop(0)

            update = ModelUpdate(
                timestamp=time.time(),
                model_name=model_name,
                version=self.model_versions[model_name],
                update_type='online',
                performance_before=current_performance,
                performance_after=new_performance,
                samples_processed=len(self.sample_buffer[model_name]),
                update_time_ms=(time.time() - start_time) * 1000
            )

            self.update_history.append(update)
            logger.info(f"Online update completed for {model_name}: "
                       f"accuracy {current_performance['accuracy']:.1f}% → {new_performance['accuracy']:.1f}%")

        except Exception as e:
            logger.error(f"Online update failed for {model_name}: {e}")
            self._rollback_model(model_name, checkpoint)

        finally:
            self.sample_buffer[model_name].clear()

    def _evaluate_model_performance(self, model_name: str) -> Dict[str, float]:

        if not self.performance_buffer[model_name]:
            return {'accuracy': 0.0, 'mae': float('inf')}

        errors = list(self.performance_buffer[model_name])

        mae = np.mean(errors)
        accuracy = np.mean([e < 0.1 for e in errors]) * 100

        return {
            'accuracy': accuracy,
            'mae': mae,
            'sample_count': len(errors)
        }

    def _create_checkpoint(self, model_name: str) -> Dict:

        checkpoint = {
            'timestamp': time.time(),
            'version': self.model_versions[model_name],
            'performance': self._evaluate_model_performance(model_name)
        }

        if hasattr(self.models[model_name], 'get_params'):
            checkpoint['params'] = self.models[model_name].get_params()
        elif hasattr(self.models[model_name], 'state_dict'):
            checkpoint['state_dict'] = self.models[model_name].state_dict()

        return checkpoint

    def _rollback_model(self, model_name: str, checkpoint: Dict):

        logger.info(f"Rolling back {model_name} to {checkpoint['version']}")

        if 'params' in checkpoint:
            self.models[model_name].set_params(**checkpoint['params'])
        elif 'state_dict' in checkpoint:
            self.models[model_name].load_state_dict(checkpoint['state_dict'])

        self.model_versions[model_name] = checkpoint['version']

    def _get_next_version(self, model_name: str) -> str:

        current_version = self.model_versions[model_name]
        version_parts = current_version.split('_')[0].split('.')

        try:
            major = int(version_parts[0][1:])
            minor = int(version_parts[1]) + 1
            return f"{major}.{minor}"
        except:
            return "1.1"

    def _trigger_adaptation(self, model_name: str):

        logger.info(f"Triggering adaptation for {model_name} due to drift")

        if len(self.sample_buffer[model_name]) > 10:
            self._perform_online_update(model_name)

    def start_ab_test(self, test_name: str, model_a: str, model_b: str,
                     duration_seconds: int = 3600):

        if model_a not in self.models or model_b not in self.models:
            logger.error(f"Invalid models for A/B test: {model_a}, {model_b}")
            return

        ab_test = ABTest(
            name=test_name,
            model_a=model_a,
            model_b=model_b,
            start_time=time.time(),
            duration=duration_seconds
        )

        self.ab_tests[test_name] = ab_test
        logger.info(f"Started A/B test '{test_name}': {model_a} vs {model_b}")

    def get_ab_test_results(self, test_name: str) -> Optional[Dict[str, Any]]:

        if test_name not in self.ab_tests:
            return None

        ab_test = self.ab_tests[test_name]
        return ab_test.get_results()
class DriftDetector:


    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reference_window = deque(maxlen=window_size)
        self.detection_methods = ['ks_test', 'wasserstein', 'psi']

    def detect_drift(self, recent_errors: List[float]) -> float:

        if len(self.reference_window) < self.window_size:
            self.reference_window.extend(recent_errors[:self.window_size])
            return 0.0

        recent_window = recent_errors[-self.window_size:]
        reference = list(self.reference_window)

        scores = []

        from scipy import stats
        ks_statistic, ks_pvalue = stats.ks_2samp(reference, recent_window)
        scores.append(1 - ks_pvalue)

        wasserstein = stats.wasserstein_distance(reference, recent_window)
        scores.append(min(wasserstein, 1.0))

        psi = self._calculate_psi(reference, recent_window)
        scores.append(min(psi / 0.25, 1.0))

        drift_score = np.mean(scores)

        if drift_score < 0.1:
            self.reference_window.extend(recent_window[-10:])

        return drift_score

    def _calculate_psi(self, reference: List[float], recent: List[float]) -> float:

        min_val = min(min(reference), min(recent))
        max_val = max(max(reference), max(recent))
        bins = np.linspace(min_val, max_val, 11)

        ref_hist, _ = np.histogram(reference, bins=bins)
        recent_hist, _ = np.histogram(recent, bins=bins)

        ref_hist = ref_hist / len(reference)
        recent_hist = recent_hist / len(recent)

        psi = 0
        for i in range(len(ref_hist)):
            if ref_hist[i] > 0 and recent_hist[i] > 0:
                psi += (recent_hist[i] - ref_hist[i]) * np.log(recent_hist[i] / ref_hist[i])

        return abs(psi)
@dataclass
class ABTest:

    name: str
    model_a: str
    model_b: str
    start_time: float
    duration: float

    def __post_init__(self):
        self.results_a = {'predictions': 0, 'errors': [], 'latencies': []}
        self.results_b = {'predictions': 0, 'errors': [], 'latencies': []}
        self.allocation_ratio = 0.5

    def record_result(self, model: str, error: float, latency_ms: float):

        if model == self.model_a:
            self.results_a['predictions'] += 1
            self.results_a['errors'].append(error)
            self.results_a['latencies'].append(latency_ms)
        elif model == self.model_b:
            self.results_b['predictions'] += 1
            self.results_b['errors'].append(error)
            self.results_b['latencies'].append(latency_ms)

    def get_results(self) -> Dict[str, Any]:

        elapsed = time.time() - self.start_time
        is_complete = elapsed >= self.duration

        results = {
            'test_name': self.name,
            'elapsed_seconds': elapsed,
            'is_complete': is_complete,
            'model_a': {
                'name': self.model_a,
                'predictions': self.results_a['predictions'],
                'mean_error': np.mean(self.results_a['errors']) if self.results_a['errors'] else 0,
                'std_error': np.std(self.results_a['errors']) if self.results_a['errors'] else 0,
                'mean_latency_ms': np.mean(self.results_a['latencies']) if self.results_a['latencies'] else 0
            },
            'model_b': {
                'name': self.model_b,
                'predictions': self.results_b['predictions'],
                'mean_error': np.mean(self.results_b['errors']) if self.results_b['errors'] else 0,
                'std_error': np.std(self.results_b['errors']) if self.results_b['errors'] else 0,
                'mean_latency_ms': np.mean(self.results_b['latencies']) if self.results_b['latencies'] else 0
            }
        }

        if len(self.results_a['errors']) > 30 and len(self.results_b['errors']) > 30:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(
                self.results_a['errors'],
                self.results_b['errors']
            )
            results['statistical_significance'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'is_significant': p_value < 0.05
            }

            if p_value < 0.05:
                if results['model_a']['mean_error'] < results['model_b']['mean_error']:
                    results['winner'] = self.model_a
                else:
                    results['winner'] = self.model_b
            else:
                results['winner'] = 'no_significant_difference'

        return results
class ModelManager:


    def __init__(self):
        self.models = {}
        self.active_models = {}
        self.model_registry = {}
        self.deployment_history = []
        self.performance_thresholds = {
            'min_accuracy': 85.0,
            'max_latency_ms': 5.0,
            'max_error_rate': 0.15
        }

        logger.info("ModelManager initialized")

    def register_model(self, model_id: str, model_instance: Any,
                      model_type: str, metadata: Dict[str, Any] = None):

        self.model_registry[model_id] = {
            'instance': model_instance,
            'type': model_type,
            'metadata': metadata or {},
            'registered_at': time.time(),
            'version': metadata.get('version', 'v1.0') if metadata else 'v1.0',
            'status': 'registered',
            'performance_history': []
        }

        logger.info(f"Model registered: {model_id} (type: {model_type})")

    def deploy_model(self, model_id: str, venue: str = None) -> bool:

        if model_id not in self.model_registry:
            logger.error(f"Model {model_id} not found in registry")
            return False

        model_info = self.model_registry[model_id]

        if not self._validate_model(model_id):
            logger.error(f"Model {model_id} failed validation")
            return False

        deployment_key = venue or 'global'

        if deployment_key in self.active_models:
            self._backup_model(deployment_key)

        self.active_models[deployment_key] = model_id
        model_info['status'] = 'deployed'

        deployment = {
            'timestamp': time.time(),
            'model_id': model_id,
            'deployment_key': deployment_key,
            'action': 'deploy'
        }
        self.deployment_history.append(deployment)

        logger.info(f"Model {model_id} deployed to {deployment_key}")
        return True

    def _validate_model(self, model_id: str) -> bool:

        model_info = self.model_registry[model_id]

        required_methods = ['predict', 'get_performance_metrics']
        model_instance = model_info['instance']

        for method in required_methods:
            if not hasattr(model_instance, method):
                logger.error(f"Model {model_id} missing required method: {method}")
                return False

        return True

    def _backup_model(self, deployment_key: str):

        current_model_id = self.active_models.get(deployment_key)
        if current_model_id:
            backup_id = f"{current_model_id}_backup_{int(time.time())}"
            self.model_registry[backup_id] = self.model_registry[current_model_id].copy()
            self.model_registry[backup_id]['status'] = 'backup'
            logger.info(f"Created backup: {backup_id}")

    def rollback_model(self, deployment_key: str) -> bool:

        backups = [
            (model_id, info) for model_id, info in self.model_registry.items()
            if info['status'] == 'backup' and deployment_key in model_id
        ]

        if not backups:
            logger.error(f"No backup found for {deployment_key}")
            return False

        backups.sort(key=lambda x: x[1]['registered_at'], reverse=True)
        backup_id, backup_info = backups[0]

        self.active_models[deployment_key] = backup_id
        backup_info['status'] = 'deployed'

        deployment = {
            'timestamp': time.time(),
            'model_id': backup_id,
            'deployment_key': deployment_key,
            'action': 'rollback'
        }
        self.deployment_history.append(deployment)

        logger.info(f"Rolled back to {backup_id} for {deployment_key}")
        return True

    def get_active_model(self, deployment_key: str = 'global') -> Optional[Any]:

        model_id = self.active_models.get(deployment_key)
        if model_id and model_id in self.model_registry:
            return self.model_registry[model_id]['instance']
        return None

    def update_model_performance(self, model_id: str, metrics: Dict[str, float]):

        if model_id not in self.model_registry:
            return

        performance_entry = {
            'timestamp': time.time(),
            'metrics': metrics
        }

        self.model_registry[model_id]['performance_history'].append(performance_entry)

        if metrics.get('accuracy', 100) < self.performance_thresholds['min_accuracy']:
            logger.warning(f"Model {model_id} accuracy below threshold: {metrics['accuracy']:.1f}%")

        if metrics.get('latency_ms', 0) > self.performance_thresholds['max_latency_ms']:
            logger.warning(f"Model {model_id} latency above threshold: {metrics['latency_ms']:.1f}ms")

    def get_model_comparison(self) -> pd.DataFrame:

        comparison_data = []

        for model_id, model_info in self.model_registry.items():
            if model_info['performance_history']:
                latest_perf = model_info['performance_history'][-1]['metrics']

                comparison_data.append({
                    'model_id': model_id,
                    'type': model_info['type'],
                    'status': model_info['status'],
                    'version': model_info['version'],
                    'accuracy': latest_perf.get('accuracy', 0),
                    'latency_ms': latest_perf.get('latency_ms', 0),
                    'error_rate': latest_perf.get('error_rate', 0),
                    'last_updated': model_info['performance_history'][-1]['timestamp']
                })

        return pd.DataFrame(comparison_data)
class PerformanceMonitor:


    def __init__(self, alert_callback=None):
        self.metrics_buffer = {}
        self.alert_thresholds = {}
        self.alert_callback = alert_callback
        self.anomaly_detector = AnomalyDetector()

        self.tracked_metrics = [
            'prediction_accuracy',
            'inference_latency_ms',
            'throughput_per_second',
            'memory_usage_mb',
            'cpu_usage_percent',
            'error_rate',
            'drift_score'
        ]

        for metric in self.tracked_metrics:
            self.metrics_buffer[metric] = deque(maxlen=10000)

        logger.info("PerformanceMonitor initialized")

    def record_metric(self, metric_name: str, value: float,
                     timestamp: float = None, tags: Dict[str, str] = None):

        if metric_name not in self.metrics_buffer:
            self.metrics_buffer[metric_name] = deque(maxlen=10000)

        metric_entry = {
            'timestamp': timestamp or time.time(),
            'value': value,
            'tags': tags or {}
        }

        self.metrics_buffer[metric_name].append(metric_entry)

        if len(self.metrics_buffer[metric_name]) > 100:
            recent_values = [
                m['value'] for m in list(self.metrics_buffer[metric_name])[-100:]
            ]

            if self.anomaly_detector.is_anomaly(recent_values):
                self._trigger_alert(
                    metric_name,
                    value,
                    'anomaly_detected',
                    f"Anomalous value detected for {metric_name}: {value}"
                )

        self._check_thresholds(metric_name, value)

    def set_alert_threshold(self, metric_name: str, threshold_type: str,
                           threshold_value: float):

        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}

        self.alert_thresholds[metric_name][threshold_type] = threshold_value
        logger.info(f"Alert threshold set: {metric_name} {threshold_type} {threshold_value}")

    def _check_thresholds(self, metric_name: str, value: float):

        if metric_name not in self.alert_thresholds:
            return

        thresholds = self.alert_thresholds[metric_name]

        if 'min' in thresholds and value < thresholds['min']:
            self._trigger_alert(
                metric_name, value, 'below_minimum',
                f"{metric_name} below minimum threshold: {value} < {thresholds['min']}"
            )

        if 'max' in thresholds and value > thresholds['max']:
            self._trigger_alert(
                metric_name, value, 'above_maximum',
                f"{metric_name} above maximum threshold: {value} > {thresholds['max']}"
            )

    def _trigger_alert(self, metric_name: str, value: float,
                      alert_type: str, message: str):

        alert = {
            'timestamp': time.time(),
            'metric_name': metric_name,
            'value': value,
            'alert_type': alert_type,
            'message': message
        }

        logger.warning(f"Performance alert: {message}")

        if self.alert_callback:
            self.alert_callback(alert)

    def get_metric_summary(self, metric_name: str,
                          window_minutes: int = 60) -> Dict[str, float]:

        if metric_name not in self.metrics_buffer:
            return {}

        current_time = time.time()
        window_start = current_time - (window_minutes * 60)

        recent_entries = [
            entry for entry in self.metrics_buffer[metric_name]
            if entry['timestamp'] >= window_start
        ]

        if not recent_entries:
            return {}

        values = [entry['value'] for entry in recent_entries]

        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }

    def get_dashboard_data(self) -> Dict[str, Any]:

        dashboard_data = {
            'metrics': {},
            'alerts': [],
            'system_health': self._calculate_system_health()
        }

        for metric in self.tracked_metrics:
            dashboard_data['metrics'][metric] = self.get_metric_summary(metric, 5)

        dashboard_data['alerts'] = []

        return dashboard_data

    def _calculate_system_health(self) -> float:

        health_score = 100.0

        if 'error_rate' in self.metrics_buffer:
            error_summary = self.get_metric_summary('error_rate', 5)
            if error_summary and error_summary['mean'] > 0.05:
                health_score -= 20 * error_summary['mean']

        if 'inference_latency_ms' in self.metrics_buffer:
            latency_summary = self.get_metric_summary('inference_latency_ms', 5)
            if latency_summary and latency_summary['p95'] > 5.0:
                health_score -= 10 * (latency_summary['p95'] / 5.0 - 1)

        if 'drift_score' in self.metrics_buffer:
            drift_summary = self.get_metric_summary('drift_score', 5)
            if drift_summary and drift_summary['mean'] > 0.1:
                health_score -= 30 * drift_summary['mean']

        return max(0, min(100, health_score))
class AnomalyDetector:


    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.zscore_threshold = 3.0

    def is_anomaly(self, values: List[float]) -> bool:

        if len(values) < 10:
            return False

        mean = np.mean(values[:-1])
        std = np.std(values[:-1])

        if std == 0:
            return False

        zscore = abs(values[-1] - mean) / std
        return zscore > self.zscore_threshold
class Phase2CIntegration:


    def __init__(self, latency_predictor, ensemble_model, rl_router):
        self.latency_predictor = latency_predictor
        self.ensemble_model = ensemble_model
        self.rl_router = rl_router

        self.market_regime_detector = MarketRegimeDetector()
        self.online_learner = OnlineLearner({
            'latency_predictor': latency_predictor,
            'ensemble_model': ensemble_model
        })
        self.model_manager = ModelManager()
        self.performance_monitor = PerformanceMonitor()

        self._register_models()

        logger.info("Phase 2C Integration initialized")

    def _register_models(self):

        self.model_manager.register_model(
            'latency_predictor_v1',
            self.latency_predictor,
            'latency_prediction',
            {'version': 'v1.0', 'framework': 'pytorch'}
        )

        self.model_manager.register_model(
            'ensemble_model_v1',
            self.ensemble_model,
            'ensemble_prediction',
            {'version': 'v1.0', 'models': ['lstm', 'gru', 'xgboost']}
        )

        self.model_manager.register_model(
            'rl_router_v1',
            self.rl_router,
            'routing_optimization',
            {'version': 'v1.0', 'algorithm': 'dqn'}
        )

    async def process_market_update(self, market_data: Dict[str, Any]) -> Dict[str, Any]:

        start_time = time.time()

        regime_detection = self.market_regime_detector.detect_regime(market_data)

        strategy_params = self.market_regime_detector.adapt_strategy_parameters(
            regime_detection.regime
        )

        routing_decision = self.rl_router.make_routing_decision(
            market_data['symbol'],
            urgency=strategy_params['routing_urgency']
        )

        inference_time = (time.time() - start_time) * 1000
        self.performance_monitor.record_metric(
            'inference_latency_ms',
            inference_time,
            tags={'regime': regime_detection.regime.value}
        )

        response = {
            'regime': regime_detection.regime.value,
            'regime_confidence': regime_detection.confidence,
            'routing_decision': routing_decision,
            'strategy_parameters': strategy_params,
            'processing_time_ms': inference_time
        }

        return response

    def update_models_online(self, features: np.ndarray,
                            actual_latency: float, predicted_latency: float,
                            venue: str):

        self.online_learner.update(
            'latency_predictor',
            features,
            actual_latency,
            predicted_latency
        )

        error = abs(actual_latency - predicted_latency) / actual_latency
        self.performance_monitor.record_metric(
            'prediction_accuracy',
            1.0 - error,
            tags={'venue': venue}
        )

        if error > 0.2:
            logger.warning(f"High prediction error for {venue}: {error:.1%}")

    def get_system_status(self) -> Dict[str, Any]:

        return {
            'regime_statistics': self.market_regime_detector.get_regime_statistics(),
            'model_comparison': self.model_manager.get_model_comparison().to_dict(),
            'performance_dashboard': self.performance_monitor.get_dashboard_data(),
            'online_learning_status': {
                'updates_pending': {
                    model: len(buffer)
                    for model, buffer in self.online_learner.sample_buffer.items()
                },
                'model_versions': self.online_learner.model_versions
            }
        }