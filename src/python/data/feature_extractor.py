import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from enum import Enum
import json
from scipy import stats
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class FeatureVector:

    timestamp: float
    symbol: str
    venue: str
    target_latency: Optional[float] = None
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
class FeatureCategory(Enum):

    TEMPORAL = "temporal"
    PRICE = "price"
    VOLUME = "volume"
    SPREAD = "spread"
    BOOK = "book"
    NETWORK = "network"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    MICROSTRUCTURE = "microstructure"
    CROSS_VENUE = "cross_venue"

@dataclass
class WindowConfig:

    window_size: int
    step_size: int = 1
    min_periods: int = 1
class FeatureExtractor:


    def __init__(self, symbols: List[str], venues: List[str],
                 lookback_windows: Dict[str, int] = None):
        self.symbols = symbols
        self.venues = venues

        self.lookback_windows = lookback_windows or {
            'micro': 10,
            'short': 50,
            'medium': 200,
            'long': 1000
        }

        self.tick_buffers: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=max(self.lookback_windows.values()))
        )

        self.order_book_buffers: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=max(self.lookback_windows.values()))
        )

        self.network_buffers: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=max(self.lookback_windows.values()))
        )

        self.feature_configs = self._initialize_feature_configs()

        self.feature_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.feature_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        self.extraction_count = 0
        self.extraction_times = deque(maxlen=1000)

        logger.info(f"FeatureExtractor initialized: {len(symbols)} symbols × {len(venues)} venues")
        logger.info(f"Lookback windows: {self.lookback_windows}")

    def _initialize_feature_configs(self) -> Dict[str, Dict]:

        return {
            'hour_of_day': {'category': FeatureCategory.TEMPORAL, 'normalize': False},
            'minute_of_hour': {'category': FeatureCategory.TEMPORAL, 'normalize': False},
            'seconds_since_open': {'category': FeatureCategory.TEMPORAL, 'normalize': True},
            'time_to_close': {'category': FeatureCategory.TEMPORAL, 'normalize': True},
            'day_of_week': {'category': FeatureCategory.TEMPORAL, 'normalize': False},

            'mid_price': {'category': FeatureCategory.PRICE, 'normalize': True},
            'log_price': {'category': FeatureCategory.PRICE, 'normalize': True},
            'price_change_1': {'category': FeatureCategory.PRICE, 'normalize': True},
            'price_change_5': {'category': FeatureCategory.PRICE, 'normalize': True},
            'price_change_10': {'category': FeatureCategory.PRICE, 'normalize': True},
            'price_return_1': {'category': FeatureCategory.PRICE, 'normalize': True},
            'price_return_5': {'category': FeatureCategory.PRICE, 'normalize': True},

            'volatility_micro': {'category': FeatureCategory.VOLATILITY, 'normalize': True},
            'volatility_short': {'category': FeatureCategory.VOLATILITY, 'normalize': True},
            'volatility_medium': {'category': FeatureCategory.VOLATILITY, 'normalize': True},
            'realized_volatility': {'category': FeatureCategory.VOLATILITY, 'normalize': True},
            'volatility_regime': {'category': FeatureCategory.VOLATILITY, 'normalize': False},

            'bid_ask_spread': {'category': FeatureCategory.SPREAD, 'normalize': True},
            'spread_bps': {'category': FeatureCategory.SPREAD, 'normalize': True},
            'relative_spread': {'category': FeatureCategory.SPREAD, 'normalize': True},
            'spread_volatility': {'category': FeatureCategory.SPREAD, 'normalize': True},

            'book_imbalance': {'category': FeatureCategory.BOOK, 'normalize': True},
            'weighted_mid_price': {'category': FeatureCategory.BOOK, 'normalize': True},
            'book_pressure': {'category': FeatureCategory.BOOK, 'normalize': True},
            'order_flow_imbalance': {'category': FeatureCategory.BOOK, 'normalize': True},
            'liquidity_ratio': {'category': FeatureCategory.BOOK, 'normalize': True},

            'volume': {'category': FeatureCategory.VOLUME, 'normalize': True},
            'volume_ma_ratio': {'category': FeatureCategory.VOLUME, 'normalize': True},
            'volume_volatility': {'category': FeatureCategory.VOLUME, 'normalize': True},
            'turnover_rate': {'category': FeatureCategory.VOLUME, 'normalize': True},

            'momentum_1': {'category': FeatureCategory.MOMENTUM, 'normalize': True},
            'momentum_5': {'category': FeatureCategory.MOMENTUM, 'normalize': True},
            'momentum_10': {'category': FeatureCategory.MOMENTUM, 'normalize': True},
            'rsi': {'category': FeatureCategory.MOMENTUM, 'normalize': False},
            'macd': {'category': FeatureCategory.MOMENTUM, 'normalize': True},

            'network_latency': {'category': FeatureCategory.NETWORK, 'normalize': True},
            'latency_percentile': {'category': FeatureCategory.NETWORK, 'normalize': False},
            'packet_loss_rate': {'category': FeatureCategory.NETWORK, 'normalize': True},
            'network_congestion': {'category': FeatureCategory.NETWORK, 'normalize': True},
            'jitter': {'category': FeatureCategory.NETWORK, 'normalize': True},

            'tick_direction': {'category': FeatureCategory.MICROSTRUCTURE, 'normalize': False},
            'effective_spread': {'category': FeatureCategory.MICROSTRUCTURE, 'normalize': True},
            'price_improvement': {'category': FeatureCategory.MICROSTRUCTURE, 'normalize': True},
            'market_impact': {'category': FeatureCategory.MICROSTRUCTURE, 'normalize': True},

            'venue_spread_rank': {'category': FeatureCategory.CROSS_VENUE, 'normalize': False},
            'cross_venue_momentum': {'category': FeatureCategory.CROSS_VENUE, 'normalize': True},
            'arbitrage_signal': {'category': FeatureCategory.CROSS_VENUE, 'normalize': True},
            'relative_liquidity': {'category': FeatureCategory.CROSS_VENUE, 'normalize': True}
        }

    def add_tick_data(self, symbol: str, venue: str, tick_data: Dict):

        key = (symbol, venue)
        self.tick_buffers[key].append(tick_data)

    def add_order_book_data(self, symbol: str, venue: str, book_data: Dict):

        key = (symbol, venue)
        self.order_book_buffers[key].append(book_data)

    def add_network_data(self, symbol: str, venue: str, network_data: Dict):

        key = (symbol, venue)
        self.network_buffers[key].append(network_data)

    def _calculate_temporal_features(self, timestamp: float) -> Dict[str, float]:

        dt = datetime.fromtimestamp(timestamp)

        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)

        seconds_since_open = (dt - market_open).total_seconds()
        seconds_to_close = (market_close - dt).total_seconds()

        return {
            'hour_of_day': dt.hour,
            'minute_of_hour': dt.minute,
            'seconds_since_open': max(0, seconds_since_open),
            'time_to_close': max(0, seconds_to_close),
            'day_of_week': dt.weekday(),
            'is_market_open': 1.0 if 9.5 <= (dt.hour + dt.minute/60) <= 16 else 0.0
        }

    def _calculate_price_features(self, symbol: str, venue: str) -> Dict[str, float]:

        key = (symbol, venue)
        ticks = list(self.tick_buffers[key])

        if len(ticks) < 2:
            return {}

        features = {}

        current_tick = ticks[-1]
        mid_price = current_tick.get('mid_price', 0)

        features['mid_price'] = mid_price
        features['log_price'] = np.log(mid_price) if mid_price > 0 else 0

        for window in [1, 5, 10]:
            if len(ticks) > window:
                past_price = ticks[-(window+1)].get('mid_price', 0)
                if past_price > 0:
                    price_change = mid_price - past_price
                    price_return = (mid_price / past_price - 1) if past_price > 0 else 0

                    features[f'price_change_{window}'] = price_change
                    features[f'price_return_{window}'] = price_return

        return features

    def _calculate_volatility_features(self, symbol: str, venue: str) -> Dict[str, float]:

        key = (symbol, venue)
        ticks = list(self.tick_buffers[key])

        if len(ticks) < 10:
            return {}

        features = {}

        mid_prices = np.array([tick.get('mid_price', 0) for tick in ticks])
        returns = np.diff(np.log(mid_prices[mid_prices > 0]))

        if len(returns) == 0:
            return features

        for window_name, window_size in self.lookback_windows.items():
            if len(returns) >= window_size:
                recent_returns = returns[-window_size:]
                vol = np.std(recent_returns) * np.sqrt(252 * 390 * 60)
                features[f'volatility_{window_name}'] = vol

        if len(returns) >= 50:
            realized_vol = np.sqrt(np.sum(returns[-50:] ** 2))
            features['realized_volatility'] = realized_vol

        if len(returns) >= 100:
            current_vol = np.std(returns[-20:]) if len(returns) >= 20 else 0
            long_term_vol = np.std(returns[-100:])

            if current_vol > long_term_vol * 1.5:
                features['volatility_regime'] = 2.0
            elif current_vol < long_term_vol * 0.7:
                features['volatility_regime'] = 0.0
            else:
                features['volatility_regime'] = 1.0

        return features

    def _calculate_spread_features(self, symbol: str, venue: str) -> Dict[str, float]:

        key = (symbol, venue)
        ticks = list(self.tick_buffers[key])

        if len(ticks) < 2:
            return {}

        features = {}
        current_tick = ticks[-1]

        bid_price = current_tick.get('bid_price', 0)
        ask_price = current_tick.get('ask_price', 0)
        mid_price = current_tick.get('mid_price', 0)

        if bid_price > 0 and ask_price > 0:
            spread = ask_price - bid_price
            features['bid_ask_spread'] = spread

            if mid_price > 0:
                features['spread_bps'] = (spread / mid_price) * 10000
                features['relative_spread'] = spread / mid_price

        if len(ticks) >= 20:
            spreads = []
            for tick in ticks[-20:]:
                bid = tick.get('bid_price', 0)
                ask = tick.get('ask_price', 0)
                if bid > 0 and ask > 0:
                    spreads.append(ask - bid)

            if spreads:
                features['spread_volatility'] = np.std(spreads)

        return features

    def _calculate_book_features(self, symbol: str, venue: str) -> Dict[str, float]:

        key = (symbol, venue)
        book_data = list(self.order_book_buffers[key])

        if not book_data:
            return {}

        features = {}
        current_book = book_data[-1]

        bid_size = current_book.get('bid_size', 0)
        ask_size = current_book.get('ask_size', 0)

        if bid_size + ask_size > 0:
            features['book_imbalance'] = (bid_size - ask_size) / (bid_size + ask_size)

        bid_price = current_book.get('bid_price', 0)
        ask_price = current_book.get('ask_price', 0)

        if bid_price > 0 and ask_price > 0 and bid_size + ask_size > 0:
            weighted_mid = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
            features['weighted_mid_price'] = weighted_mid

        if len(book_data) >= 5:
            recent_imbalances = []
            for book in book_data[-5:]:
                b_size = book.get('bid_size', 0)
                a_size = book.get('ask_size', 0)
                if b_size + a_size > 0:
                    recent_imbalances.append((b_size - a_size) / (b_size + a_size))

            if len(recent_imbalances) >= 2:
                features['book_pressure'] = np.mean(recent_imbalances)

        if len(book_data) >= 10:
            buy_flow = 0
            sell_flow = 0

            for i in range(1, min(10, len(book_data))):
                prev_book = book_data[-(i+1)]
                curr_book = book_data[-i]

                bid_change = curr_book.get('bid_size', 0) - prev_book.get('bid_size', 0)
                ask_change = curr_book.get('ask_size', 0) - prev_book.get('ask_size', 0)

                buy_flow += max(0, bid_change)
                sell_flow += max(0, ask_change)

            total_flow = buy_flow + sell_flow
            if total_flow > 0:
                features['order_flow_imbalance'] = (buy_flow - sell_flow) / total_flow

        total_liquidity = bid_size + ask_size
        if total_liquidity > 0:
            features['liquidity_ratio'] = min(bid_size, ask_size) / total_liquidity

        return features

    def _calculate_volume_features(self, symbol: str, venue: str) -> Dict[str, float]:

        key = (symbol, venue)
        ticks = list(self.tick_buffers[key])

        if len(ticks) < 2:
            return {}

        features = {}
        current_tick = ticks[-1]

        volume = current_tick.get('volume', 0)
        features['volume'] = volume

        if len(ticks) >= 20:
            volumes = [tick.get('volume', 0) for tick in ticks[-20:]]
            volume_ma = np.mean(volumes)
            if volume_ma > 0:
                features['volume_ma_ratio'] = volume / volume_ma

        if len(ticks) >= 50:
            volumes = [tick.get('volume', 0) for tick in ticks[-50:]]
            if volumes and np.mean(volumes) > 0:
                features['volume_volatility'] = np.std(volumes) / np.mean(volumes)

        return features

    def _calculate_momentum_features(self, symbol: str, venue: str) -> Dict[str, float]:

        key = (symbol, venue)
        ticks = list(self.tick_buffers[key])

        if len(ticks) < 10:
            return {}

        features = {}
        mid_prices = np.array([tick.get('mid_price', 0) for tick in ticks])
        mid_prices = mid_prices[mid_prices > 0]

        if len(mid_prices) < 10:
            return features

        for window in [1, 5, 10]:
            if len(mid_prices) > window:
                momentum = (mid_prices[-1] - mid_prices[-(window+1)]) / mid_prices[-(window+1)]
                features[f'momentum_{window}'] = momentum

        if len(mid_prices) >= 14:
            returns = np.diff(mid_prices[-15:])
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)

            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                features['rsi'] = rsi

        if len(mid_prices) >= 26:
            ema_12 = self._calculate_ema(mid_prices[-12:], 12)
            ema_26 = self._calculate_ema(mid_prices[-26:], 26)
            features['macd'] = ema_12 - ema_26

        return features

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:

        if len(prices) == 0:
            return 0.0

        alpha = 2.0 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def _calculate_network_features(self, symbol: str, venue: str) -> Dict[str, float]:

        key = (symbol, venue)
        network_data = list(self.network_buffers[key])

        if not network_data:
            return {}

        features = {}
        current_network = network_data[-1]

        features['network_latency'] = current_network.get('latency_us', 0)
        features['packet_loss_rate'] = current_network.get('packet_loss_rate', 0)
        features['jitter'] = current_network.get('jitter_us', 0)
        features['network_congestion'] = current_network.get('congestion_score', 0)

        if len(network_data) >= 20:
            latencies = [data.get('latency_us', 0) for data in network_data[-20:]]
            current_latency = current_network.get('latency_us', 0)

            if latencies:
                percentile = (sum(1 for lat in latencies if lat <= current_latency) / len(latencies)) * 100
                features['latency_percentile'] = percentile

        return features

    def _calculate_microstructure_features(self, symbol: str, venue: str) -> Dict[str, float]:

        key = (symbol, venue)
        ticks = list(self.tick_buffers[key])

        if len(ticks) < 3:
            return {}

        features = {}

        if len(ticks) >= 2:
            current_price = ticks[-1].get('mid_price', 0)
            prev_price = ticks[-2].get('mid_price', 0)

            if current_price > prev_price:
                features['tick_direction'] = 1.0
            elif current_price < prev_price:
                features['tick_direction'] = -1.0
            else:
                features['tick_direction'] = 0.0

        current_tick = ticks[-1]
        mid_price = current_tick.get('mid_price', 0)
        last_price = current_tick.get('last_price', 0)

        if mid_price > 0 and last_price > 0:
            effective_spread = 2 * abs(last_price - mid_price)
            features['effective_spread'] = effective_spread

        return features

    def _calculate_cross_venue_features(self, symbol: str, venue: str) -> Dict[str, float]:

        features = {}

        venue_data = {}
        for v in self.venues:
            key = (symbol, v)
            if self.tick_buffers[key]:
                venue_data[v] = list(self.tick_buffers[key])[-1]

        if len(venue_data) < 2:
            return features

        current_venue_data = venue_data.get(venue, {})
        current_mid = current_venue_data.get('mid_price', 0)

        spreads = []
        for v, data in venue_data.items():
            bid = data.get('bid_price', 0)
            ask = data.get('ask_price', 0)
            if bid > 0 and ask > 0:
                spreads.append((v, ask - bid))

        if spreads:
            spreads.sort(key=lambda x: x[1])
            for i, (v, spread) in enumerate(spreads):
                if v == venue:
                    features['venue_spread_rank'] = i / (len(spreads) - 1)
                    break

        venue_prices = []
        for v, data in venue_data.items():
            mid_price = data.get('mid_price', 0)
            if mid_price > 0:
                venue_prices.append(mid_price)

        if len(venue_prices) >= 2:
            price_std = np.std(venue_prices)
            features['cross_venue_momentum'] = price_std

        if current_mid > 0:
            best_bid = max((data.get('bid_price', 0) for data in venue_data.values()
                           if data.get('bid_price', 0) > 0), default=0)
            best_ask = min((data.get('ask_price', 0) for data in venue_data.values()
                           if data.get('ask_price', 0) > 0), default=float('inf'))

            if best_ask < float('inf') and best_bid > 0:
                current_ask = current_venue_data.get('ask_price', 0)
                current_bid = current_venue_data.get('bid_price', 0)

                buy_signal = (current_ask - best_ask) / current_mid if current_ask > 0 else 0
                sell_signal = (best_bid - current_bid) / current_mid if current_bid > 0 else 0

                features['arbitrage_signal'] = max(buy_signal, sell_signal)

        return features

    def extract_features(self, symbol: str, venue: str, timestamp: float,
                        target_latency: Optional[float] = None) -> FeatureVector:

        start_time = time.time()

        feature_vector = FeatureVector(
            timestamp=timestamp,
            symbol=symbol,
            venue=venue,
            target_latency=target_latency
        )

        feature_calculators = [
            ('temporal', self._calculate_temporal_features, (timestamp,)),
            ('price', self._calculate_price_features, (symbol, venue)),
            ('volatility', self._calculate_volatility_features, (symbol, venue)),
            ('spread', self._calculate_spread_features, (symbol, venue)),
            ('book', self._calculate_book_features, (symbol, venue)),
            ('volume', self._calculate_volume_features, (symbol, venue)),
            ('momentum', self._calculate_momentum_features, (symbol, venue)),
            ('network', self._calculate_network_features, (symbol, venue)),
            ('microstructure', self._calculate_microstructure_features, (symbol, venue)),
            ('cross_venue', self._calculate_cross_venue_features, (symbol, venue))
        ]

        for category_name, calculator_func, args in feature_calculators:
            try:
                category_features = calculator_func(*args)
                feature_vector.features.update(category_features)
            except Exception as e:
                logger.warning(f"Error calculating {category_name} features: {e}")

        self._update_feature_statistics(feature_vector.features)

        normalized_features = self._normalize_features(feature_vector.features)
        feature_vector.features = normalized_features

        extraction_time = time.time() - start_time
        self.extraction_times.append(extraction_time)
        self.extraction_count += 1

        feature_vector.metadata = {
            'extraction_time_ms': extraction_time * 1000,
            'feature_count': len(feature_vector.features),
            'data_completeness': self._calculate_data_completeness(symbol, venue)
        }

        return feature_vector

    def _update_feature_statistics(self, features: Dict[str, float]):

        for feature_name, value in features.items():
            if np.isfinite(value):
                history = self.feature_history[feature_name]
                history.append(value)

                if len(history) >= 10:
                    values = np.array(list(history))
                    self.feature_stats[feature_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }

    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:

        normalized = {}

        for feature_name, value in features.items():
            if not np.isfinite(value):
                normalized[feature_name] = 0.0
                continue

            config = self.feature_configs.get(feature_name, {})
            if not config.get('normalize', True):
                normalized[feature_name] = value
                continue

            stats = self.feature_stats.get(feature_name, {})
            if 'mean' in stats and 'std' in stats and stats['std'] > 0:
                normalized_value = (value - stats['mean']) / stats['std']
                normalized[feature_name] = np.clip(normalized_value, -3.0, 3.0)
            else:
                normalized[feature_name] = value

        return normalized

    def _calculate_data_completeness(self, symbol: str, venue: str) -> float:

        key = (symbol, venue)

    def _calculate_data_completeness(self, symbol: str, venue: str) -> float:

        key = (symbol, venue)

        tick_completeness = len(self.tick_buffers[key]) / max(self.lookback_windows.values())
        book_completeness = len(self.order_book_buffers[key]) / max(self.lookback_windows.values())
        network_completeness = len(self.network_buffers[key]) / max(self.lookback_windows.values())

        return min(1.0, (tick_completeness + book_completeness + network_completeness) / 3)

    def get_feature_importance(self, window_size: int = 1000) -> Dict[str, Dict[str, float]]:

        importance_metrics = {}

        for feature_name in self.feature_configs.keys():
            if feature_name in self.feature_history:
                history = list(self.feature_history[feature_name])[-window_size:]

                if len(history) >= 10:
                    values = np.array(history)

                    importance_metrics[feature_name] = {
                        'variance': np.var(values),
                        'range': np.max(values) - np.min(values),
                        'stability': 1.0 / (1.0 + np.std(values)),
                        'information_content': self._calculate_information_content(values)
                    }

        return importance_metrics

    def _calculate_information_content(self, values: np.ndarray) -> float:

        try:
            if len(values) < 5:
                return 0.0

            bins = min(10, len(values) // 5)
            hist, _ = np.histogram(values, bins=bins)

            probabilities = hist / np.sum(hist)
            probabilities = probabilities[probabilities > 0]

            return entropy(probabilities)
        except:
            return 0.0

    def batch_extract_features(self, requests: List[Tuple[str, str, float, Optional[float]]]) -> List[FeatureVector]:

        feature_vectors = []

        for symbol, venue, timestamp, target_latency in requests:
            try:
                fv = self.extract_features(symbol, venue, timestamp, target_latency)
                feature_vectors.append(fv)
            except Exception as e:
                logger.error(f"Failed to extract features for {symbol}@{venue}: {e}")

        return feature_vectors

    def export_features_dataframe(self, feature_vectors: List[FeatureVector]) -> pd.DataFrame:

        if not feature_vectors:
            return pd.DataFrame()

        data = []
        for fv in feature_vectors:
            row = {
                'timestamp': fv.timestamp,
                'symbol': fv.symbol,
                'venue': fv.venue,
                'target_latency': fv.target_latency
            }
            row.update(fv.features)
            row.update({f'meta_{k}': v for k, v in fv.metadata.items()})
            data.append(row)

        df = pd.DataFrame(data)

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        return df

    def get_feature_correlation_matrix(self, feature_vectors: List[FeatureVector]) -> pd.DataFrame:

        df = self.export_features_dataframe(feature_vectors)

        feature_columns = [col for col in df.columns
                          if col not in ['timestamp', 'symbol', 'venue', 'target_latency']
                          and not col.startswith('meta_')]

        if feature_columns:
            return df[feature_columns].corr()
        else:
            return pd.DataFrame()

    def detect_feature_anomalies(self, feature_vector: FeatureVector) -> List[Dict[str, Any]]:

        anomalies = []

        for feature_name, value in feature_vector.features.items():
            if not np.isfinite(value):
                anomalies.append({
                    'type': 'invalid_value',
                    'feature': feature_name,
                    'value': value,
                    'severity': 'high'
                })
                continue

            stats = self.feature_stats.get(feature_name, {})
            if 'mean' in stats and 'std' in stats and stats['std'] > 0:
                z_score = abs((value - stats['mean']) / stats['std'])

                if z_score > 5.0:
                    anomalies.append({
                        'type': 'extreme_value',
                        'feature': feature_name,
                        'value': value,
                        'z_score': z_score,
                        'severity': 'high'
                    })
                elif z_score > 3.0:
                    anomalies.append({
                        'type': 'outlier',
                        'feature': feature_name,
                        'value': value,
                        'z_score': z_score,
                        'severity': 'medium'
                    })

        return anomalies

    def get_feature_summary(self) -> Dict[str, Any]:

        avg_extraction_time = np.mean(list(self.extraction_times)) if self.extraction_times else 0

        feature_coverage = {}
        for feature_name, config in self.feature_configs.items():
            category = config['category'].value
            if category not in feature_coverage:
                feature_coverage[category] = {'total': 0, 'computed': 0}

            feature_coverage[category]['total'] += 1
            if feature_name in self.feature_stats:
                feature_coverage[category]['computed'] += 1

        feature_quality = {}
        for feature_name, stats in self.feature_stats.items():
            if stats.get('count', 0) >= 100:
                feature_quality[feature_name] = {
                    'completeness': min(1.0, stats['count'] / 1000),
                    'stability': 1.0 / (1.0 + stats.get('std', 1.0)),
                    'dynamic_range': (stats.get('max', 0) - stats.get('min', 0))
                }

        return {
            'extraction_count': self.extraction_count,
            'avg_extraction_time_ms': avg_extraction_time * 1000,
            'feature_categories': len(set(config['category'].value for config in self.feature_configs.values())),
            'total_features_defined': len(self.feature_configs),
            'features_with_data': len(self.feature_stats),
            'feature_coverage_by_category': {
                category: f"{stats['computed']}/{stats['total']}"
                for category, stats in feature_coverage.items()
            },
            'feature_quality_metrics': feature_quality,
            'data_buffers_status': {
                'tick_buffers': len(self.tick_buffers),
                'book_buffers': len(self.order_book_buffers),
                'network_buffers': len(self.network_buffers)
            }
        }

    def save_feature_config(self, filename: str = None) -> str:

        if not filename:
            filename = f"feature_config_{int(time.time())}.json"

        config_data = {
            'feature_configs': {
                name: {
                    'category': config['category'].value,
                    'normalize': config.get('normalize', True)
                }
                for name, config in self.feature_configs.items()
            },
            'lookback_windows': self.lookback_windows,
            'feature_statistics': dict(self.feature_stats),
            'extraction_summary': self.get_feature_summary()
        }

        try:
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            logger.info(f"Feature configuration saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save feature config: {e}")

        return filename

    async def start_real_time_extraction(self, symbol: str, venue: str,
                                       extraction_interval: float = 1.0,
                                       callback_func: Optional[callable] = None):

        logger.info(f"Starting real-time feature extraction for {symbol}@{venue}")

        while True:
            try:
                current_time = time.time()

                feature_vector = self.extract_features(symbol, venue, current_time)

                if callback_func:
                    await callback_func(feature_vector)

                anomalies = self.detect_feature_anomalies(feature_vector)
                if anomalies:
                    logger.warning(f"Feature anomalies detected for {symbol}@{venue}: {len(anomalies)} anomalies")

                await asyncio.sleep(extraction_interval)

            except Exception as e:
                logger.error(f"Error in real-time feature extraction: {e}")
                await asyncio.sleep(extraction_interval)
async def test_feature_extractor():


    symbols = ['AAPL', 'MSFT']
    venues = ['NYSE', 'NASDAQ']

    extractor = FeatureExtractor(symbols, venues)

    print("Testing FeatureExtractor...")

    import random

    for i in range(100):
        timestamp = time.time() + i

        for symbol in symbols:
            for venue in venues:
                base_price = 150.0 if symbol == 'AAPL' else 300.0
                price_noise = random.uniform(-0.5, 0.5)

                tick_data = {
                    'timestamp': timestamp,
                    'mid_price': base_price + price_noise,
                    'bid_price': base_price + price_noise - 0.01,
                    'ask_price': base_price + price_noise + 0.01,
                    'volume': random.randint(100, 1000),
                    'last_price': base_price + price_noise
                }

                book_data = {
                    'timestamp': timestamp,
                    'bid_price': tick_data['bid_price'],
                    'ask_price': tick_data['ask_price'],
                    'bid_size': random.randint(500, 2000),
                    'ask_size': random.randint(500, 2000)
                }

                network_data = {
                    'timestamp': timestamp,
                    'latency_us': random.randint(800, 1200),
                    'jitter_us': random.randint(20, 80),
                    'packet_loss_rate': random.uniform(0, 0.002),
                    'congestion_score': random.uniform(0, 1)
                }

                extractor.add_tick_data(symbol, venue, tick_data)
                extractor.add_order_book_data(symbol, venue, book_data)
                extractor.add_network_data(symbol, venue, network_data)

    feature_vectors = []

    for symbol in symbols:
        for venue in venues:
            fv = extractor.extract_features(symbol, venue, time.time(), target_latency=950.0)
            feature_vectors.append(fv)

            print(f"\n{symbol}@{venue} Features ({len(fv.features)} total):")

            features_by_category = defaultdict(list)
            for feature_name, value in fv.features.items():
                config = extractor.feature_configs.get(feature_name, {})
                category = config.get('category', FeatureCategory.TEMPORAL).value
                features_by_category[category].append((feature_name, value))

            for category, features in features_by_category.items():
                print(f"  {category.upper()}: {len(features)} features")
                for name, value in features[:3]:
                    print(f"    {name}: {value:.4f}")

    df = extractor.export_features_dataframe(feature_vectors)
    print(f"\nExported DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")

    importance = extractor.get_feature_importance()
    print(f"\nFeature importance calculated for {len(importance)} features")

    summary = extractor.get_feature_summary()
    print(f"\n=== Feature Extraction Summary ===")
    print(f"Total extractions: {summary['extraction_count']}")
    print(f"Avg extraction time: {summary['avg_extraction_time_ms']:.2f}ms")
    print(f"Features with data: {summary['features_with_data']}/{summary['total_features_defined']}")

    config_file = extractor.save_feature_config()
    print(f"Configuration saved to: {config_file}")

    return extractor, feature_vectors
if __name__ == "__main__":
    asyncio.run(test_feature_extractor())