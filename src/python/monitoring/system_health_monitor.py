import time
import numpy as np
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import threading
import asyncio
import json
from datetime import datetime, timedelta
import statistics
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
class SystemComponent(Enum):
    DATA_FEED = "data_feed"
    LATENCY_MONITOR = "latency_monitor"
    EXECUTION_ENGINE = "execution_engine"
    ML_MODELS = "ml_models"
    NETWORK = "network"
    VENUE_CONNECTIVITY = "venue_connectivity"
@dataclass
class HealthAlert:

    timestamp: float
    component: SystemComponent
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    suggested_action: str
    alert_id: str = field(default_factory=lambda: f"alert_{int(time.time()*1000)}")
@dataclass
class PriceAnomalyAlert:

    timestamp: float
    symbol: str
    venue: str
    anomaly_type: str
    current_price: float
    expected_price: float
    deviation_pct: float
    alert_level: AlertLevel
    other_venue_prices: Dict[str, float]
    suggested_action: str
class CrossVenuePriceValidator:


    def __init__(self, venues: List[str], tolerance_pct: float = 0.5):
        self.venues = venues
        self.tolerance_pct = tolerance_pct

        self.price_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
        self.last_update_time = defaultdict(lambda: defaultdict(float))

        self.anomaly_history = deque(maxlen=1000)
        self.venue_reliability = defaultdict(lambda: {'uptime': 1.0, 'accuracy': 1.0})

        self.alert_callbacks = []

        print(f"💰 Cross-Venue Price Validator initialized")
        print(f"📊 Monitoring {len(venues)} venues with {tolerance_pct}% tolerance")
    def update_price(self, symbol: str, venue: str, price: float, timestamp: float = None):

        if timestamp is None:
            timestamp = time.time()

        self.price_history[symbol][venue].append((timestamp, price))
        self.last_update_time[symbol][venue] = timestamp

        anomalies = self._detect_price_anomalies(symbol, venue, price, timestamp)

        for anomaly in anomalies:
            self.anomaly_history.append(anomaly)
            self._trigger_alerts(anomaly)
    def _detect_price_anomalies(self, symbol: str, venue: str, price: float,
                               timestamp: float) -> List[PriceAnomalyAlert]:

        anomalies = []

        other_venue_prices = self._get_concurrent_prices(symbol, venue, timestamp)

        if len(other_venue_prices) >= 1:
            anomaly = self._check_price_outliers(symbol, venue, price, timestamp, other_venue_prices)
            if anomaly:
                anomalies.append(anomaly)

        spike_anomaly = self._check_price_spikes(symbol, venue, price, timestamp)
        if spike_anomaly:
            anomalies.append(spike_anomaly)

        stale_anomaly = self._check_stale_feeds(symbol, venue, timestamp)
        if stale_anomaly:
            anomalies.append(stale_anomaly)

        disconnect_anomaly = self._check_feed_disconnections(symbol, timestamp)
        if disconnect_anomaly:
            anomalies.append(disconnect_anomaly)

        return anomalies
    def _get_concurrent_prices(self, symbol: str, exclude_venue: str,
                              timestamp: float, window_seconds: float = 5.0) -> Dict[str, float]:

        concurrent_prices = {}

        for venue in self.venues:
            if venue == exclude_venue:
                continue

            venue_history = self.price_history[symbol][venue]
            if not venue_history:
                continue

            for hist_timestamp, hist_price in reversed(venue_history):
                if abs(hist_timestamp - timestamp) <= window_seconds:
                    concurrent_prices[venue] = hist_price
                    break

        return concurrent_prices
    def _check_price_outliers(self, symbol: str, venue: str, price: float,
                             timestamp: float, other_prices: Dict[str, float]) -> Optional[PriceAnomalyAlert]:

        if not other_prices:
            return None

        other_price_values = list(other_prices.values())
        median_price = statistics.median(other_price_values)

        if median_price == 0:
            return None

        deviation_pct = abs(price - median_price) / median_price * 100

        if deviation_pct > self.tolerance_pct:
            alert_level = AlertLevel.WARNING if deviation_pct < self.tolerance_pct * 2 else AlertLevel.CRITICAL

            return PriceAnomalyAlert(
                timestamp=timestamp,
                symbol=symbol,
                venue=venue,
                anomaly_type='outlier',
                current_price=price,
                expected_price=median_price,
                deviation_pct=deviation_pct,
                alert_level=alert_level,
                other_venue_prices=other_prices,
                suggested_action=f"Investigate {venue} feed quality or routing logic"
            )

        return None
    def _check_price_spikes(self, symbol: str, venue: str, price: float,
                           timestamp: float) -> Optional[PriceAnomalyAlert]:

        venue_history = self.price_history[symbol][venue]

        if len(venue_history) < 10:
            return None

        recent_prices = [p for _, p in list(venue_history)[-10:]]

        if len(recent_prices) < 5:
            return None

        recent_avg = statistics.mean(recent_prices[:-1])
        recent_std = statistics.stdev(recent_prices[:-1]) if len(recent_prices) > 2 else 0

        if recent_std == 0 or recent_avg == 0:
            return None

        z_score = abs(price - recent_avg) / recent_std

        if z_score > 3.0:
            deviation_pct = abs(price - recent_avg) / recent_avg * 100

            return PriceAnomalyAlert(
                timestamp=timestamp,
                symbol=symbol,
                venue=venue,
                anomaly_type='spike',
                current_price=price,
                expected_price=recent_avg,
                deviation_pct=deviation_pct,
                alert_level=AlertLevel.WARNING if z_score < 5 else AlertLevel.CRITICAL,
                other_venue_prices={},
                suggested_action=f"Price spike detected on {venue} - verify trade or data quality"
            )

        return None
    def _check_stale_feeds(self, symbol: str, venue: str, timestamp: float) -> Optional[PriceAnomalyAlert]:

        last_update = self.last_update_time[symbol][venue]

        if last_update == 0:
            return None

        staleness_seconds = timestamp - last_update

        if staleness_seconds > 30:
            return PriceAnomalyAlert(
                timestamp=timestamp,
                symbol=symbol,
                venue=venue,
                anomaly_type='stale',
                current_price=0,
                expected_price=0,
                deviation_pct=0,
                alert_level=AlertLevel.WARNING if staleness_seconds < 60 else AlertLevel.CRITICAL,
                other_venue_prices={},
                suggested_action=f"Feed for {symbol} on {venue} is stale ({staleness_seconds:.0f}s)"
            )

        return None
    def _check_feed_disconnections(self, symbol: str, timestamp: float) -> Optional[PriceAnomalyAlert]:

        active_venues = 0
        disconnected_venues = []

        for venue in self.venues:
            last_update = self.last_update_time[symbol][venue]

            if last_update == 0 or (timestamp - last_update) > 60:
                disconnected_venues.append(venue)
            else:
                active_venues += 1

        if len(disconnected_venues) > len(self.venues) / 2:
            return PriceAnomalyAlert(
                timestamp=timestamp,
                symbol=symbol,
                venue="multiple",
                anomaly_type='disconnect',
                current_price=0,
                expected_price=0,
                deviation_pct=0,
                alert_level=AlertLevel.CRITICAL,
                other_venue_prices={},
                suggested_action=f"Multiple venue disconnection: {disconnected_venues}"
            )

        return None
    def _trigger_alerts(self, anomaly: PriceAnomalyAlert):

        for callback in self.alert_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    def add_alert_callback(self, callback: Callable[[PriceAnomalyAlert], None]):

        self.alert_callbacks.append(callback)
    def get_venue_reliability_stats(self) -> Dict[str, Dict]:

        current_time = time.time()
        stats = {}

        for venue in self.venues:
            recent_updates = 0
            total_symbols = 0

            for symbol in self.price_history.keys():
                total_symbols += 1
                last_update = self.last_update_time[symbol][venue]

                if last_update > 0 and (current_time - last_update) < 300:
                    recent_updates += 1

            uptime_pct = (recent_updates / total_symbols * 100) if total_symbols > 0 else 0

            venue_anomalies = [a for a in self.anomaly_history
                             if a.venue == venue and (current_time - a.timestamp) < 3600]

            accuracy_pct = max(0, 100 - len(venue_anomalies) * 5)

            stats[venue] = {
                'uptime_pct': uptime_pct,
                'accuracy_pct': accuracy_pct,
                'recent_anomalies': len(venue_anomalies),
                'reliability_score': (uptime_pct + accuracy_pct) / 2
            }

        return stats
class SystemHealthMonitor:


    def __init__(self, monitoring_interval_seconds: float = 5.0):
        self.monitoring_interval = monitoring_interval_seconds
        self.is_monitoring = False
        self.monitor_thread = None

        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alert_callbacks = []
        self.health_checkers = {}

        self.component_states = {comp: 'unknown' for comp in SystemComponent}
        self.last_health_check = {comp: 0 for comp in SystemComponent}

        self.performance_metrics = {
            'data_feed_latency': deque(maxlen=100),
            'execution_latency': deque(maxlen=100),
            'ml_prediction_time': deque(maxlen=100),
            'network_latency': deque(maxlen=100)
        }

        self._register_default_health_checkers()

        print("🏥 System Health Monitor initialized")
        print(f"⚡ Monitoring every {monitoring_interval_seconds}s")
    def _register_default_health_checkers(self):


        self.health_checkers[SystemComponent.DATA_FEED] = self._check_data_feed_health
        self.health_checkers[SystemComponent.LATENCY_MONITOR] = self._check_latency_monitor_health
        self.health_checkers[SystemComponent.EXECUTION_ENGINE] = self._check_execution_health
        self.health_checkers[SystemComponent.ML_MODELS] = self._check_ml_model_health
        self.health_checkers[SystemComponent.NETWORK] = self._check_network_health
        self.health_checkers[SystemComponent.VENUE_CONNECTIVITY] = self._check_venue_connectivity
    def start_monitoring(self):

        if self.is_monitoring:
            print("⚠️ Health monitoring already running")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("✅ System health monitoring started")
    def stop_monitoring(self):

        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("🛑 System health monitoring stopped")
    def _monitoring_loop(self):

        while self.is_monitoring:
            try:
                current_time = time.time()

                for component, checker in self.health_checkers.items():
                    try:
                        health_result = checker()
                        self._process_health_result(component, health_result, current_time)
                    except Exception as e:
                        logger.error(f"Health check error for {component}: {e}")

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)
    def _process_health_result(self, component: SystemComponent,
                              health_result: Dict[str, Any], timestamp: float):


        self.component_states[component] = health_result.get('status', 'unknown')
        self.last_health_check[component] = timestamp

        for metric_name, metric_value in health_result.get('metrics', {}).items():
            self.metrics_history[f"{component.value}_{metric_name}"].append((timestamp, metric_value))

        alerts = health_result.get('alerts', [])
        for alert in alerts:
            self._trigger_health_alert(alert)
    def _check_data_feed_health(self) -> Dict[str, Any]:

        current_time = time.time()

        recent_latencies = list(self.performance_metrics['data_feed_latency'])[-10:]
        avg_latency = statistics.mean(recent_latencies) if recent_latencies else 0

        alerts = []
        status = 'healthy'

        if avg_latency > 100:
            alerts.append(HealthAlert(
                timestamp=current_time,
                component=SystemComponent.DATA_FEED,
                level=AlertLevel.WARNING if avg_latency < 500 else AlertLevel.CRITICAL,
                message=f"Data feed latency high: {avg_latency:.1f}ms",
                metric_name="feed_latency_ms",
                current_value=avg_latency,
                expected_range=(0, 100),
                suggested_action="Check network connection and data source"
            ))
            status = 'degraded'

        return {
            'status': status,
            'metrics': {
                'avg_latency_ms': avg_latency,
                'feed_count': len(recent_latencies)
            },
            'alerts': alerts
        }
    def _check_latency_monitor_health(self) -> Dict[str, Any]:

        current_time = time.time()

        recent_measurements = list(self.performance_metrics['network_latency'])[-20:]
        avg_latency = statistics.mean(recent_measurements) if recent_measurements else 0

        alerts = []
        status = 'healthy'

        if avg_latency > 50000:
            alerts.append(HealthAlert(
                timestamp=current_time,
                component=SystemComponent.LATENCY_MONITOR,
                level=AlertLevel.WARNING if avg_latency < 100000 else AlertLevel.CRITICAL,
                message=f"Network latency high: {avg_latency/1000:.1f}ms",
                metric_name="network_latency_us",
                current_value=avg_latency,
                expected_range=(0, 50000),
                suggested_action="Check network connectivity to exchanges"
            ))
            status = 'degraded'

        return {
            'status': status,
            'metrics': {
                'avg_network_latency_us': avg_latency,
                'measurements_count': len(recent_measurements)
            },
            'alerts': alerts
        }
    def _check_execution_health(self) -> Dict[str, Any]:

        current_time = time.time()

        recent_exec_times = list(self.performance_metrics['execution_latency'])[-10:]
        avg_exec_time = statistics.mean(recent_exec_times) if recent_exec_times else 0

        alerts = []
        status = 'healthy'

        if avg_exec_time > 1000:
            alerts.append(HealthAlert(
                timestamp=current_time,
                component=SystemComponent.EXECUTION_ENGINE,
                level=AlertLevel.WARNING,
                message=f"Execution latency high: {avg_exec_time:.1f}μs",
                metric_name="execution_latency_us",
                current_value=avg_exec_time,
                expected_range=(0, 1000),
                suggested_action="Check execution engine performance"
            ))
            status = 'degraded'

        return {
            'status': status,
            'metrics': {
                'avg_execution_time_us': avg_exec_time,
                'execution_count': len(recent_exec_times)
            },
            'alerts': alerts
        }
    def _check_ml_model_health(self) -> Dict[str, Any]:

        current_time = time.time()

        recent_pred_times = list(self.performance_metrics['ml_prediction_time'])[-20:]
        avg_pred_time = statistics.mean(recent_pred_times) if recent_pred_times else 0

        alerts = []
        status = 'healthy'

        if avg_pred_time > 10:
            alerts.append(HealthAlert(
                timestamp=current_time,
                component=SystemComponent.ML_MODELS,
                level=AlertLevel.WARNING,
                message=f"ML prediction time high: {avg_pred_time:.1f}ms",
                metric_name="ml_prediction_time_ms",
                current_value=avg_pred_time,
                expected_range=(0, 10),
                suggested_action="Check ML model performance or GPU utilization"
            ))
            status = 'degraded'

        return {
            'status': status,
            'metrics': {
                'avg_prediction_time_ms': avg_pred_time,
                'prediction_count': len(recent_pred_times)
            },
            'alerts': alerts
        }
    def _check_network_health(self) -> Dict[str, Any]:

        return {
            'status': 'healthy',
            'metrics': {
                'connection_quality': 100.0
            },
            'alerts': []
        }
    def _check_venue_connectivity(self) -> Dict[str, Any]:

        return {
            'status': 'healthy',
            'metrics': {
                'connected_venues': 5
            },
            'alerts': []
        }
    def record_performance_metric(self, metric_type: str, value: float):

        if metric_type in self.performance_metrics:
            self.performance_metrics[metric_type].append(value)
    def _trigger_health_alert(self, alert: HealthAlert):

        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Health alert callback error: {e}")
    def add_health_alert_callback(self, callback: Callable[[HealthAlert], None]):

        self.alert_callbacks.append(callback)
    def get_system_health_summary(self) -> Dict[str, Any]:

        current_time = time.time()

        component_health = {}
        healthy_components = 0

        for component, status in self.component_states.items():
            last_check = self.last_health_check[component]
            check_age = current_time - last_check

            if check_age > 60:
                status = 'stale'

            component_health[component.value] = {
                'status': status,
                'last_check_age_seconds': check_age
            }

            if status == 'healthy':
                healthy_components += 1

        system_health_score = (healthy_components / len(SystemComponent)) * 100

        recent_alerts = []

        return {
            'timestamp': current_time,
            'overall_health_score': system_health_score,
            'components': component_health,
            'recent_alerts_count': len(recent_alerts),
            'monitoring_active': self.is_monitoring,
            'performance_summary': self._get_performance_summary()
        }
    def _get_performance_summary(self) -> Dict[str, float]:

        summary = {}

        for metric_name, values in self.performance_metrics.items():
            if values:
                recent_values = list(values)[-10:]
                summary[f"{metric_name}_avg"] = statistics.mean(recent_values)
                summary[f"{metric_name}_count"] = len(recent_values)
            else:
                summary[f"{metric_name}_avg"] = 0
                summary[f"{metric_name}_count"] = 0

        return summary
def create_monitoring_system(venues: List[str]) -> Tuple[CrossVenuePriceValidator, SystemHealthMonitor]:


    price_validator = CrossVenuePriceValidator(venues)
    health_monitor = SystemHealthMonitor()

    def print_price_alert(alert: PriceAnomalyAlert):
        level_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨", "emergency": "💥"}
        emoji = level_emoji.get(alert.alert_level.value, "⚠️")

        print(f"{emoji} PRICE ALERT: {alert.anomaly_type.upper()} - {alert.symbol} on {alert.venue}")
        print(f"    Current: ${alert.current_price:.4f} | Expected: ${alert.expected_price:.4f}")
        print(f"    Deviation: {alert.deviation_pct:.2f}% | Action: {alert.suggested_action}")

    def print_health_alert(alert: HealthAlert):
        level_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨", "emergency": "💥"}
        emoji = level_emoji.get(alert.level.value, "⚠️")

        print(f"{emoji} HEALTH ALERT: {alert.component.value.upper()}")
        print(f"    {alert.message}")
        print(f"    Current: {alert.current_value:.2f} | Expected: {alert.expected_range}")
        print(f"    Action: {alert.suggested_action}")

    price_validator.add_alert_callback(print_price_alert)
    health_monitor.add_health_alert_callback(print_health_alert)

    return price_validator, health_monitor
def integrate_monitoring_with_trading_system(trading_system, price_validator: CrossVenuePriceValidator,
                                           health_monitor: SystemHealthMonitor):


    if hasattr(trading_system, 'process_market_data'):
        original_process = trading_system.process_market_data

        def enhanced_process_market_data(symbol, venue, price, volume, timestamp=None):
            price_validator.update_price(symbol, venue, price, timestamp or time.time())

            health_monitor.record_performance_metric('data_feed_latency', 50)

            return original_process(symbol, venue, price, volume, timestamp)

        trading_system.process_market_data = enhanced_process_market_data

    health_monitor.start_monitoring()

    print("✅ Monitoring system integrated with trading system")
if __name__ == "__main__":
    venues = ['NYSE', 'NASDAQ', 'CBOE', 'IEX', 'ARCA']

    print("🧪 Testing System Health & Price Validation...")

    price_validator, health_monitor = create_monitoring_system(venues)

    health_monitor.start_monitoring()

    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    base_prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0}

    for i in range(20):
        for symbol in test_symbols:
            base_price = base_prices[symbol]

            for venue in venues:
                if venue != 'CBOE' or i < 10:
                    price = base_price + np.random.normal(0, 0.1)
                else:
                    price = base_price + np.random.normal(0, 0.1) + (2.0 if i == 15 else 0)

                price_validator.update_price(symbol, venue, price)

        time.sleep(0.1)

    health_summary = health_monitor.get_system_health_summary()
    venue_stats = price_validator.get_venue_reliability_stats()

    print(f"\n📊 System Health Score: {health_summary['overall_health_score']:.1f}%")
    print("📈 Venue Reliability:")
    for venue, stats in venue_stats.items():
        print(f"  {venue}: {stats['reliability_score']:.1f}% (Anomalies: {stats['recent_anomalies']})")

    health_monitor.stop_monitoring()
    print("✅ Monitoring system test complete!")