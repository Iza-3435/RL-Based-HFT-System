import time
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)
@dataclass
class NetworkRoute:

    source: str
    destination: str
    base_latency_us: float
    capacity_mbps: float
    utilization: float
    congestion_events: List[Dict]
@dataclass
class LatencyPrediction:

    predicted_latency_us: float
    confidence_interval: Tuple[float, float]
    prediction_confidence: float
    contributing_factors: Dict[str, float]
    route_quality_score: float
@dataclass
class CongestionEvent:

    event_id: str
    affected_routes: List[str]
    severity: float
    start_time: float
    duration_sec: float
    cause: str
class LatencySimulator:


    def __init__(self, venues: List[str]):
        self.venues = venues
        self.routes = {}
        self.congestion_events = deque(maxlen=100)
        self.historical_latencies = defaultdict(deque)

        self.market_hours_multiplier = 1.2
        self.lunch_hour_multiplier = 0.8

        self._initialize_network_topology()

        self._last_congestion_check = time.time()

        logger.info(f"Enhanced Latency Simulator initialized for {len(venues)} venues")
        logger.info(f"   • Network routes: {len(self.routes)}")
        logger.info(f"   • Congestion modeling: ✅ Active")
        logger.info(f"   • Time-of-day effects: ✅ Active")

    def _initialize_network_topology(self):

        base_latencies = {
            ('NYSE', 'NASDAQ'): 250,
            ('NYSE', 'CBOE'): 850,
            ('NYSE', 'IEX'): 180,
            ('NYSE', 'ARCA'): 220,
            ('NASDAQ', 'CBOE'): 820,
            ('NASDAQ', 'IEX'): 150,
            ('NASDAQ', 'ARCA'): 200,
            ('CBOE', 'IEX'): 870,
            ('CBOE', 'ARCA'): 880,
            ('IEX', 'ARCA'): 120,
        }

        for venues_pair, base_latency in base_latencies.items():
            venue1, venue2 = venues_pair

            self.routes[f"{venue1}_to_{venue2}"] = NetworkRoute(
                source=venue1,
                destination=venue2,
                base_latency_us=base_latency,
                capacity_mbps=np.random.uniform(1000, 10000),
                utilization=np.random.uniform(0.1, 0.7),
                congestion_events=[]
            )

            self.routes[f"{venue2}_to_{venue1}"] = NetworkRoute(
                source=venue2,
                destination=venue1,
                base_latency_us=base_latency + np.random.uniform(-20, 20),
                capacity_mbps=np.random.uniform(1000, 10000),
                utilization=np.random.uniform(0.1, 0.7),
                congestion_events=[]
            )

    def _get_time_of_day_multiplier(self) -> float:

        now = datetime.now()
        hour = now.hour

        if 9 <= hour <= 16:
            if 12 <= hour <= 13:
                return self.lunch_hour_multiplier
            else:
                return self.market_hours_multiplier
        else:
            return 1.0

    def _simulate_congestion(self):

        current_time = time.time()

        if current_time - self._last_congestion_check < 30:
            return

        self._last_congestion_check = current_time

        if np.random.random() < 0.05:
            affected_routes = random.sample(list(self.routes.keys()),
                                          k=np.random.randint(1, 4))

            event = CongestionEvent(
                event_id=f"CONG_{int(current_time)}",
                affected_routes=affected_routes,
                severity=np.random.uniform(1.2, 2.5),
                start_time=current_time,
                duration_sec=np.random.uniform(30, 300),
                cause=random.choice(['Market volatility', 'Network maintenance',
                                   'High trading volume', 'Infrastructure upgrade'])
            )

            self.congestion_events.append(event)
            logger.debug(f"Congestion event {event.event_id}: {event.cause} affecting {len(affected_routes)} routes")

    def _get_active_congestion_multiplier(self, route_name: str) -> float:

        current_time = time.time()
        total_multiplier = 1.0

        for event in self.congestion_events:
            if (current_time >= event.start_time and
                current_time <= event.start_time + event.duration_sec and
                route_name in event.affected_routes):
                total_multiplier *= event.severity

        return total_multiplier

    def predict_latency(self, source_venue: str, dest_venue: str,
                       market_conditions: Optional[Dict] = None) -> LatencyPrediction:

        self._simulate_congestion()

        route_name = f"{source_venue}_to_{dest_venue}"

        if route_name not in self.routes:
            base_latency = 1000.0 + np.random.uniform(-200, 500)
        else:
            route = self.routes[route_name]
            base_latency = route.base_latency_us

        time_multiplier = self._get_time_of_day_multiplier()
        congestion_multiplier = self._get_active_congestion_multiplier(route_name)

        market_multiplier = 1.0
        if market_conditions:
            volatility = market_conditions.get('volatility', 0.01)
            volume = market_conditions.get('volume', 1.0)

            market_multiplier *= (1.0 + volatility * 2)

            market_multiplier *= (1.0 + np.log(volume) * 0.1)

        jitter_multiplier = np.random.lognormal(0, 0.1)

        predicted_latency = (base_latency *
                           time_multiplier *
                           congestion_multiplier *
                           market_multiplier *
                           jitter_multiplier)

        historical_accuracy = self._get_historical_accuracy(route_name)
        confidence = max(0.5, min(0.95, historical_accuracy))

        margin = predicted_latency * 0.15 / confidence
        conf_interval = (predicted_latency - margin, predicted_latency + margin)

        factors = {
            'base_latency': base_latency,
            'time_of_day_effect': (time_multiplier - 1.0) * 100,
            'congestion_effect': (congestion_multiplier - 1.0) * 100,
            'market_conditions_effect': (market_multiplier - 1.0) * 100,
            'network_jitter': (jitter_multiplier - 1.0) * 100
        }

        route_quality = (1.0 / congestion_multiplier) * confidence * 0.8

        self.historical_latencies[route_name].append({
            'predicted': predicted_latency,
            'timestamp': time.time(),
            'factors': factors
        })

        return LatencyPrediction(
            predicted_latency_us=predicted_latency,
            confidence_interval=conf_interval,
            prediction_confidence=confidence,
            contributing_factors=factors,
            route_quality_score=route_quality
        )

    def _get_historical_accuracy(self, route_name: str) -> float:

        history = self.historical_latencies.get(route_name, [])

        if len(history) < 5:
            return 0.75

        recent_predictions = history[-10:]

        avg_congestion = np.mean([p['factors'].get('congestion_effect', 0)
                                for p in recent_predictions])

        base_accuracy = 0.85
        congestion_penalty = min(0.2, avg_congestion / 100 * 0.3)

        return max(0.5, base_accuracy - congestion_penalty)

    def get_network_status(self) -> Dict:

        current_time = time.time()

        active_events = [e for e in self.congestion_events
                        if current_time <= e.start_time + e.duration_sec]

        route_health = {}
        for route_name, route in self.routes.items():
            congestion_mult = self._get_active_congestion_multiplier(route_name)

            if congestion_mult <= 1.1:
                status = "🟢 EXCELLENT"
            elif congestion_mult <= 1.3:
                status = "🟡 GOOD"
            elif congestion_mult <= 1.8:
                status = "🟠 DEGRADED"
            else:
                status = "🔴 CONGESTED"

            route_health[route_name] = {
                'status': status,
                'base_latency_us': route.base_latency_us,
                'current_multiplier': congestion_mult,
                'utilization': route.utilization
            }

        return {
            'timestamp': current_time,
            'active_congestion_events': len(active_events),
            'total_routes': len(self.routes),
            'route_health': route_health,
            'time_of_day_multiplier': self._get_time_of_day_multiplier(),
            'prediction_accuracy': {
                'overall': np.mean([self._get_historical_accuracy(r) for r in self.routes.keys()]),
                'by_route': {r: self._get_historical_accuracy(r) for r in self.routes.keys()}
            }
        }
class EnhancedOrderExecutionEngine:


    def __init__(self, latency_simulator: LatencySimulator):
        self.latency_simulator = latency_simulator
        self.execution_history = []

        logger.info("Enhanced Order Execution Engine initialized")
        logger.info("   • Latency-aware routing: ✅ Active")
        logger.info("   • Dynamic route optimization: ✅ Active")

    def execute_with_latency_optimization(self, symbol: str, venue: str,
                                        order_details: Dict) -> Dict:

        start_time = time.perf_counter()

        latency_pred = self.latency_simulator.predict_latency(
            source_venue='TRADING_SYSTEM',
            dest_venue=venue,
            market_conditions={
                'volatility': order_details.get('urgency', 0.5),
                'volume': order_details.get('quantity', 100) / 100
            }
        )

        actual_latency_us = latency_pred.predicted_latency_us + np.random.normal(0, 50)

        success_prob = max(0.85, latency_pred.route_quality_score)
        execution_success = np.random.random() < success_prob

        execution_result = {
            'success': execution_success,
            'symbol': symbol,
            'venue': venue,
            'predicted_latency_us': latency_pred.predicted_latency_us,
            'actual_latency_us': actual_latency_us,
            'latency_accuracy': 1.0 - abs(actual_latency_us - latency_pred.predicted_latency_us) / latency_pred.predicted_latency_us,
            'route_quality': latency_pred.route_quality_score,
            'contributing_factors': latency_pred.contributing_factors,
            'execution_time_us': (time.perf_counter() - start_time) * 1e6
        }

        self.execution_history.append(execution_result)
        return execution_result

    def get_enhanced_execution_stats(self) -> Dict:

        if not self.execution_history:
            return {"error": "No execution history available"}

        latency_accuracies = [e['latency_accuracy'] for e in self.execution_history]
        route_qualities = [e['route_quality'] for e in self.execution_history]
        actual_latencies = [e['actual_latency_us'] for e in self.execution_history]

        venue_performance = defaultdict(list)
        for exec_result in self.execution_history:
            venue_performance[exec_result['venue']].append(exec_result)

        venue_stats = {}
        for venue, executions in venue_performance.items():
            venue_stats[venue] = {
                'avg_latency_us': np.mean([e['actual_latency_us'] for e in executions]),
                'avg_accuracy': np.mean([e['latency_accuracy'] for e in executions]),
                'avg_route_quality': np.mean([e['route_quality'] for e in executions]),
                'execution_count': len(executions)
            }

        return {
            'execution_stats': {
                'total_executions': len(self.execution_history),
                'avg_latency_us': np.mean(actual_latencies),
                'avg_latency_accuracy': np.mean(latency_accuracies),
                'avg_route_quality': np.mean(route_qualities),
                'avg_latency_cost_bps': np.mean(actual_latencies) / 1000 * 0.5
            },
            'venue_performance': venue_stats,
            'latency_analysis': {
                'prediction_accuracy': {
                    'prediction_within_10pct': np.mean([1 for acc in latency_accuracies if acc > 0.9]) * 100,
                    'avg_prediction_error_us': np.mean([abs(1-acc) * 1000 for acc in latency_accuracies])
                },
                'congestion_analysis': {
                    'active_congestion_events': len(self.latency_simulator.congestion_events),
                    'avg_congestion_penalty_us': np.mean([
                        sum(e['contributing_factors'].get('congestion_effect', 0) for e in self.execution_history) / len(self.execution_history) * 10
                    ])
                }
            },
            'network_status': self.latency_simulator.get_network_status()
        }
class LatencyAnalytics:


    def __init__(self):
        self.analysis_cache = {}

    def analyze_latency_trends(self, execution_data: List[Dict]) -> Dict:

        if not execution_data:
            return {"error": "No execution data provided"}

        timestamps = [e.get('timestamp', time.time()) for e in execution_data]
        latencies = [e.get('actual_latency_us', 1000) for e in execution_data]

        if len(latencies) > 2:
            trend = np.polyfit(range(len(latencies)), latencies, 1)[0]
        else:
            trend = 0

        return {
            'trend_analysis': {
                'latency_trend_us_per_hour': trend * 3600,
                'trend_direction': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable'
            },
            'performance_summary': {
                'min_latency_us': np.min(latencies),
                'max_latency_us': np.max(latencies),
                'avg_latency_us': np.mean(latencies),
                'p95_latency_us': np.percentile(latencies, 95),
                'p99_latency_us': np.percentile(latencies, 99)
            }
        }
def create_enhanced_latency_system(venues: List[str]) -> Tuple[LatencySimulator, EnhancedOrderExecutionEngine]:

    latency_sim = LatencySimulator(venues)
    execution_engine = EnhancedOrderExecutionEngine(latency_sim)
    return latency_sim, execution_engine