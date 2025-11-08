import asyncio
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
from enum import Enum
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.trading_simulator import Fill

logger = logging.getLogger(__name__)

class NetworkCondition(Enum):
    """Network condition states"""
    OPTIMAL = "optimal"
    NORMAL = "normal" 
    CONGESTED = "congested"
    DEGRADED = "degraded"
    CRITICAL = "critical"

@dataclass
class LatencyMeasurement:
    """Single latency measurement with metadata"""
    venue: str
    timestamp: float
    latency_us: int
    jitter_us: int
    packet_loss: bool
    condition: NetworkCondition
    route_id: str
    hop_count: int
    
@dataclass
class NetworkRoute:
    """Network route configuration"""
    route_id: str
    venues: List[str]
    base_latency_us: int
    hop_latencies: List[int]
    reliability: float
    bandwidth_mbps: int
    cost_factor: float

@dataclass
class CongestionEvent:
    """Network congestion event"""
    timestamp: float
    venue: str
    severity: float  # 0.0 to 1.0
    duration_seconds: float
    cause: str  # "volume_spike", "infrastructure", "market_event"

class NetworkLatencySimulator:
    """
    Advanced network latency simulation for HFT environments.
    
    Features:
    - Venue-specific latency modeling with realistic distributions
    - Dynamic congestion simulation based on market conditions
    - Packet loss and jitter modeling
    - Multiple network route simulation
    - Time-varying network conditions
    - Correlated latency events across venues
    """
    
    def __init__(self, venue_configs: Dict = None):
        # Default venue network configurations
        self.venue_configs = venue_configs or {
            'NYSE': {
                'base_latency_us': 850,
                'jitter_std': 45,
                'packet_loss_base': 0.001,
                'congestion_sensitivity': 1.5,
                'fiber_routes': 3,
                'microwave_routes': 2
            },
            'NASDAQ': {
                'base_latency_us': 920,
                'jitter_std': 50,
                'packet_loss_base': 0.0008,
                'congestion_sensitivity': 1.3,
                'fiber_routes': 4,
                'microwave_routes': 1
            },
            'CBOE': {
                'base_latency_us': 1100,
                'jitter_std': 65,
                'packet_loss_base': 0.0015,
                'congestion_sensitivity': 1.8,
                'fiber_routes': 2,
                'microwave_routes': 2
            },
            'IEX': {
                'base_latency_us': 750,
                'jitter_std': 35,
                'packet_loss_base': 0.0005,
                'congestion_sensitivity': 1.2,
                'fiber_routes': 2,
                'microwave_routes': 3
            },
            'ARCA': {
                'base_latency_us': 780,
                'jitter_std': 40,
                'packet_loss_base': 0.0007,
                'congestion_sensitivity': 1.4,
                'fiber_routes': 3,
                'microwave_routes': 2
            }
        }
        
        # Network state tracking
        self.current_conditions: Dict[str, NetworkCondition] = {}
        self.congestion_events: Deque[CongestionEvent] = deque(maxlen=1000)
        self.latency_history: Dict[str, Deque[LatencyMeasurement]] = defaultdict(lambda: deque(maxlen=1000))
        
        # Network routes between venues
        self.network_routes: Dict[str, NetworkRoute] = {}
        self._initialize_network_routes()
        
        # Dynamic state variables
        self.market_volatility = 0.02  # Updated by market data
        self.trading_volume_factor = 1.0  # Updated by market activity
        self.global_congestion = 0.0  # 0.0 to 1.0
        
        # Performance tracking
        self.measurements_count = 0
        self.packet_loss_events = 0
        
        logger.info(f"NetworkLatencySimulator initialized for {len(self.venue_configs)} venues")
    
    def _initialize_network_routes(self):
        """Initialize network routes between venues"""
        venues = list(self.venue_configs.keys())
        
        # Create routes between all venue pairs
        for i, venue_a in enumerate(venues):
            for j, venue_b in enumerate(venues):
                if i < j:  # Avoid duplicate routes
                    route_id = f"{venue_a}_{venue_b}"
                    
                    # Calculate combined latency
                    latency_a = self.venue_configs[venue_a]['base_latency_us']
                    latency_b = self.venue_configs[venue_b]['base_latency_us']
                    
                    # Inter-venue latency (additional network hops)
                    inter_venue_latency = random.randint(50, 200)
                    
                    route = NetworkRoute(
                        route_id=route_id,
                        venues=[venue_a, venue_b],
                        base_latency_us=latency_a + latency_b + inter_venue_latency,
                        hop_latencies=[latency_a, inter_venue_latency, latency_b],
                        reliability=0.99 - random.uniform(0, 0.02),  # 97-99% reliability
                        bandwidth_mbps=random.choice([1000, 10000, 40000, 100000]),  # Various bandwidth tiers
                        cost_factor=random.uniform(0.8, 1.5)
                    )
                    
                    self.network_routes[route_id] = route
        
        logger.info(f"Initialized {len(self.network_routes)} network routes")
    
    def update_market_conditions(self, volatility: float, volume_factor: float):
        """Update network simulation based on current market conditions"""
        self.market_volatility = volatility
        self.trading_volume_factor = volume_factor
        
        # Higher volatility and volume increase network stress
        self.global_congestion = min(0.8, (volatility * 10 + volume_factor * 0.3))
        
        # Update venue conditions based on global state
        for venue in self.venue_configs:
            sensitivity = self.venue_configs[venue]['congestion_sensitivity']
            
            if self.global_congestion > 0.6:
                self.current_conditions[venue] = NetworkCondition.CONGESTED
            elif self.global_congestion > 0.4:
                self.current_conditions[venue] = NetworkCondition.NORMAL
            elif self.global_congestion < 0.1:
                self.current_conditions[venue] = NetworkCondition.OPTIMAL
            else:
                self.current_conditions[venue] = NetworkCondition.NORMAL
    
    def _get_time_of_day_factors(self, timestamp: float) -> Dict[str, float]:
        """Calculate time-of-day network effects"""
        current_time = datetime.fromtimestamp(timestamp)
        hour = current_time.hour
        minute = current_time.minute
        
        # Market open: 9:30 AM - high network load
        if 9 <= hour < 11:
            latency_multiplier = 1.5
            congestion_probability = 0.3
            packet_loss_multiplier = 2.0
        # Lunch period: 12:00-1:00 PM - reduced load
        elif 12 <= hour < 13:
            latency_multiplier = 0.8
            congestion_probability = 0.05
            packet_loss_multiplier = 0.5
        # Market close: 3:30-4:00 PM - extreme load
        elif hour >= 15 and (hour > 15 or minute >= 30):
            latency_multiplier = 2.0
            congestion_probability = 0.5
            packet_loss_multiplier = 3.0
        # After hours: 4:00-6:00 PM - maintenance window
        elif 16 <= hour < 18:
            latency_multiplier = 1.2
            congestion_probability = 0.2
            packet_loss_multiplier = 1.5
        else:
            latency_multiplier = 1.0
            congestion_probability = 0.1
            packet_loss_multiplier = 1.0
        
        return {
            'latency_multiplier': latency_multiplier,
            'congestion_probability': congestion_probability,
            'packet_loss_multiplier': packet_loss_multiplier
        }
    
    def _simulate_congestion_event(self, venue: str, timestamp: float):
        """Simulate a network congestion event"""
        # Determine congestion severity based on current conditions
        base_severity = self.global_congestion
        random_factor = random.uniform(0.2, 0.8)
        severity = min(1.0, base_severity + random_factor)
        
        # Congestion duration varies by severity
        if severity > 0.8:
            duration = random.uniform(30, 300)  # 30s to 5min for severe congestion
            cause = "infrastructure"
        elif severity > 0.6:
            duration = random.uniform(10, 60)   # 10s to 1min for moderate congestion
            cause = "volume_spike"
        else:
            duration = random.uniform(2, 15)    # 2s to 15s for light congestion
            cause = "market_event"
        
        event = CongestionEvent(
            timestamp=timestamp,
            venue=venue,
            severity=severity,
            duration_seconds=duration,
            cause=cause
        )
        
        self.congestion_events.append(event)
        
        logger.info(f"Congestion event at {venue}: severity={severity:.2f}, "
                   f"duration={duration:.1f}s, cause={cause}")
        
        return event
    
    def _calculate_dynamic_latency(self, venue: str, timestamp: float) -> Tuple[int, int, bool]:
        """Calculate dynamic latency with all effects included"""
        config = self.venue_configs[venue]
        time_factors = self._get_time_of_day_factors(timestamp)
        
        # Base latency with time-of-day effects
        base_latency = config['base_latency_us'] * time_factors['latency_multiplier']
        
        # Add market condition effects
        condition_multiplier = {
            NetworkCondition.OPTIMAL: 0.8,
            NetworkCondition.NORMAL: 1.0,
            NetworkCondition.CONGESTED: 1.8,
            NetworkCondition.DEGRADED: 2.5,
            NetworkCondition.CRITICAL: 4.0
        }
        
        current_condition = self.current_conditions.get(venue, NetworkCondition.NORMAL)
        latency_with_conditions = base_latency * condition_multiplier[current_condition]
        
        # Check for active congestion events
        current_time = timestamp
        active_congestion = 0.0
        
        for event in self.congestion_events:
            if (event.venue == venue and 
                current_time >= event.timestamp and 
                current_time <= event.timestamp + event.duration_seconds):
                active_congestion = max(active_congestion, event.severity)
        
        # Apply congestion multiplier
        if active_congestion > 0:
            congestion_multiplier = 1.0 + (active_congestion * 2.0)  # Up to 3x latency
            latency_with_conditions *= congestion_multiplier
        
        # Generate jitter using normal distribution
        jitter_std = config['jitter_std'] * (1 + active_congestion)
        jitter = max(0, int(np.random.normal(0, jitter_std)))
        
        final_latency = int(latency_with_conditions + jitter)
        
        # Calculate packet loss probability
        base_loss_rate = config['packet_loss_base']
        loss_multiplier = time_factors['packet_loss_multiplier']
        condition_loss_multiplier = {
            NetworkCondition.OPTIMAL: 0.5,
            NetworkCondition.NORMAL: 1.0,
            NetworkCondition.CONGESTED: 3.0,
            NetworkCondition.DEGRADED: 8.0,
            NetworkCondition.CRITICAL: 20.0
        }
        
        final_loss_rate = (base_loss_rate * loss_multiplier * 
                          condition_loss_multiplier[current_condition] *
                          (1 + active_congestion * 5))
        
        packet_lost = random.random() < final_loss_rate
        
        if packet_lost:
            self.packet_loss_events += 1
        
        return final_latency, jitter, packet_lost
    
    def measure_latency(self, venue: str, timestamp: float = None) -> LatencyMeasurement:
        """Measure latency to a specific venue"""
        if timestamp is None:
            timestamp = time.time()
        
        # Randomly trigger congestion events
        time_factors = self._get_time_of_day_factors(timestamp)
        if random.random() < time_factors['congestion_probability']:
            self._simulate_congestion_event(venue, timestamp)
        
        # Calculate latency with all dynamic effects
        latency_us, jitter_us, packet_lost = self._calculate_dynamic_latency(venue, timestamp)
        
        # Determine route (simplified - use primary route)
        route_id = f"primary_{venue}"
        hop_count = 3  # Typical hop count
        
        measurement = LatencyMeasurement(
            venue=venue,
            timestamp=timestamp,
            latency_us=latency_us,
            jitter_us=jitter_us,
            packet_loss=packet_lost,
            condition=self.current_conditions.get(venue, NetworkCondition.NORMAL),
            route_id=route_id,
            hop_count=hop_count
        )
        
        # Store measurement
        self.latency_history[venue].append(measurement)
        self.measurements_count += 1
        
        return measurement
    
    def measure_route_latency(self, route_id: str, timestamp: float = None) -> Optional[LatencyMeasurement]:
        """Measure latency across a specific network route"""
        if route_id not in self.network_routes:
            logger.error(f"Unknown route: {route_id}")
            return None
        
        if timestamp is None:
            timestamp = time.time()
        
        route = self.network_routes[route_id]
        
        # Calculate total route latency as sum of hop latencies
        total_latency = 0
        total_jitter = 0
        any_packet_loss = False
        
        for i, hop_latency in enumerate(route.hop_latencies):
            # Add random variation to each hop
            hop_variation = random.randint(-50, 100)
            hop_latency_actual = max(50, hop_latency + hop_variation)
            
            # Add congestion effects
            congestion_factor = 1.0 + (self.global_congestion * random.uniform(0.5, 1.5))
            hop_latency_final = int(hop_latency_actual * congestion_factor)
            
            total_latency += hop_latency_final
            total_jitter += random.randint(0, 30)  # Per-hop jitter
            
            # Check for packet loss at any hop
            if random.random() < (1 - route.reliability) * (1 + self.global_congestion):
                any_packet_loss = True
        
        measurement = LatencyMeasurement(
            venue=f"route_{route_id}",
            timestamp=timestamp,
            latency_us=total_latency,
            jitter_us=total_jitter,
            packet_loss=any_packet_loss,
            condition=NetworkCondition.NORMAL,  # Routes don't have conditions
            route_id=route_id,
            hop_count=len(route.hop_latencies)
        )
        
        return measurement
    
    def get_venue_latency_percentiles(self, venue: str, window_seconds: int = 60) -> Dict[str, float]:
        """Calculate latency percentiles for a venue over recent history"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_measurements = [
            m.latency_us for m in self.latency_history[venue]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_measurements:
            return {}
        
        percentiles = {}
        for p in [50, 75, 90, 95, 99]:
            percentiles[f'p{p}'] = np.percentile(recent_measurements, p)
        
        percentiles['mean'] = np.mean(recent_measurements)
        percentiles['std'] = np.std(recent_measurements)
        percentiles['count'] = len(recent_measurements)
        
        return percentiles
    
    def detect_network_anomalies(self, venue: str = None) -> List[Dict]:
        """Detect network anomalies and performance issues"""
        anomalies = []
        current_time = time.time()
        
        venues_to_check = [venue] if venue else list(self.venue_configs.keys())
        
        for v in venues_to_check:
            recent_measurements = [
                m for m in self.latency_history[v]
                if current_time - m.timestamp < 300  # Last 5 minutes
            ]
            
            if len(recent_measurements) < 10:
                continue
            
            latencies = [m.latency_us for m in recent_measurements]
            packet_losses = sum(1 for m in recent_measurements if m.packet_loss)
            
            # Detect latency spikes
            mean_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            baseline_latency = self.venue_configs[v]['base_latency_us']
            
            if mean_latency > baseline_latency * 2:
                anomalies.append({
                    'type': 'high_latency',
                    'venue': v,
                    'severity': mean_latency / baseline_latency,
                    'description': f'Latency {mean_latency:.0f}μs vs baseline {baseline_latency}μs'
                })
            
            # Detect excessive jitter
            if std_latency > mean_latency * 0.3:
                anomalies.append({
                    'type': 'high_jitter',
                    'venue': v,
                    'severity': std_latency / mean_latency,
                    'description': f'High jitter: std={std_latency:.0f}μs'
                })
            
            # Detect packet loss issues
            loss_rate = packet_losses / len(recent_measurements)
            if loss_rate > 0.01:  # > 1% loss
                anomalies.append({
                    'type': 'packet_loss',
                    'venue': v,
                    'severity': loss_rate * 100,
                    'description': f'Packet loss rate: {loss_rate:.2%}'
                })
        
        return anomalies
    
    def get_optimal_routes(self, source_venue: str, target_venue: str) -> List[NetworkRoute]:
        """Get optimal network routes between venues sorted by performance"""
        possible_routes = []
        
        # Direct route
        direct_route_id = f"{source_venue}_{target_venue}"
        reverse_route_id = f"{target_venue}_{source_venue}"
        
        if direct_route_id in self.network_routes:
            possible_routes.append(self.network_routes[direct_route_id])
        elif reverse_route_id in self.network_routes:
            possible_routes.append(self.network_routes[reverse_route_id])
        
        # Calculate route scores (lower is better)
        def route_score(route):
            latency_score = route.base_latency_us
            reliability_penalty = (1 - route.reliability) * 10000  # Heavy penalty for unreliability
            congestion_penalty = self.global_congestion * route.base_latency_us * 0.5
            
            return latency_score + reliability_penalty + congestion_penalty
        
        # Sort by performance score
        possible_routes.sort(key=route_score)
        
        return possible_routes
    
    def get_network_performance_summary(self) -> Dict:
        """Get comprehensive network performance summary"""
        current_time = time.time()
        
        summary = {
            'timestamp': current_time,
            'total_measurements': self.measurements_count,
            'packet_loss_events': self.packet_loss_events,
            'global_congestion': self.global_congestion,
            'market_volatility': self.market_volatility,
            'trading_volume_factor': self.trading_volume_factor,
            'venue_performance': {},
            'active_congestion_events': 0,
            'network_anomalies': self.detect_network_anomalies()
        }
        
        # Count active congestion events
        for event in self.congestion_events:
            if (current_time >= event.timestamp and 
                current_time <= event.timestamp + event.duration_seconds):
                summary['active_congestion_events'] += 1
        
        # Venue-specific performance
        for venue in self.venue_configs:
            venue_stats = self.get_venue_latency_percentiles(venue, window_seconds=300)
            venue_stats['current_condition'] = self.current_conditions.get(venue, NetworkCondition.NORMAL).value
            summary['venue_performance'][venue] = venue_stats
        
        return summary

    def get_current_latency(self, venue: str) -> float:
        return self.measure_latency(venue).latency_us

    
    async def start_background_monitoring(self, interval_seconds: float = 1.0):
        """Start background network monitoring and event generation"""
        logger.info("Starting background network monitoring")
        
        while True:
            try:
                current_time = time.time()
                
                # Randomly generate network events
                for venue in self.venue_configs:
                    # Random congestion events
                    if random.random() < 0.001:  # 0.1% chance per second
                        self._simulate_congestion_event(venue, current_time)
                    
                    # Random condition changes
                    if random.random() < 0.01:  # 1% chance per second
                        conditions = list(NetworkCondition)
                        weights = [0.1, 0.4, 0.3, 0.15, 0.05]  # Favor normal conditions
                        new_condition = np.random.choice(conditions, p=weights)
                        self.current_conditions[venue] = new_condition
                
                # Update global congestion trends
                trend = random.uniform(-0.05, 0.05)
                self.global_congestion = max(0.0, min(1.0, self.global_congestion + trend))
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                await asyncio.sleep(interval_seconds)


# Example usage and testing
async def test_network_simulator():
    """Test the NetworkLatencySimulator"""
    
    simulator = NetworkLatencySimulator()
    
    # Start background monitoring
    monitor_task = asyncio.create_task(simulator.start_background_monitoring())
    
    print("Testing NetworkLatencySimulator...")
    
    # Simulate different market conditions
    test_conditions = [
        (0.01, 1.0, "Normal market"),
        (0.05, 2.0, "High volatility"),
        (0.02, 0.5, "Low volume"),
        (0.08, 3.0, "Market stress")
    ]
    
    for volatility, volume_factor, description in test_conditions:
        print(f"\n=== {description} ===")
        simulator.update_market_conditions(volatility, volume_factor)
        
        # Measure latency to all venues
        for venue in ['NYSE', 'NASDAQ', 'CBOE', 'IEX', 'ARCA']:
            measurement = simulator.measure_latency(venue)
            print(f"{venue}: {measurement.latency_us}μs, jitter={measurement.jitter_us}μs, "
                  f"loss={measurement.packet_loss}, condition={measurement.condition.value}")
        
        # Test route measurement
        route_measurement = simulator.measure_route_latency("NYSE_NASDAQ")
        if route_measurement:
            print(f"NYSE->NASDAQ route: {route_measurement.latency_us}μs")
        
        await asyncio.sleep(2)  # Let background events accumulate
    
    # Get performance summary
    summary = simulator.get_network_performance_summary()
    print(f"\n=== Performance Summary ===")
    print(f"Total measurements: {summary['total_measurements']}")
    print(f"Packet loss events: {summary['packet_loss_events']}")
    print(f"Global congestion: {summary['global_congestion']:.2f}")
    print(f"Active congestion events: {summary['active_congestion_events']}")
    print(f"Network anomalies: {len(summary['network_anomalies'])}")
    
    # Cancel monitoring
    monitor_task.cancel()
    
    return simulator

if __name__ == "__main__":
    asyncio.run(test_network_simulator())