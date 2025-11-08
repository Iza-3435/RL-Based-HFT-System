#!/usr/bin/env python3
"""
Enhanced Latency Simulation for HFT System

Provides realistic latency modeling including:
- Variable network delays with spikes
- Message queueing effects
- Exchange matching engine delays
- Venue-specific profiles
- Time-of-day and congestion effects
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Deque, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import asyncio
from datetime import datetime
import json
from simulator.trading_simulator import Fill

logger = logging.getLogger(__name__)


class LatencyComponent(Enum):
    """Components of total latency"""
    NETWORK = "network"
    QUEUE = "queue"
    EXCHANGE = "exchange"
    PROCESSING = "processing"


class CongestionLevel(Enum):
    """Network congestion levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LatencyBreakdown:
    """Detailed latency component breakdown"""
    timestamp: float
    venue: str
    symbol: str
    order_type: str
    
    # Latency components (microseconds)
    network_latency_us: float
    queue_delay_us: float
    exchange_delay_us: float
    processing_delay_us: float
    
    # Total and metadata
    total_latency_us: float
    predicted_latency_us: Optional[float] = None
    congestion_level: CongestionLevel = CongestionLevel.NORMAL
    time_of_day_factor: float = 1.0
    volatility_factor: float = 1.0
    
    @property
    def prediction_error_us(self) -> Optional[float]:
        """Calculate prediction error if available"""
        if self.predicted_latency_us is not None:
            return abs(self.total_latency_us - self.predicted_latency_us)
        return None
    
    @property
    def prediction_accuracy_pct(self) -> Optional[float]:
        """Calculate prediction accuracy percentage"""
        if self.predicted_latency_us is not None and self.predicted_latency_us > 0:
            error_pct = abs(self.total_latency_us - self.predicted_latency_us) / self.predicted_latency_us
            return max(0, 100 - error_pct * 100)
        return None


@dataclass
class VenueLatencyProfile:
    """Latency characteristics for each venue"""
    venue: str
    
    # Network latency parameters (microseconds)
    base_network_latency_us: float = 800.0
    network_latency_std_us: float = 150.0
    spike_probability: float = 0.001  # Probability of latency spike per message
    spike_multiplier: float = 4.0  # Spike latency multiplier
    
    # Queue parameters
    base_queue_capacity: int = 1000
    processing_rate_msg_per_sec: int = 100000
    queue_buildup_factor: float = 1.5  # How quickly queue builds under load
    
    # Exchange matching engine
    base_exchange_delay_us: float = 75.0
    exchange_delay_std_us: float = 25.0
    market_order_delay_multiplier: float = 0.8  # Market orders process faster
    limit_order_delay_multiplier: float = 1.2
    
    # Time-of-day effects
    market_open_multiplier: float = 2.0
    market_close_multiplier: float = 2.5
    lunch_multiplier: float = 0.7
    after_hours_multiplier: float = 1.2
    
    # Volatility correlation
    volatility_sensitivity: float = 1.8  # How much volatility affects latency
    
    # Infrastructure quality
    reliability_factor: float = 0.99  # Percentage of time venue is operating normally
    degraded_mode_multiplier: float = 3.0


class MessageQueue:
    """Simulates message queuing at venue gateways"""
    
    def __init__(self, capacity: int, processing_rate: int):
        self.capacity = capacity
        self.processing_rate = processing_rate  # messages per second
        self.queue: Deque[Tuple[float, str]] = deque()  # (arrival_time, message_id)
        self.last_processed_time = time.time()
        self.queue_full_events = 0
        
    def add_message(self, message_id: str) -> Tuple[bool, float]:
        """
        Add message to queue
        
        Returns:
            (success, queue_delay_seconds)
        """
        current_time = time.time()
        
        # Process any completed messages first
        self._process_completed_messages(current_time)
        
        # Check if queue is full
        if len(self.queue) >= self.capacity:
            self.queue_full_events += 1
            # In reality, this would cause order rejection
            # For simulation, we'll allow with severe delay penalty
            queue_delay = len(self.queue) / self.processing_rate * 2.0
            return False, queue_delay
        
        # Add to queue
        self.queue.append((current_time, message_id))
        
        # Calculate expected delay
        queue_position = len(self.queue) - 1
        queue_delay = queue_position / self.processing_rate
        
        return True, queue_delay
    
    def _process_completed_messages(self, current_time: float):
        """Remove messages that have been processed"""
        messages_to_process = int((current_time - self.last_processed_time) * self.processing_rate)
        
        for _ in range(min(messages_to_process, len(self.queue))):
            if self.queue:
                self.queue.popleft()
        
        self.last_processed_time = current_time
    
    def get_current_delay(self) -> float:
        """Get current queue delay in seconds"""
        return len(self.queue) / self.processing_rate
    
    def get_queue_stats(self) -> Dict[str, float]:
        """Get queue performance statistics"""
        return {
            'current_depth': len(self.queue),
            'capacity_utilization': len(self.queue) / self.capacity,
            'current_delay_ms': self.get_current_delay() * 1000,
            'queue_full_events': self.queue_full_events
        }


class LatencySimulator:
    """
    Comprehensive latency simulation system
    
    Models all components of trading latency with realistic distributions
    and time-varying effects
    """
    
    def __init__(self, venues: List[str]):
        self.venues = venues
        
        # Initialize venue profiles with realistic parameters
        self.venue_profiles = self._initialize_venue_profiles()
        
        # Message queues for each venue
        self.message_queues = self._initialize_message_queues()
        
        # Market state tracking
        self.current_volatility = defaultdict(lambda: 0.02)  # Default 2% volatility
        self.current_volume = defaultdict(lambda: 1.0)  # Volume multiplier
        self.congestion_level = CongestionLevel.NORMAL
        
        # Performance tracking
        self.latency_history: Deque[LatencyBreakdown] = deque(maxlen=10000)
        self.prediction_errors = []
        
        # Congestion simulation
        self.base_congestion = 0.0  # 0.0 to 1.0 scale
        self.congestion_events = deque(maxlen=100)
        
        logger.info(f"LatencySimulator initialized for {len(venues)} venues")
    
    def _initialize_venue_profiles(self) -> Dict[str, VenueLatencyProfile]:
        """Initialize realistic latency profiles for each venue"""
        profiles = {}
        
        # NYSE - Premium connectivity, higher fees but good performance
        profiles['NYSE'] = VenueLatencyProfile(
            venue='NYSE',
            base_network_latency_us=850.0,
            network_latency_std_us=120.0,
            spike_probability=0.0008,
            spike_multiplier=3.5,
            base_queue_capacity=1500,
            processing_rate_msg_per_sec=120000,
            base_exchange_delay_us=65.0,
            exchange_delay_std_us=20.0,
            market_open_multiplier=1.8,
            market_close_multiplier=2.2,
            volatility_sensitivity=1.5,
            reliability_factor=0.995
        )
        
        # NASDAQ - High-tech infrastructure, generally fast
        profiles['NASDAQ'] = VenueLatencyProfile(
            venue='NASDAQ',
            base_network_latency_us=750.0,
            network_latency_std_us=100.0,
            spike_probability=0.0005,
            spike_multiplier=3.0,
            base_queue_capacity=2000,
            processing_rate_msg_per_sec=150000,
            base_exchange_delay_us=55.0,
            exchange_delay_std_us=15.0,
            market_open_multiplier=1.6,
            market_close_multiplier=2.0,
            volatility_sensitivity=1.3,
            reliability_factor=0.997
        )
        
        # CBOE - Good for options, moderate latency
        profiles['CBOE'] = VenueLatencyProfile(
            venue='CBOE',
            base_network_latency_us=950.0,
            network_latency_std_us=180.0,
            spike_probability=0.0012,
            spike_multiplier=4.0,
            base_queue_capacity=1200,
            processing_rate_msg_per_sec=90000,
            base_exchange_delay_us=85.0,
            exchange_delay_std_us=30.0,
            market_open_multiplier=2.0,
            market_close_multiplier=2.5,
            volatility_sensitivity=1.8,
            reliability_factor=0.992
        )
        
        # IEX - Speed bump by design, consistent but slower
        profiles['IEX'] = VenueLatencyProfile(
            venue='IEX',
            base_network_latency_us=1200.0,  # Includes speed bump
            network_latency_std_us=50.0,  # Very consistent
            spike_probability=0.0002,
            spike_multiplier=2.0,
            base_queue_capacity=1000,
            processing_rate_msg_per_sec=80000,
            base_exchange_delay_us=90.0,
            exchange_delay_std_us=10.0,  # Very predictable
            market_open_multiplier=1.2,
            market_close_multiplier=1.3,
            volatility_sensitivity=0.8,  # Less affected by volatility
            reliability_factor=0.998
        )
        
        # ARCA - Electronic, variable performance
        profiles['ARCA'] = VenueLatencyProfile(
            venue='ARCA',
            base_network_latency_us=800.0,
            network_latency_std_us=140.0,
            spike_probability=0.0010,
            spike_multiplier=3.8,
            base_queue_capacity=1300,
            processing_rate_msg_per_sec=110000,
            base_exchange_delay_us=70.0,
            exchange_delay_std_us=25.0,
            market_open_multiplier=1.9,
            market_close_multiplier=2.3,
            volatility_sensitivity=1.6,
            reliability_factor=0.993
        )
        
        return profiles
    
    def _initialize_message_queues(self) -> Dict[str, MessageQueue]:
        """Initialize message queues for each venue"""
        queues = {}
        
        for venue, profile in self.venue_profiles.items():
            queues[venue] = MessageQueue(
                capacity=profile.base_queue_capacity,
                processing_rate=profile.processing_rate_msg_per_sec
            )
        
        return queues
    
    def update_market_conditions(self, symbol: str, volatility: float, 
                                volume_factor: float, timestamp: float = None):
        """Update market conditions affecting latency"""
        if timestamp is None:
            timestamp = time.time()
        
        self.current_volatility[symbol] = volatility
        self.current_volume[symbol] = volume_factor
        
        # Update global congestion based on market conditions
        volatility_stress = min(volatility / 0.05, 2.0)  # Normalize to 5% vol
        volume_stress = min(volume_factor / 2.0, 2.0)  # Normalize to 2x volume
        
        # Combine stresses to determine congestion
        combined_stress = (volatility_stress + volume_stress) / 2.0
        
        if combined_stress > 1.5:
            self.congestion_level = CongestionLevel.CRITICAL
            self.base_congestion = 0.8
        elif combined_stress > 1.2:
            self.congestion_level = CongestionLevel.HIGH
            self.base_congestion = 0.6
        elif combined_stress > 0.8:
            self.congestion_level = CongestionLevel.NORMAL
            self.base_congestion = 0.3
        else:
            self.congestion_level = CongestionLevel.LOW
            self.base_congestion = 0.1
    
    def simulate_latency(self, venue: str, symbol: str, order_type: str = "limit",
                        predicted_latency_us: Optional[float] = None,
                        timestamp: float = None) -> LatencyBreakdown:
        """
        Simulate complete latency for an order
        
        Args:
            venue: Target venue
            symbol: Trading symbol
            order_type: Order type (market, limit, etc.)
            predicted_latency_us: ML prediction for validation
            timestamp: Order timestamp
            
        Returns:
            Detailed latency breakdown
        """
        if timestamp is None:
            timestamp = time.time()
        
        if venue not in self.venue_profiles:
            raise ValueError(f"Unknown venue: {venue}")
        
        profile = self.venue_profiles[venue]
        
        # Get current market factors
        time_factor = self._get_time_of_day_factor(timestamp)
        volatility_factor = self._get_volatility_factor(symbol, profile)
        congestion_factor = self._get_congestion_factor()
        
        # 1. Network Latency Component
        network_latency = self._simulate_network_latency(profile, time_factor, congestion_factor)
        
        # 2. Queue Delay Component
        queue_delay = self._simulate_queue_delay(venue, congestion_factor)
        
        # 3. Exchange Processing Delay
        exchange_delay = self._simulate_exchange_delay(profile, order_type, volatility_factor)
        
        # 4. Additional Processing Delay
        processing_delay = self._simulate_processing_delay(profile, congestion_factor)
        
        # Calculate total latency
        total_latency = network_latency + queue_delay + exchange_delay + processing_delay
        
        # Create detailed breakdown
        breakdown = LatencyBreakdown(
            timestamp=timestamp,
            venue=venue,
            symbol=symbol,
            order_type=order_type,
            network_latency_us=network_latency,
            queue_delay_us=queue_delay,
            exchange_delay_us=exchange_delay,
            processing_delay_us=processing_delay,
            total_latency_us=total_latency,
            predicted_latency_us=predicted_latency_us,
            congestion_level=self.congestion_level,
            time_of_day_factor=time_factor,
            volatility_factor=volatility_factor
        )
        
        # Track for analytics
        self.latency_history.append(breakdown)
        
        # Track prediction accuracy
        if predicted_latency_us is not None:
            error = abs(total_latency - predicted_latency_us)
            self.prediction_errors.append(error)
        
        return breakdown
    
    def _simulate_network_latency(self, profile: VenueLatencyProfile, 
                                 time_factor: float, congestion_factor: float) -> float:
        """Simulate network latency with realistic distribution"""
        # Base latency with time-of-day effects
        base_latency = profile.base_network_latency_us * time_factor
        
        # Lognormal distribution for network latency (realistic for networks)
        # Convert to lognormal parameters
        mu = np.log(base_latency)
        sigma = profile.network_latency_std_us / base_latency
        
        network_latency = np.random.lognormal(mu, sigma)
        
        # Apply congestion effects
        network_latency *= (1.0 + congestion_factor * 0.8)
        
        # Random latency spikes
        if np.random.random() < profile.spike_probability * congestion_factor:
            network_latency *= profile.spike_multiplier
        
        # Venue reliability issues
        if np.random.random() > profile.reliability_factor:
            network_latency *= profile.degraded_mode_multiplier
        
        return max(50.0, network_latency)  # Minimum 50μs
    
    def _simulate_queue_delay(self, venue: str, congestion_factor: float) -> float:
        """Simulate message queue delay"""
        queue = self.message_queues[venue]
        
        # Increase queue load during congestion
        if congestion_factor > 0.5:
            # Simulate additional messages during high congestion
            num_additional = int(congestion_factor * 50)
            for i in range(num_additional):
                queue.add_message(f"congestion_msg_{i}")
        
        # Add our message and get delay
        success, queue_delay_seconds = queue.add_message(f"order_{time.time()}")
        
        # Convert to microseconds
        queue_delay_us = queue_delay_seconds * 1e6
        
        # Add randomness for processing variations
        queue_delay_us += np.random.exponential(20.0)  # Exponential for queueing systems
        
        return max(0.0, queue_delay_us)
    
    def _simulate_exchange_delay(self, profile: VenueLatencyProfile, 
                                order_type: str, volatility_factor: float) -> float:
        """Simulate exchange matching engine delay"""
        base_delay = profile.base_exchange_delay_us
        
        # Order type affects processing time
        if order_type.lower() == "market":
            base_delay *= profile.market_order_delay_multiplier
        elif order_type.lower() in ["limit", "stop_limit"]:
            base_delay *= profile.limit_order_delay_multiplier
        
        # Normal distribution for exchange processing
        exchange_delay = np.random.normal(base_delay, profile.exchange_delay_std_us)
        
        # Volatility affects matching complexity
        exchange_delay *= (1.0 + (volatility_factor - 1.0) * 0.3)
        
        return max(10.0, exchange_delay)  # Minimum 10μs
    
    def _simulate_processing_delay(self, profile: VenueLatencyProfile, 
                                  congestion_factor: float) -> float:
        """Simulate additional processing delays"""
        # Base processing overhead
        base_processing = 25.0  # microseconds
        
        # Exponential distribution for processing time
        processing_delay = np.random.exponential(base_processing)
        
        # Congestion increases processing time
        processing_delay *= (1.0 + congestion_factor * 0.5)
        
        return processing_delay
    
    def _get_time_of_day_factor(self, timestamp: float) -> float:
        """Calculate time-of-day latency multiplier"""
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        minute = dt.minute
        
        # Market hours: 9:30 AM - 4:00 PM EST
        if hour == 9 and minute >= 30:
            # Market open
            return 2.0
        elif hour == 10:
            # Post-open activity
            return 1.6
        elif 11 <= hour <= 12:
            # Mid-morning
            return 1.2
        elif hour == 12:
            # Lunch period
            return 0.8
        elif 13 <= hour <= 14:
            # Afternoon
            return 1.1
        elif hour == 15:
            # Pre-close
            return 1.8
        elif hour == 16 and minute <= 30:
            # Market close
            return 2.5
        elif 17 <= hour <= 19:
            # After hours
            return 1.3
        else:
            # Overnight
            return 0.9
    
    def _get_volatility_factor(self, symbol: str, profile: VenueLatencyProfile) -> float:
        """Calculate volatility-based latency multiplier"""
        volatility = self.current_volatility[symbol]
        
        # Normalize volatility (2% is baseline)
        vol_ratio = volatility / 0.02
        
        # Apply venue-specific sensitivity
        factor = 1.0 + (vol_ratio - 1.0) * (profile.volatility_sensitivity - 1.0)
        
        return max(0.5, min(3.0, factor))  # Clamp between 0.5x and 3.0x
    
    def _get_congestion_factor(self) -> float:
        """Get current network congestion factor"""
        # Add random variation to base congestion
        random_variation = np.random.normal(0, 0.1)
        congestion = self.base_congestion + random_variation
        
        return max(0.0, min(1.0, congestion))
    
    def get_venue_latency_stats(self, venue: str, window_minutes: int = 5) -> Dict[str, float]:
        """Get latency statistics for a venue over recent window"""
        current_time = time.time()
        cutoff_time = current_time - (window_minutes * 60)
        
        # Filter recent latency data
        recent_latencies = [
            breakdown for breakdown in self.latency_history
            if breakdown.venue == venue and breakdown.timestamp >= cutoff_time
        ]
        
        if not recent_latencies:
            return {}
        
        # Extract total latencies
        total_latencies = [b.total_latency_us for b in recent_latencies]
        
        # Calculate statistics
        stats = {
            'count': len(total_latencies),
            'mean_us': np.mean(total_latencies),
            'median_us': np.median(total_latencies),
            'std_us': np.std(total_latencies),
            'min_us': np.min(total_latencies),
            'max_us': np.max(total_latencies),
            'p95_us': np.percentile(total_latencies, 95),
            'p99_us': np.percentile(total_latencies, 99)
        }
        
        # Component breakdown
        component_stats = {}
        for component in ['network_latency_us', 'queue_delay_us', 'exchange_delay_us', 'processing_delay_us']:
            values = [getattr(b, component) for b in recent_latencies]
            component_stats[f'{component}_mean'] = np.mean(values)
            component_stats[f'{component}_contribution_pct'] = np.mean(values) / stats['mean_us'] * 100
        
        stats.update(component_stats)
        
        # Queue statistics
        if venue in self.message_queues:
            queue_stats = self.message_queues[venue].get_queue_stats()
            stats.update({f'queue_{k}': v for k, v in queue_stats.items()})
        
        return stats
    
    def get_prediction_accuracy_stats(self) -> Dict[str, float]:
        """Get ML prediction accuracy statistics"""
        if not self.prediction_errors:
            return {}
        
        errors = np.array(self.prediction_errors)
        
        # Recent errors (last 100 predictions)
        recent_errors = errors[-100:] if len(errors) > 100 else errors
        
        return {
            'total_predictions': len(errors),
            'mean_error_us': np.mean(errors),
            'median_error_us': np.median(errors),
            'rmse_us': np.sqrt(np.mean(errors**2)),
            'mean_absolute_percentage_error': np.mean(recent_errors) / np.mean([
                b.total_latency_us for b in self.latency_history[-len(recent_errors):]
            ]) * 100 if self.latency_history else 0,
            'prediction_within_10pct': np.sum(recent_errors < np.mean(recent_errors) * 0.1) / len(recent_errors) * 100
        }
    
    def get_congestion_analysis(self) -> Dict[str, Any]:
        """Analyze current congestion and impact"""
        current_time = time.time()
        
        # Recent latency impact
        recent_latencies = [
            b for b in self.latency_history
            if current_time - b.timestamp < 300  # Last 5 minutes
        ]
        
        if not recent_latencies:
            return {}
        
        # Latency by congestion level
        latency_by_congestion = defaultdict(list)
        for breakdown in recent_latencies:
            latency_by_congestion[breakdown.congestion_level].append(breakdown.total_latency_us)
        
        congestion_stats = {}
        for level, latencies in latency_by_congestion.items():
            if latencies:
                congestion_stats[level.value] = {
                    'count': len(latencies),
                    'mean_latency_us': np.mean(latencies),
                    'latency_increase_pct': (np.mean(latencies) / np.mean([b.total_latency_us for b in recent_latencies]) - 1) * 100
                }
        
        return {
            'current_congestion_level': self.congestion_level.value,
            'base_congestion_factor': self.base_congestion,
            'latency_by_congestion': congestion_stats,
            'queue_states': {venue: queue.get_queue_stats() for venue, queue in self.message_queues.items()}
        }
    
    def export_latency_data(self, filename: str = None) -> List[Dict]:
        """Export latency data for analysis"""
        data = []
        
        for breakdown in self.latency_history:
            record = {
                'timestamp': breakdown.timestamp,
                'venue': breakdown.venue,
                'symbol': breakdown.symbol,
                'order_type': breakdown.order_type,
                'total_latency_us': breakdown.total_latency_us,
                'network_latency_us': breakdown.network_latency_us,
                'queue_delay_us': breakdown.queue_delay_us,
                'exchange_delay_us': breakdown.exchange_delay_us,
                'processing_delay_us': breakdown.processing_delay_us,
                'predicted_latency_us': breakdown.predicted_latency_us,
                'prediction_error_us': breakdown.prediction_error_us,
                'prediction_accuracy_pct': breakdown.prediction_accuracy_pct,
                'congestion_level': breakdown.congestion_level.value,
                'time_of_day_factor': breakdown.time_of_day_factor,
                'volatility_factor': breakdown.volatility_factor
            }
            data.append(record)
        
        if filename:
            import json
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        
        return data
    
    def reset_statistics(self):
        """Reset all accumulated statistics"""
        self.latency_history.clear()
        self.prediction_errors.clear()
        
        # Reset queue statistics
        for queue in self.message_queues.values():
            queue.queue.clear()
            queue.queue_full_events = 0
        
        logger.info("Latency simulator statistics reset")


# Integration functions for existing architecture
def integrate_latency_simulation(simulator_instance, latency_simulator: LatencySimulator):
    """
    Integration function to add latency simulation to existing OrderExecutionEngine
    
    This patches the existing execute_order method to use realistic latency
    """
    original_execute_order = simulator_instance.execute_order
    
    async def enhanced_execute_order(order, market_state, predicted_latency_us=None):
        """Enhanced execute_order with realistic latency simulation"""
        # Simulate realistic latency
        latency_breakdown = latency_simulator.simulate_latency(
            venue=order.venue,
            symbol=order.symbol,
            order_type=order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
            predicted_latency_us=predicted_latency_us or getattr(order, 'predicted_latency_us', None),
            timestamp=order.timestamp
        )
        
        # Use simulated latency instead of random/predicted latency
        actual_latency_us = latency_breakdown.total_latency_us
        
        # Store latency breakdown in order for analysis
        if hasattr(order, '__dict__'):
            order.latency_breakdown = latency_breakdown
        
        # Call original execute_order with simulated latency
        return await original_execute_order(order, market_state, actual_latency_us)
    
    # Replace the method
    simulator_instance.execute_order = enhanced_execute_order
    
    # Add latency analytics methods
    simulator_instance.get_latency_stats = lambda venue, window=5: latency_simulator.get_venue_latency_stats(venue, window)
    simulator_instance.get_prediction_accuracy = latency_simulator.get_prediction_accuracy_stats
    simulator_instance.get_congestion_analysis = latency_simulator.get_congestion_analysis
    
    return latency_simulator


def create_latency_configuration() -> Dict[str, Any]:
    """Create configuration for different market conditions"""
    return {
        'market_conditions': {
            'normal': {
                'volatility_threshold': 0.02,
                'volume_threshold': 1.0,
                'expected_latency_range_us': (500, 1500)
            },
            'volatile': {
                'volatility_threshold': 0.05,
                'volume_threshold': 2.0,
                'expected_latency_range_us': (800, 3000)
            },
            'stressed': {
                'volatility_threshold': 0.08,
                'volume_threshold': 3.0,
                'expected_latency_range_us': (1200, 5000)
            }
        },
        'venue_priorities': {
            # Order venues by typical latency performance
            'low_latency': ['NASDAQ', 'ARCA', 'NYSE'],
            'consistent': ['IEX', 'NYSE', 'NASDAQ'],
            'high_volume': ['NYSE', 'NASDAQ', 'ARCA']
        },
        'latency_thresholds': {
            'market_making': {
                'acceptable_us': 1000,
                'optimal_us': 500
            },
            'arbitrage': {
                'acceptable_us': 300,
                'optimal_us': 150
            },
            'momentum': {
                'acceptable_us': 2000,
                'optimal_us': 800
            }
        },
        'prediction_accuracy_targets': {
            'minimum_accuracy_pct': 75.0,
            'target_accuracy_pct': 85.0,
            'excellent_accuracy_pct': 95.0
        }
    }


# Enhanced Trading Simulator Integration
class EnhancedOrderExecutionEngine:
    """
    Enhanced execution engine with realistic latency simulation
    
    Integrates seamlessly with existing TradingSimulator architecture
    """
    
    def __init__(self, venues: List[str], fee_schedule: Dict[str, Dict] = None):
        # Initialize base execution engine components
        from simulator.trading_simulator import MarketImpactModel
        
        self.market_impact_model = MarketImpactModel()
        self.execution_queue = []
        self.order_book = defaultdict(lambda: {'bids': [], 'asks': []})
        
        # Enhanced latency simulation
        self.latency_simulator = LatencySimulator(venues)
        
        # Fee schedule
        self.fee_schedule = fee_schedule or {
            'NYSE': {'maker_fee': -0.20, 'taker_fee': 0.30},
            'NASDAQ': {'maker_fee': -0.25, 'taker_fee': 0.30},
            'CBOE': {'maker_fee': -0.23, 'taker_fee': 0.28},
            'IEX': {'maker_fee': 0.0, 'taker_fee': 0.09},
            'ARCA': {'maker_fee': -0.20, 'taker_fee': 0.30}
        }
        
        # Enhanced execution statistics
        self.fill_count = 0
        self.total_latency_us = 0
        self.total_slippage_bps = 0
        self.latency_cost_bps = 0
        
        # Performance tracking by venue
        self.venue_performance = defaultdict(lambda: {
            'fills': 0,
            'total_latency': 0,
            'total_slippage': 0,
            'prediction_errors': []
        })
        
        logger.info("EnhancedOrderExecutionEngine initialized with realistic latency simulation")
    
    async def execute_order(self, order, market_state: Dict, 
                           predicted_latency_us: Optional[float] = None) -> Optional['Fill']:
        """
        Execute order with comprehensive latency simulation
        
        Args:
            order: Order object to execute
            market_state: Current market conditions
            predicted_latency_us: ML-predicted latency for validation
            
        Returns:
            Fill object with latency analytics
        """
        from simulator.trading_simulator import Fill, OrderSide
        
        # Update latency simulator with current market conditions
        self.latency_simulator.update_market_conditions(
            symbol=order.symbol,
            volatility=market_state.get('volatility', 0.02),
            volume_factor=market_state.get('volume', 1.0) / 1000000,  # Normalize volume
            timestamp=order.timestamp
        )
        
        # Simulate realistic latency
        latency_breakdown = self.latency_simulator.simulate_latency(
            venue=order.venue,
            symbol=order.symbol,
            order_type=order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
            predicted_latency_us=predicted_latency_us or getattr(order, 'predicted_latency_us', None),
            timestamp=order.timestamp
        )
        
        actual_latency_us = latency_breakdown.total_latency_us
        
        # Calculate market impact during latency period
        arrival_price = market_state['mid_price']
        
        # Enhanced price movement simulation during latency
        price_drift = self._simulate_enhanced_price_drift(
            arrival_price, actual_latency_us, market_state, latency_breakdown
        )
        execution_price = arrival_price + price_drift
        
        # Calculate market impact with latency considerations
        permanent_impact, temporary_impact = self.market_impact_model.calculate_impact(
            order, market_state, actual_latency_us
        )
        
        # Enhanced latency cost calculation
        latency_cost_bps = self._calculate_latency_cost(
            order, latency_breakdown, market_state
        )
        
        # Determine fill price with all effects
        fill_price = self._calculate_fill_price(
            order, market_state, temporary_impact, latency_cost_bps
        )
        
        # Calculate fees/rebates
        is_maker = self._determine_maker_status(order, market_state, actual_latency_us)
        fee_structure = self.fee_schedule[order.venue]
        
        fee_bps = fee_structure['maker_fee'] if is_maker else fee_structure['taker_fee']
        fees = abs(fee_bps) * fill_price * order.quantity / 10000 if fee_bps > 0 else 0
        rebate = abs(fee_bps) * fill_price * order.quantity / 10000 if fee_bps < 0 else 0
        
        # Enhanced slippage calculation
        slippage_bps = self._calculate_enhanced_slippage(
            order, fill_price, arrival_price, latency_breakdown
        )
        
        # Create enhanced fill object
        fill = Fill(
            fill_id=f"F{self.fill_count:08d}",
            order_id=order.order_id,
            symbol=order.symbol,
            venue=order.venue,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=order.timestamp + actual_latency_us / 1e6,
            fees=fees,
            rebate=rebate,
            latency_us=actual_latency_us,
            slippage_bps=slippage_bps,
            market_impact_bps=temporary_impact
        )
        
        # Store latency breakdown in fill for analysis
        fill.latency_breakdown = latency_breakdown
        fill.latency_cost_bps = latency_cost_bps
        fill.prediction_error_us = latency_breakdown.prediction_error_us
        
        # Update order status with enhanced information
        order.status = 'FILLED'  # Assuming OrderStatus enum
        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.fill_timestamp = fill.timestamp
        order.latency_us = actual_latency_us
        order.latency_breakdown = latency_breakdown
        
        # Update comprehensive statistics
        self._update_enhanced_statistics(fill, latency_breakdown)
        
        # Apply permanent market impact
        self._apply_permanent_impact(market_state, order.side, permanent_impact)
        
        return fill
    
    def _simulate_enhanced_price_drift(self, price: float, latency_us: float,
                                     market_state: Dict, latency_breakdown: LatencyBreakdown) -> float:
        """Enhanced price movement simulation considering latency components"""
        # Base drift calculation
        latency_days = latency_us / (1e6 * 60 * 60 * 6.5)
        volatility = market_state.get('volatility', 0.02)
        
        # Component-specific price impact
        # Network latency allows more time for adverse selection
        network_drift = (latency_breakdown.network_latency_us / 1000) * volatility * np.random.randn() * 0.1
        
        # Queue delays indicate congestion - prices move more during congestion
        queue_drift = (latency_breakdown.queue_delay_us / 1000) * volatility * np.random.randn() * 0.15
        
        # Exchange delays might indicate complexity - neutral effect
        exchange_drift = 0
        
        total_drift = network_drift + queue_drift + exchange_drift
        
        # Apply momentum effect - prices tend to continue moving in same direction during latency
        if hasattr(market_state, 'recent_price_direction'):
            momentum_factor = market_state['recent_price_direction'] * 0.05
            total_drift += momentum_factor * (latency_us / 1000)
        
        return price * total_drift
    
    def _calculate_latency_cost(self, order, latency_breakdown: LatencyBreakdown,
                               market_state: Dict) -> float:
        """Calculate opportunity cost due to latency in basis points"""
        # Base latency cost - opportunity cost of delayed execution
        base_cost_bps = latency_breakdown.total_latency_us / 100 * 0.1  # 0.1 bps per 100μs
        
        # Volatility amplifies latency cost
        volatility_multiplier = market_state.get('volatility', 0.02) / 0.02
        vol_adjusted_cost = base_cost_bps * volatility_multiplier
        
        # Queue delays are especially costly for time-sensitive strategies
        if order.strategy.value == 'arbitrage' and latency_breakdown.queue_delay_us > 100:
            queue_penalty = (latency_breakdown.queue_delay_us - 100) / 100 * 0.5
            vol_adjusted_cost += queue_penalty
        
        # Network spikes are costly for all strategies
        if latency_breakdown.network_latency_us > 2000:  # Spike detected
            spike_penalty = (latency_breakdown.network_latency_us - 2000) / 1000 * 0.3
            vol_adjusted_cost += spike_penalty
        
        return vol_adjusted_cost
    
    def _calculate_fill_price(self, order, market_state: Dict, 
                             temporary_impact: float, latency_cost_bps: float) -> float:
        """Calculate final fill price with all effects"""
        if order.order_type.value == 'market':
            # Market orders cross the spread and pay impact
            if order.side.value == 'buy':
                base_price = market_state['ask_price']
                price_impact = base_price * (temporary_impact + latency_cost_bps) / 10000
                fill_price = base_price + price_impact
            else:
                base_price = market_state['bid_price']
                price_impact = base_price * (temporary_impact + latency_cost_bps) / 10000
                fill_price = base_price - price_impact
        
        elif order.order_type.value == 'limit':
            # Limit orders - check if marketable
            if order.side.value == 'buy':
                if order.price >= market_state['ask_price']:
                    # Marketable buy
                    base_price = market_state['ask_price']
                    price_impact = base_price * (temporary_impact + latency_cost_bps) / 10000
                    fill_price = base_price + price_impact
                else:
                    # Posted as maker
                    fill_price = order.price
            else:  # sell
                if order.price <= market_state['bid_price']:
                    # Marketable sell
                    base_price = market_state['bid_price']
                    price_impact = base_price * (temporary_impact + latency_cost_bps) / 10000
                    fill_price = base_price - price_impact
                else:
                    # Posted as maker
                    fill_price = order.price
        else:
            # Other order types - simplified
            fill_price = market_state['mid_price']
        
        return fill_price
    
    def _determine_maker_status(self, order, market_state: Dict, latency_us: float) -> bool:
        """Determine if order is maker or taker considering latency"""
        if order.order_type.value != 'limit':
            return False  # Only limit orders can be makers
        
        # High latency reduces chance of being maker due to price movement
        latency_factor = max(0.1, 1.0 - (latency_us - 500) / 2000)  # Reduce probability for high latency
        
        if order.side.value == 'buy':
            is_marketable = order.price >= market_state['ask_price']
        else:
            is_marketable = order.price <= market_state['bid_price']
        
        if is_marketable:
            return False  # Marketable orders are takers
        
        # Non-marketable orders have reduced maker probability due to latency
        maker_probability = 0.7 * latency_factor  # Base 70% chance, reduced by latency
        return np.random.random() < maker_probability
    
    def _calculate_enhanced_slippage(self, order, fill_price: float, 
                                   arrival_price: float, latency_breakdown: LatencyBreakdown) -> float:
        """Calculate enhanced slippage including latency effects"""
        if order.order_type.value == 'market':
            expected_price = arrival_price
        else:
            expected_price = order.price
        
        # Base slippage
        base_slippage_bps = abs(fill_price - expected_price) / expected_price * 10000
        
        # Latency-attributed slippage
        latency_slippage = latency_breakdown.total_latency_us / 1000 * 0.1  # 0.1 bps per ms
        
        # Component-specific slippage attribution
        network_slippage = latency_breakdown.network_latency_us / 1000 * 0.05
        queue_slippage = latency_breakdown.queue_delay_us / 1000 * 0.15  # Queue delays more costly
        
        total_slippage = base_slippage_bps + latency_slippage
        
        return total_slippage
    
    def _update_enhanced_statistics(self, fill, latency_breakdown: LatencyBreakdown):
        """Update comprehensive execution statistics"""
        # Global statistics
        self.fill_count += 1
        self.total_latency_us += fill.latency_us
        self.total_slippage_bps += fill.slippage_bps
        self.latency_cost_bps += getattr(fill, 'latency_cost_bps', 0)
        
        # Venue-specific statistics
        venue_stats = self.venue_performance[fill.venue]
        venue_stats['fills'] += 1
        venue_stats['total_latency'] += fill.latency_us
        venue_stats['total_slippage'] += fill.slippage_bps
        
        # Track prediction accuracy
        if latency_breakdown.prediction_error_us is not None:
            venue_stats['prediction_errors'].append(latency_breakdown.prediction_error_us)
    
    def _apply_permanent_impact(self, market_state: Dict, side, impact_bps: float):
        """Apply permanent market impact to prices"""
        impact_pct = impact_bps / 10000
        
        if side.value == 'buy':
            market_state['bid_price'] *= (1 + impact_pct * 0.5)
            market_state['ask_price'] *= (1 + impact_pct * 0.5)
            market_state['mid_price'] *= (1 + impact_pct * 0.5)
        else:
            market_state['bid_price'] *= (1 - impact_pct * 0.5)
            market_state['ask_price'] *= (1 - impact_pct * 0.5)
            market_state['mid_price'] *= (1 - impact_pct * 0.5)
    
    def get_enhanced_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        base_stats = {
            'total_fills': self.fill_count,
            'avg_latency_us': self.total_latency_us / max(self.fill_count, 1),
            'avg_slippage_bps': self.total_slippage_bps / max(self.fill_count, 1),
            'avg_latency_cost_bps': self.latency_cost_bps / max(self.fill_count, 1),
            'fill_rate': 1.0  # Simplified
        }
        
        # Venue performance breakdown
        venue_stats = {}
        for venue, stats in self.venue_performance.items():
            if stats['fills'] > 0:
                venue_stats[venue] = {
                    'fills': stats['fills'],
                    'avg_latency_us': stats['total_latency'] / stats['fills'],
                    'avg_slippage_bps': stats['total_slippage'] / stats['fills'],
                    'prediction_accuracy': self._calculate_venue_prediction_accuracy(stats)
                }
        
        # Latency simulator statistics
        latency_stats = {
            'prediction_accuracy': self.latency_simulator.get_prediction_accuracy_stats(),
            'congestion_analysis': self.latency_simulator.get_congestion_analysis()
        }
        
        return {
            'execution_stats': base_stats,
            'venue_performance': venue_stats,
            'latency_analysis': latency_stats
        }
    
    def _calculate_venue_prediction_accuracy(self, venue_stats: Dict) -> Dict[str, float]:
        """Calculate prediction accuracy for a venue"""
        errors = venue_stats['prediction_errors']
        if not errors:
            return {}
        
        # Calculate average latency for this venue
        avg_latency = venue_stats['total_latency'] / venue_stats['fills']
        
        return {
            'mean_error_us': np.mean(errors),
            'rmse_us': np.sqrt(np.mean(np.array(errors)**2)),
            'mape_pct': np.mean(errors) / avg_latency * 100,
            'within_10pct': np.sum(np.array(errors) < avg_latency * 0.1) / len(errors) * 100
        }
    
    def get_venue_latency_rankings(self) -> List[Tuple[str, float]]:
        """Get venues ranked by average latency performance"""
        rankings = []
        
        for venue in self.latency_simulator.venues:
            stats = self.latency_simulator.get_venue_latency_stats(venue, window_minutes=10)
            if stats:
                rankings.append((venue, stats['mean_us']))
        
        # Sort by average latency (ascending)
        rankings.sort(key=lambda x: x[1])
        
        return rankings
    
    def get_latency_cost_analysis(self) -> Dict[str, Any]:
        """Analyze costs specifically attributed to latency"""
        total_fills = max(self.fill_count, 1)
        
        # Calculate latency costs
        avg_latency_cost_bps = self.latency_cost_bps / total_fills
        
        # Estimate dollar impact (simplified)
        avg_trade_value = 50000  # Assume $50k average trade
        latency_cost_per_trade = avg_trade_value * avg_latency_cost_bps / 10000
        total_latency_cost = latency_cost_per_trade * total_fills
        
        # Breakdown by venue
        venue_costs = {}
        for venue, stats in self.venue_performance.items():
            if stats['fills'] > 0:
                venue_avg_latency = stats['total_latency'] / stats['fills']
                venue_cost_bps = venue_avg_latency / 100 * 0.1  # Simplified calculation
                venue_costs[venue] = {
                    'avg_latency_us': venue_avg_latency,
                    'estimated_cost_bps': venue_cost_bps,
                    'fills': stats['fills']
                }
        
        return {
            'total_latency_cost_usd': total_latency_cost,
            'avg_latency_cost_per_trade_usd': latency_cost_per_trade,
            'avg_latency_cost_bps': avg_latency_cost_bps,
            'venue_cost_breakdown': venue_costs,
            'potential_savings_analysis': self._calculate_potential_savings()
        }
    
    def _calculate_potential_savings(self) -> Dict[str, float]:
        """Calculate potential savings from latency optimization"""
        if not self.venue_performance:
            return {}
        
        # Find best performing venue
        best_venue_latency = min(
            stats['total_latency'] / stats['fills']
            for stats in self.venue_performance.values()
            if stats['fills'] > 0
        )
        
        current_avg_latency = self.total_latency_us / max(self.fill_count, 1)
        
        # Calculate potential improvement
        latency_improvement_us = current_avg_latency - best_venue_latency
        improvement_pct = latency_improvement_us / current_avg_latency * 100
        
        # Estimate cost savings
        cost_savings_bps = latency_improvement_us / 100 * 0.1
        
        return {
            'current_avg_latency_us': current_avg_latency,
            'best_venue_latency_us': best_venue_latency,
            'potential_improvement_us': latency_improvement_us,
            'potential_improvement_pct': improvement_pct,
            'estimated_savings_bps': cost_savings_bps
        }


# Analytics and Reporting Functions
class LatencyAnalytics:
    """Comprehensive latency analytics and reporting"""
    
    def __init__(self, execution_engine: EnhancedOrderExecutionEngine):
        self.execution_engine = execution_engine
        self.latency_simulator = execution_engine.latency_simulator
    
    def generate_latency_report(self, timeframe_minutes: int = 60) -> Dict[str, Any]:
        """Generate comprehensive latency performance report"""
        report = {
            'report_timestamp': time.time(),
            'timeframe_minutes': timeframe_minutes,
            'summary': self._generate_summary_stats(timeframe_minutes),
            'venue_analysis': self._generate_venue_analysis(timeframe_minutes),
            'component_breakdown': self._generate_component_breakdown(timeframe_minutes),
            'prediction_performance': self._generate_prediction_analysis(),
            'cost_impact': self._generate_cost_impact_analysis(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_summary_stats(self, timeframe_minutes: int) -> Dict[str, Any]:
        """Generate summary statistics"""
        execution_stats = self.execution_engine.get_enhanced_execution_stats()
        
        return {
            'total_orders_executed': execution_stats['execution_stats']['total_fills'],
            'avg_total_latency_us': execution_stats['execution_stats']['avg_latency_us'],
            'avg_slippage_bps': execution_stats['execution_stats']['avg_slippage_bps'],
            'avg_latency_cost_bps': execution_stats['execution_stats']['avg_latency_cost_bps'],
            'prediction_accuracy_pct': execution_stats['latency_analysis']['prediction_accuracy'].get('prediction_within_10pct', 0),
            'current_congestion_level': execution_stats['latency_analysis']['congestion_analysis']['current_congestion_level']
        }
    
    def _generate_venue_analysis(self, timeframe_minutes: int) -> Dict[str, Any]:
        """Generate per-venue analysis"""
        venue_analysis = {}
        
        for venue in self.latency_simulator.venues:
            stats = self.latency_simulator.get_venue_latency_stats(venue, timeframe_minutes)
            
            if stats:
                venue_analysis[venue] = {
                    'performance_metrics': {
                        'mean_latency_us': stats['mean_us'],
                        'p95_latency_us': stats['p95_us'],
                        'p99_latency_us': stats['p99_us'],
                        'latency_std_us': stats['std_us']
                    },
                    'component_breakdown': {
                        'network_contribution_pct': stats.get('network_latency_us_contribution_pct', 0),
                        'queue_contribution_pct': stats.get('queue_delay_us_contribution_pct', 0),
                        'exchange_contribution_pct': stats.get('exchange_delay_us_contribution_pct', 0)
                    },
                    'queue_health': {
                        'utilization_pct': stats.get('queue_capacity_utilization', 0) * 100,
                        'current_delay_ms': stats.get('queue_current_delay_ms', 0),
                        'queue_full_events': stats.get('queue_queue_full_events', 0)
                    }
                }
        
        return venue_analysis
    
    def _generate_component_breakdown(self, timeframe_minutes: int) -> Dict[str, Any]:
        """Analyze latency by component"""
        recent_latencies = [
            b for b in self.latency_simulator.latency_history
            if time.time() - b.timestamp < timeframe_minutes * 60
        ]
        
        if not recent_latencies:
            return {}
        
        components = ['network_latency_us', 'queue_delay_us', 'exchange_delay_us', 'processing_delay_us']
        breakdown = {}
        
        for component in components:
            values = [getattr(b, component) for b in recent_latencies]
            breakdown[component] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'p95': np.percentile(values, 95),
                'contribution_pct': np.mean(values) / np.mean([b.total_latency_us for b in recent_latencies]) * 100
            }
        
        return breakdown
    
    def _generate_prediction_analysis(self) -> Dict[str, Any]:
        """Analyze ML prediction performance"""
        return self.latency_simulator.get_prediction_accuracy_stats()
    
    def _generate_cost_impact_analysis(self) -> Dict[str, Any]:
        """Analyze cost impact of latency"""
        return self.execution_engine.get_latency_cost_analysis()
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check prediction accuracy
        pred_stats = self.latency_simulator.get_prediction_accuracy_stats()
        if pred_stats.get('prediction_within_10pct', 0) < 75:
            recommendations.append("ML latency prediction accuracy is below 75%. Consider retraining models.")
        
        # Check congestion levels
        congestion = self.latency_simulator.get_congestion_analysis()
        if congestion['current_congestion_level'] in ['high', 'critical']:
            recommendations.append("Current network congestion is high. Consider routing to less congested venues.")
        
        # Check venue performance
        rankings = self.execution_engine.get_venue_latency_rankings()
        if len(rankings) > 1:
            best_venue, best_latency = rankings[0]
            worst_venue, worst_latency = rankings[-1]
            
            if worst_latency > best_latency * 1.5:
                recommendations.append(f"Consider avoiding {worst_venue} (avg {worst_latency:.0f}μs) in favor of {best_venue} (avg {best_latency:.0f}μs)")
        
        # Check queue issues
        for venue in self.latency_simulator.venues:
            queue_stats = self.latency_simulator.message_queues[venue].get_queue_stats()
            if queue_stats['capacity_utilization'] > 0.8:
                recommendations.append(f"Queue utilization at {venue} is {queue_stats['capacity_utilization']:.1%}. Consider reducing order flow.")
        
        return recommendations
    
    def export_detailed_data(self, filename: str = None) -> Dict[str, Any]:
        """Export detailed latency data for further analysis"""
        data = {
            'latency_breakdown_data': self.latency_simulator.export_latency_data(),
            'execution_statistics': self.execution_engine.get_enhanced_execution_stats(),
            'venue_rankings': self.execution_engine.get_venue_latency_rankings(),
            'cost_analysis': self.execution_engine.get_latency_cost_analysis(),
            'configuration': create_latency_configuration()
        }
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        return data